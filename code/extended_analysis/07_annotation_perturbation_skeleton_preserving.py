#!/usr/bin/env python
"""Corrected annotation-uncertainty analysis with skeleton-preserving narrowing.

Why this v2 exists
------------------
Plain binary erosion is degenerate for very thin annotations. In the first
revision analysis, CrackTree260 was entirely erased by 1-px erosion and CFD
lost some masks under 2-px erosion. Those erosion cells must not be used.

This corrected analysis replaces destructive erosion with a controlled
skeleton-preserving narrowing operation:

    narrow_r(G) = erosion_r(G) OR skeleton(G)

The perturbation remains a subset of the original annotation, preserves the
original centerline support, and cannot erase a non-empty crack mask solely
because its label is one pixel wide. Dilation conditions are unchanged.

No model is retrained. Each run keeps its original validation-selected
threshold. The analysis is diagnostic and does not claim to reproduce true
independent annotator disagreement.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize
from tqdm import tqdm

PAPER_ROOT = Path(os.environ.get("CRACKSEG_ROOT", Path.cwd())).expanduser().resolve()
DATASET_BASE = Path(os.environ.get("CRACKSEG_DATASETS", PAPER_ROOT / "Datasets")).expanduser().resolve()
OUTPUT_BASE = Path(os.environ.get("CRACKSEG_OUTPUT", PAPER_ROOT / "Output")).expanduser().resolve()
OUT = OUTPUT_BASE / "Extended_Analysis" / "Annotation_Uncertainty_v2_SkeletonPreserving"

LOSS_ORDER = ["bce", "focal", "bce_dice", "focal_tversky", "dice_boundary", "dice_cldice"]
STRUCT8 = np.ones((3, 3), dtype=bool)
EPS = 1e-12
BOUNDARY_THETA = 2
PERTURBATIONS = ["original", "narrow1_skeleton_preserving", "dilate1", "narrow2_skeleton_preserving", "dilate2"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    output_root: Path
    test_mask_dir: Path
    expected_test: int


DATASETS: Dict[str, DatasetSpec] = {
    "Crack500": DatasetSpec(
        "Crack500", DATASET_BASE / "Crack500",
        OUTPUT_BASE / "OUTPUT Crack500 Full 100 Epochs",
        DATASET_BASE / "Crack500" / "testcrop" / "testcrop", 1124,
    ),
    "DeepCrack": DatasetSpec(
        "DeepCrack", DATASET_BASE / "DeepCrack Dataset",
        OUTPUT_BASE / "OUTPUT DeepCrack Full 100 Epochs",
        DATASET_BASE / "DeepCrack Dataset" / "test_lab", 237,
    ),
    "CFD": DatasetSpec(
        "CFD", DATASET_BASE / "CrackForest-dataset-master",
        OUTPUT_BASE / "OUTPUT CFD Full 100 Epochs",
        DATASET_BASE / "CrackForest-dataset-master" / "seg", 24,
    ),
    "CrackTree260": DatasetSpec(
        "CrackTree260", DATASET_BASE / "CrackTree260",
        OUTPUT_BASE / "OUTPUT CrackTree260 Full 100 Epochs",
        DATASET_BASE / "CrackTree260" / "gt", 52,
    ),
}


@dataclass(frozen=True)
class SplitItem:
    image_name: str
    mask_name: str

    @property
    def stem(self) -> str:
        return Path(self.image_name).stem


def read_split_file(path: Path) -> List[SplitItem]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split list: {path}")
    rows: List[SplitItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"Bad split-list line in {path}: {line}")
        rows.append(SplitItem(parts[0], parts[1]))
    return rows


def resolve_split_path(base_dir: Path, listed_name: str) -> Path:
    raw = str(listed_name).strip().strip('"').strip("'")
    p = Path(raw.replace("\\", "/"))
    candidates: List[Path] = []

    def add(x: Path) -> None:
        if x not in candidates:
            candidates.append(x)

    if p.is_absolute():
        add(p)
    else:
        add(base_dir / p)
        bp = [x.casefold() for x in base_dir.parts]
        pp = list(p.parts)
        ppl = [x.casefold() for x in pp]
        for overlap in range(min(len(bp), max(0, len(ppl) - 1)), 0, -1):
            if bp[-overlap:] == ppl[:overlap]:
                rest = pp[overlap:]
                if rest:
                    add(base_dir / Path(*rest))
                break
        for parent in base_dir.parents:
            add(parent / p)
        add(base_dir / p.name)
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Could not resolve {listed_name!r} from {base_dir}")


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        m = np.asarray(im.convert("L"), dtype=np.uint8)
    u = np.unique(m)
    if np.all(np.isin(u, [0, 1])):
        return m > 0
    if np.all(np.isin(u, [0, 255])):
        return m > 127
    if np.all((u <= 15) | (u >= 240)):
        return m > 127
    raise ValueError(f"Unexpected mask encoding {path}: {u[:20]}")


def load_prob(path: Path) -> np.ndarray:
    p = np.load(path, allow_pickle=False)
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError(f"Invalid probability map: {path}")
    if float(p.min()) < -1e-6 or float(p.max()) > 1.0 + 1e-6:
        raise ValueError(f"Probability outside [0,1]: {path}")
    return p.astype(np.float32, copy=False)


def skeleton_preserving_narrow(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Erode a mask while restoring its original skeleton.

    The returned mask is a subset of the original mask because the skeleton is
    itself contained in the original foreground. A non-empty crack annotation
    therefore remains non-empty under this perturbation.
    """
    if iterations <= 0 or not mask.any():
        return mask.copy()
    skel = skeletonize(mask).astype(bool)
    eroded = binary_erosion(mask, structure=STRUCT8, iterations=iterations, border_value=0)
    return np.logical_or(eroded, skel)


def perturb(gt: np.ndarray, name: str) -> np.ndarray:
    if name == "original":
        return gt.copy()
    if name == "narrow1_skeleton_preserving":
        return skeleton_preserving_narrow(gt, 1)
    if name == "narrow2_skeleton_preserving":
        return skeleton_preserving_narrow(gt, 2)
    if name == "dilate1":
        return binary_dilation(gt, structure=STRUCT8, iterations=1)
    if name == "dilate2":
        return binary_dilation(gt, structure=STRUCT8, iterations=2)
    raise ValueError(name)


def confusion(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int]:
    return int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())


def global_f1(tp: int, fp: int, fn: int) -> float:
    return 1.0 if 2 * tp + fp + fn == 0 else float(2 * tp / (2 * tp + fp + fn + EPS))


def empty_case(pred: np.ndarray, gt: np.ndarray) -> int:
    pa, ga = bool(pred.any()), bool(gt.any())
    if not pa and not ga:
        return 0
    if pa and ga:
        return 3
    return 1 if pa else 2


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_and(mask, ~binary_erosion(mask, structure=STRUCT8, border_value=0))


def boundary_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    c = empty_case(pred, gt)
    if c == 0:
        return 1.0
    if c != 3:
        return 0.0
    pb, gb = boundary_pixels(pred), boundary_pixels(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    d_g = distance_transform_edt(~gb)
    d_p = distance_transform_edt(~pb)
    mp = int((pb & (d_g <= BOUNDARY_THETA)).sum())
    mg = int((gb & (d_p <= BOUNDARY_THETA)).sum())
    p = mp / (pb.sum() + EPS)
    r = mg / (gb.sum() + EPS)
    return float(2 * p * r / (p + r + EPS))


def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    c = empty_case(pred, gt)
    if c == 0:
        return 1.0
    if c != 3:
        return 0.0
    sp, sg = skeletonize(pred).astype(bool), skeletonize(gt).astype(bool)
    if not sp.any() or not sg.any():
        return 0.0
    tprec = (sp & gt).sum() / (sp.sum() + EPS)
    tsens = (sg & pred).sum() / (sg.sum() + EPS)
    return float(2 * tprec * tsens / (tprec + tsens + EPS))


def abs_rel_pct(a: float, b: float) -> float:
    return float(abs(a - b) / b * 100.0) if b > 0 else float("nan")


def run_threshold(run_dir: Path) -> float:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Incomplete run: {run_dir}")
    return float(manifest["best_val_threshold"])


def thickness_analysis(spec: DatasetSpec, items: Sequence[SplitItem]) -> Tuple[dict, List[dict]]:
    per: List[dict] = []
    total_area = 0
    total_skel = 0
    local_diameters: List[np.ndarray] = []
    for item in tqdm(items, desc=f"{spec.name} thickness", unit="mask", leave=False):
        gt = load_mask(resolve_split_path(spec.test_mask_dir, item.mask_name))
        sk = skeletonize(gt).astype(bool)
        area = int(gt.sum())
        length = int(sk.sum())
        total_area += area
        total_skel += length
        ratio = float(area / length) if length > 0 else np.nan
        dt = distance_transform_edt(gt)
        local_d = 2.0 * dt[sk] if sk.any() else np.asarray([], dtype=float)
        if local_d.size:
            local_diameters.append(local_d.astype(np.float32))
        per.append({
            "dataset": spec.name,
            "image": item.stem,
            "gt_area_px": area,
            "gt_skeleton_length_px": length,
            "area_over_skeleton_px": ratio,
            "dt_local_diameter_median_px": float(np.median(local_d)) if local_d.size else np.nan,
        })
    df = pd.DataFrame(per)
    merged = np.concatenate(local_diameters) if local_diameters else np.asarray([np.nan])
    summary = {
        "dataset": spec.name,
        "n_images": len(items),
        "pooled_area_over_skeleton_px": float(total_area / total_skel) if total_skel else np.nan,
        "per_image_area_over_skeleton_median_px": float(df.area_over_skeleton_px.median()),
        "per_image_area_over_skeleton_q25_px": float(df.area_over_skeleton_px.quantile(.25)),
        "per_image_area_over_skeleton_q75_px": float(df.area_over_skeleton_px.quantile(.75)),
        "skeleton_local_dt_diameter_median_px": float(np.nanmedian(merged)),
        "skeleton_local_dt_diameter_q25_px": float(np.nanquantile(merged, .25)),
        "skeleton_local_dt_diameter_q75_px": float(np.nanquantile(merged, .75)),
        "interpretation": "annotation-mask thickness proxy only; not physical crack width",
    }
    return summary, per


def annotation_analysis(spec: DatasetSpec, items: Sequence[SplitItem]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summaries: List[dict] = []
    per_rows: List[dict] = []

    for loss in LOSS_ORDER:
        for seed in (0, 1, 2):
            run_dir = spec.output_root / f"{loss}_seed{seed}"
            threshold = run_threshold(run_dir)
            prob_dir = run_dir / "probmaps_test"
            acc = {
                name: {
                    "tp": 0, "fp": 0, "fn": 0,
                    "bfs": [], "cds": [], "aes": [], "les": [],
                    "changed_frac": [], "foreground_retention": [],
                    "original_nonempty": 0,
                    "nonempty_became_empty": 0,
                }
                for name in PERTURBATIONS
            }

            for item in tqdm(items, desc=f"{spec.name} {loss} s{seed} v2", unit="img", leave=False):
                gt0 = load_mask(resolve_split_path(spec.test_mask_dir, item.mask_name))
                p = load_prob(prob_dir / f"{item.stem}.npy")
                if p.shape != gt0.shape:
                    raise ValueError(f"Shape mismatch {spec.name}/{item.stem}: {p.shape} vs {gt0.shape}")
                pred = p >= threshold
                sp = skeletonize(pred).astype(bool)
                original_nonempty = bool(gt0.any())

                for pert_name in PERTURBATIONS:
                    gt = perturb(gt0, pert_name)
                    tp, fp, fn = confusion(pred, gt)
                    a = acc[pert_name]
                    a["tp"] += tp
                    a["fp"] += fp
                    a["fn"] += fn
                    if original_nonempty:
                        a["original_nonempty"] += 1
                        if not gt.any():
                            a["nonempty_became_empty"] += 1

                    bf = boundary_f1(pred, gt)
                    cd = cldice(pred, gt)
                    sg = skeletonize(gt).astype(bool)
                    ae = abs_rel_pct(pred.sum(), gt.sum())
                    le = abs_rel_pct(sp.sum(), sg.sum())
                    if np.isfinite(ae):
                        a["aes"].append(ae)
                    if np.isfinite(le):
                        a["les"].append(le)

                    changed = float(np.logical_xor(gt0, gt).sum() / gt0.size)
                    retention = float(gt.sum() / gt0.sum()) if gt0.any() else np.nan
                    a["changed_frac"].append(changed)
                    if np.isfinite(retention):
                        a["foreground_retention"].append(retention)
                    a["bfs"].append(bf)
                    a["cds"].append(cd)

                    per_rows.append({
                        "dataset": spec.name,
                        "loss": loss,
                        "seed": seed,
                        "perturbation": pert_name,
                        "image": item.stem,
                        "threshold_frozen_from_original_validation": threshold,
                        "boundary_f1": bf,
                        "cldice": cd,
                        "area_abs_error_pct": ae,
                        "skeleton_length_abs_error_pct": le,
                        "gt_changed_pixel_fraction": changed,
                        "gt_foreground_retention_ratio": retention,
                        "gt_original_nonempty": int(original_nonempty),
                        "gt_empty_after_perturbation": int(not gt.any()),
                        "gt_nonempty_became_empty": int(original_nonempty and not gt.any()),
                    })

            for pert_name in PERTURBATIONS:
                a = acc[pert_name]
                summaries.append({
                    "dataset": spec.name,
                    "loss": loss,
                    "seed": seed,
                    "perturbation": pert_name,
                    "threshold": threshold,
                    "f1_global": global_f1(a["tp"], a["fp"], a["fn"]),
                    "boundary_f1_mean": float(np.mean(a["bfs"])),
                    "cldice_mean": float(np.mean(a["cds"])),
                    "area_abs_error_pct_median": float(np.median(a["aes"])) if a["aes"] else np.nan,
                    "skeleton_length_abs_error_pct_median": float(np.median(a["les"])) if a["les"] else np.nan,
                    "gt_changed_pixel_fraction_mean": float(np.mean(a["changed_frac"])),
                    "gt_foreground_retention_ratio_mean": float(np.mean(a["foreground_retention"])) if a["foreground_retention"] else np.nan,
                    "gt_original_nonempty_count": int(a["original_nonempty"]),
                    "gt_nonempty_became_empty_count": int(a["nonempty_became_empty"]),
                })

    return pd.DataFrame(summaries), pd.DataFrame(per_rows)


def rank_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = run_df.groupby(["dataset", "perturbation", "loss"], as_index=False).agg({
        "f1_global": "mean",
        "boundary_f1_mean": "mean",
        "cldice_mean": "mean",
        "area_abs_error_pct_median": "mean",
        "skeleton_length_abs_error_pct_median": "mean",
    })
    rows: List[dict] = []
    for (dataset, pert), g in grouped.groupby(["dataset", "perturbation"]):
        for metric in ["f1_global", "boundary_f1_mean", "cldice_mean"]:
            gg = g.sort_values(metric, ascending=False).reset_index(drop=True)
            for i, r in gg.iterrows():
                rows.append({"dataset": dataset, "perturbation": pert, "metric": metric, "loss": r.loss, "value": r[metric], "rank": i + 1})
        for metric in ["area_abs_error_pct_median", "skeleton_length_abs_error_pct_median"]:
            gg = g.sort_values(metric, ascending=True).reset_index(drop=True)
            for i, r in gg.iterrows():
                rows.append({"dataset": dataset, "perturbation": pert, "metric": metric, "loss": r.loss, "value": r[metric], "rank": i + 1})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--skip-thickness", action="store_true", help="Reuse prior thickness results if desired.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    thickness_summaries: List[dict] = []
    thickness_per: List[dict] = []
    perturb_summaries: List[pd.DataFrame] = []
    perturb_per: List[pd.DataFrame] = []

    for name in args.datasets:
        spec = DATASETS[name]
        items = read_split_file(spec.output_root / "test_files.txt")
        if len(items) != spec.expected_test:
            raise RuntimeError(f"{name}: expected {spec.expected_test} test items, found {len(items)}")

        if not args.skip_thickness:
            ts, tp = thickness_analysis(spec, items)
            thickness_summaries.append(ts)
            thickness_per.extend(tp)

        rs, rp = annotation_analysis(spec, items)
        perturb_summaries.append(rs)
        perturb_per.append(rp)

    if thickness_summaries:
        pd.DataFrame(thickness_summaries).to_csv(OUT / "Dataset_Annotation_Thickness_Proxies.csv", index=False)
        pd.DataFrame(thickness_per).to_csv(OUT / "Per_Image_Annotation_Thickness_Proxies.csv", index=False)

    run_df = pd.concat(perturb_summaries, ignore_index=True)
    per_df = pd.concat(perturb_per, ignore_index=True)
    run_df.to_csv(OUT / "Annotation_Perturbation_Run_Level_v2.csv", index=False)
    per_df.to_csv(OUT / "Annotation_Perturbation_Per_Image_v2.csv", index=False)

    ranks = rank_summary(run_df)
    ranks.to_csv(OUT / "Annotation_Perturbation_Loss_Ranks_v2.csv", index=False)

    compact = run_df.groupby(["dataset", "perturbation", "loss"], as_index=False).agg(
        f1_global_mean=("f1_global", "mean"),
        f1_global_std=("f1_global", "std"),
        boundary_f1_mean=("boundary_f1_mean", "mean"),
        cldice_mean=("cldice_mean", "mean"),
        area_error_median_mean=("area_abs_error_pct_median", "mean"),
        skeleton_length_error_median_mean=("skeleton_length_abs_error_pct_median", "mean"),
        gt_changed_fraction=("gt_changed_pixel_fraction_mean", "mean"),
        gt_foreground_retention=("gt_foreground_retention_ratio_mean", "mean"),
        nonempty_became_empty=("gt_nonempty_became_empty_count", "max"),
    )
    compact.to_csv(OUT / "Supplementary_Table_Annotation_Perturbation_v2.csv", index=False)

    invalid = run_df[run_df.gt_nonempty_became_empty_count > 0]
    if not invalid.empty:
        raise RuntimeError(
            "Skeleton-preserving narrowing unexpectedly erased a non-empty annotation. "
            f"See rows:\n{invalid[['dataset','loss','seed','perturbation','gt_nonempty_became_empty_count']].to_string(index=False)}"
        )

    manifest = {
        "version": "v2_skeleton_preserving_narrowing",
        "deprecated_v1_note": "Plain erode1/erode2 cells from the first revision analysis are not valid for thin-label datasets and must not be used in the manuscript.",
        "threshold_policy": "Each original run's validation-selected threshold is frozen; no threshold reselection after GT perturbation.",
        "perturbations": PERTURBATIONS,
        "narrowing_definition": "binary_erosion(G, r) OR skeletonize(G)",
        "narrowing_property": "Output remains a subset of G and preserves the original skeleton support; a non-empty one-pixel crack annotation is not erased.",
        "morphology_structure": "3x3 8-connected square",
        "boundary_tolerance_px": BOUNDARY_THETA,
        "interpretation_limit": "Controlled annotation-convention sensitivity only; not a substitute for independently annotated data.",
        "thickness_warning": "Thickness proxies are annotation-mask thickness in pixels, not physical crack width.",
    }
    (OUT / "annotation_analysis_manifest_v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nCorrected annotation perturbation complete.")
    print("No originally non-empty GT mask became empty under skeleton-preserving narrowing.")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
