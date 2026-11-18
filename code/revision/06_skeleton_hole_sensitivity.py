#!/usr/bin/env python
"""Crack500 skeleton-length sensitivity to small interior holes.

Purpose
-------
Diagnose whether the unusually large skeleton-length error for Dice + clDice is
partly driven by small enclosed holes in predicted crack masks. Skeletonizing a
hole creates an additional loop and can increase measured skeleton length even
when foreground area changes only modestly.

This script:
  * never retrains a model;
  * never reselects a threshold;
  * uses each run's frozen validation-selected threshold from manifest.json;
  * uses the frozen Crack500 test split from test_files.txt;
  * fills the same hole-size class in prediction and ground truth;
  * reports raw, <=4 px, <=16 px, <=64 px, and all-hole sensitivity;
  * runs all six losses for U-Net/ResNet-34 seeds 0-4 by default;
  * can optionally include the DeepLabV3+/ResNet-50 seeds 0-2 extension.

The filled-hole results are a DIAGNOSTIC SENSITIVITY ANALYSIS. They do not
replace the frozen primary skeleton-length metric.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from tqdm import tqdm

PAPER_ROOT = Path(r"C:\Users\v32t225\Desktop\Review Paper 3")
DATASET_ROOT = PAPER_ROOT / "Datasets" / "Crack500"
TEST_MASK_DIR = DATASET_ROOT / "testcrop" / "testcrop"
ORIGINAL_UNET_ROOT = PAPER_ROOT / "Output" / "OUTPUT Crack500 Full 100 Epochs"
EXTRA_UNET_ROOT = PAPER_ROOT / "Output" / "Revision" / "UNet_ResNet34_Crack500_Seeds3_4"
DEEPLAB_ROOT = PAPER_ROOT / "Output" / "Revision" / "DeepLabV3Plus_ResNet50_Crack500"
OUT = PAPER_ROOT / "Output" / "Revision_Analysis" / "Skeleton_Hole_Sensitivity"
FIG_OUT = PAPER_ROOT / "Figures" / "Revision_Hole_Sensitivity"

LOSS_ORDER = ["bce", "focal", "bce_dice", "focal_tversky", "dice_boundary", "dice_cldice"]
HOLE_LIMITS: Tuple[Optional[int], ...] = (0, 4, 16, 64, None)
FOUR = ndi.generate_binary_structure(2, 1)
EPS = 1e-12


@dataclass(frozen=True)
class SplitItem:
    image_name: str
    mask_name: str

    @property
    def stem(self) -> str:
        return Path(self.image_name).stem


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    loss: str
    seed: int
    run_dir: Path


def read_split_file(path: Path) -> List[SplitItem]:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen split list: {path}")
    items: List[SplitItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"Bad split-list line in {path}: {line}")
        items.append(SplitItem(parts[0], parts[1]))
    if len(items) != 1124:
        raise RuntimeError(f"Crack500 frozen test split should contain 1124 items; found {len(items)}")
    return items


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
    attempted = "\n    ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Could not resolve {listed_name!r}. Attempted:\n    {attempted}")


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        m = np.asarray(im.convert("L"), dtype=np.uint8)
    unique = np.unique(m)
    if np.all(np.isin(unique, [0, 1])):
        return m > 0
    if np.all(np.isin(unique, [0, 255])):
        return m > 127
    if np.all((unique <= 15) | (unique >= 240)):
        return m > 127
    raise ValueError(f"Unexpected ground-truth encoding in {path}: {unique[:20]}")


def load_prob(path: Path) -> np.ndarray:
    p = np.load(path, allow_pickle=False)
    if p.ndim != 2:
        raise ValueError(f"Probability map must be HxW: {path}, got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError(f"Non-finite probability values: {path}")
    if float(p.min()) < -1e-6 or float(p.max()) > 1.0 + 1e-6:
        raise ValueError(f"Probability values outside [0,1]: {path}")
    return p.astype(np.float32, copy=False)


def read_threshold(run_dir: Path) -> float:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: {run_dir}")
    threshold = float(manifest["best_val_threshold"])
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Invalid threshold in {manifest_path}: {threshold}")
    return threshold


def skel_len(mask: np.ndarray) -> int:
    return int(skeletonize(mask).sum()) if mask.any() else 0


def hole_labels(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    filled = ndi.binary_fill_holes(mask)
    holes = np.logical_and(filled, ~mask)
    lab, n = ndi.label(holes, structure=FOUR)
    if n == 0:
        return lab, np.asarray([], dtype=np.int64)
    sizes = np.bincount(lab.ravel())[1:].astype(np.int64, copy=False)
    return lab, sizes


def fill_by_limit(mask: np.ndarray, lab: np.ndarray, sizes: np.ndarray, limit: Optional[int]) -> Tuple[np.ndarray, int]:
    if limit == 0 or sizes.size == 0:
        return mask.copy(), 0
    if limit is None:
        ids = np.arange(1, sizes.size + 1, dtype=np.int64)
    else:
        ids = np.flatnonzero(sizes <= int(limit)).astype(np.int64) + 1
    if ids.size == 0:
        return mask.copy(), 0
    return np.logical_or(mask, np.isin(lab, ids)), int(ids.size)


def condition_name(limit: Optional[int]) -> str:
    if limit == 0:
        return "raw"
    if limit is None:
        return "fill_all"
    return f"fill_le_{limit}px"


def run_specs(architectures: Sequence[str], seeds: Sequence[int]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    if "unet" in architectures:
        for loss in LOSS_ORDER:
            for seed in seeds:
                if seed <= 2:
                    root = ORIGINAL_UNET_ROOT
                elif seed in (3, 4):
                    root = EXTRA_UNET_ROOT
                else:
                    raise ValueError(f"U-Net seed {seed} is unavailable. Expected 0-4.")
                specs.append(RunSpec("UNet_ResNet34", loss, seed, root / f"{loss}_seed{seed}"))
    if "deeplab" in architectures:
        for loss in LOSS_ORDER:
            for seed in (0, 1, 2):
                specs.append(RunSpec("DeepLabV3Plus_ResNet50", loss, seed, DEEPLAB_ROOT / f"{loss}_seed{seed}"))
    return specs


def precompute_ground_truth(items: Sequence[SplitItem]) -> Dict[str, dict]:
    gt_info: Dict[str, dict] = {}
    for item in tqdm(items, desc="Precomputing Crack500 GT hole sensitivity", unit="mask"):
        mask_path = resolve_split_path(TEST_MASK_DIR, item.mask_name)
        gt = load_mask(mask_path)
        lab, sizes = hole_labels(gt)
        row = {
            "mask_path": str(mask_path),
            "area_raw": int(gt.sum()),
            "shape": [int(gt.shape[0]), int(gt.shape[1])],
            "holes_total": int(sizes.size),
        }
        for limit in HOLE_LIMITS:
            filled, nfilled = fill_by_limit(gt, lab, sizes, limit)
            name = condition_name(limit)
            row[f"skel_{name}"] = skel_len(filled)
            row[f"holes_filled_{name}"] = nfilled
        gt_info[item.stem] = row
    return gt_info


def score_one_run(spec: RunSpec, items: Sequence[SplitItem], gt_info: Dict[str, dict]) -> List[dict]:
    threshold = read_threshold(spec.run_dir)
    prob_dir = spec.run_dir / "probmaps_test"
    if not prob_dir.exists():
        raise FileNotFoundError(f"Missing test probability maps: {prob_dir}")

    rows: List[dict] = []
    for item in tqdm(items, desc=f"{spec.architecture} {spec.loss} s{spec.seed}", unit="img", leave=False):
        p = load_prob(prob_dir / f"{item.stem}.npy")
        pred = p >= threshold
        info = gt_info[item.stem]
        if tuple(p.shape) != tuple(info["shape"]):
            raise ValueError(f"Shape mismatch for {item.stem}: prob={p.shape}, mask={tuple(info['shape'])}")

        lab, sizes = hole_labels(pred)
        area_pred_raw = int(pred.sum())
        area_gt_raw = int(info["area_raw"])
        for limit in HOLE_LIMITS:
            name = condition_name(limit)
            pred_filled, nfilled_pred = fill_by_limit(pred, lab, sizes, limit)
            pred_len = skel_len(pred_filled)
            gt_len = int(info[f"skel_{name}"])
            abs_err = np.nan if gt_len <= 0 else 100.0 * abs(pred_len - gt_len) / gt_len
            rows.append({
                "architecture": spec.architecture,
                "loss": spec.loss,
                "seed": spec.seed,
                "image": item.stem,
                "threshold": threshold,
                "hole_condition": name,
                "hole_limit_px": np.nan if limit is None else int(limit),
                "pred_area_raw": area_pred_raw,
                "gt_area_raw": area_gt_raw,
                "pred_to_gt_area_ratio_raw": (area_pred_raw / area_gt_raw) if area_gt_raw > 0 else np.nan,
                "pred_holes_total": int(sizes.size),
                "gt_holes_total": int(info["holes_total"]),
                "pred_holes_filled": nfilled_pred,
                "gt_holes_filled": int(info[f"holes_filled_{name}"]),
                "skeleton_length_pred": pred_len,
                "skeleton_length_gt": gt_len,
                "skeleton_length_abs_error_pct": abs_err,
            })
    return rows


def summarize(per_image: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: List[dict] = []
    for keys, g in per_image.groupby(["architecture", "loss", "seed", "hole_condition"], sort=False):
        arch, loss, seed, condition = keys
        valid = g[g.skeleton_length_gt > 0]
        pred_sum = float(valid.skeleton_length_pred.sum())
        gt_sum = float(valid.skeleton_length_gt.sum())
        pooled = 100.0 * abs(pred_sum - gt_sum) / gt_sum if gt_sum > 0 else np.nan
        run_rows.append({
            "architecture": arch,
            "loss": loss,
            "seed": int(seed),
            "hole_condition": condition,
            "n_images": len(g),
            "pooled_skeleton_length_error_pct": pooled,
            "pred_to_gt_skeleton_length_ratio": pred_sum / gt_sum if gt_sum > 0 else np.nan,
            "per_image_skeleton_error_median_pct": float(valid.skeleton_length_abs_error_pct.median()),
            "per_image_skeleton_error_q25_pct": float(valid.skeleton_length_abs_error_pct.quantile(.25)),
            "per_image_skeleton_error_q75_pct": float(valid.skeleton_length_abs_error_pct.quantile(.75)),
            "median_pred_holes_total": float(g.pred_holes_total.median()),
            "median_gt_holes_total": float(g.gt_holes_total.median()),
            "mean_pred_to_gt_area_ratio_raw": float(g.pred_to_gt_area_ratio_raw.mean()),
        })
    run_df = pd.DataFrame(run_rows)

    loss_df = run_df.groupby(["architecture", "loss", "hole_condition"], as_index=False).agg(
        pooled_error_mean_pct=("pooled_skeleton_length_error_pct", "mean"),
        pooled_error_std_pct=("pooled_skeleton_length_error_pct", "std"),
        length_ratio_mean=("pred_to_gt_skeleton_length_ratio", "mean"),
        length_ratio_std=("pred_to_gt_skeleton_length_ratio", "std"),
        per_image_median_error_mean_pct=("per_image_skeleton_error_median_pct", "mean"),
        median_pred_holes=("median_pred_holes_total", "mean"),
        mean_area_ratio=("mean_pred_to_gt_area_ratio_raw", "mean"),
        n_seeds=("seed", "nunique"),
    )
    return run_df, loss_df


def mechanism_table(loss_df: pd.DataFrame) -> pd.DataFrame:
    raw = loss_df[loss_df.hole_condition == "raw"].copy()
    f64 = loss_df[loss_df.hole_condition == "fill_le_64px"].copy()
    allf = loss_df[loss_df.hole_condition == "fill_all"].copy()
    merged = raw.merge(f64, on=["architecture", "loss"], suffixes=("_raw", "_fill64"))
    merged = merged.merge(
        allf[["architecture", "loss", "pooled_error_mean_pct", "length_ratio_mean"]],
        on=["architecture", "loss"], how="left"
    ).rename(columns={
        "pooled_error_mean_pct": "pooled_error_mean_pct_fill_all",
        "length_ratio_mean": "length_ratio_mean_fill_all",
    })
    merged["pooled_error_reduction_fill64_pp"] = (
        merged.pooled_error_mean_pct_raw - merged.pooled_error_mean_pct_fill64
    )
    merged["pooled_error_fraction_removed_fill64"] = np.where(
        merged.pooled_error_mean_pct_raw.abs() > EPS,
        merged.pooled_error_reduction_fill64_pp / merged.pooled_error_mean_pct_raw,
        np.nan,
    )
    keep = [
        "architecture", "loss", "n_seeds_raw", "mean_area_ratio_raw",
        "median_pred_holes_raw", "pooled_error_mean_pct_raw",
        "pooled_error_mean_pct_fill64", "pooled_error_reduction_fill64_pp",
        "pooled_error_fraction_removed_fill64", "pooled_error_mean_pct_fill_all",
        "length_ratio_mean_raw", "length_ratio_mean_fill64", "length_ratio_mean_fill_all",
    ]
    return merged[keep]


def make_plot(loss_df: pd.DataFrame, architecture: str) -> None:
    sub = loss_df[loss_df.architecture == architecture].copy()
    if sub.empty:
        return
    order = [condition_name(x) for x in HOLE_LIMITS]
    labels = ["Raw", "≤4", "≤16", "≤64", "All"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    for loss in LOSS_ORDER:
        g = sub[sub.loss == loss].set_index("hole_condition").reindex(order)
        ax.plot(labels, g.pooled_error_mean_pct, marker="o", label=loss.replace("_", "+"))
    ax.set_xlabel("Interior-hole filling sensitivity")
    ax.set_ylabel("Dataset-level skeleton-length error (%)")
    ax.set_title(f"Crack500 skeleton-length sensitivity: {architecture}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT / f"Skeleton_Hole_Sensitivity_{architecture}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--architectures", nargs="+", choices=["unet", "deeplab"], default=["unet"])
    p.add_argument("--unet-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    split_source = ORIGINAL_UNET_ROOT / "test_files.txt"
    items = read_split_file(split_source)
    gt_info = precompute_ground_truth(items)
    specs = run_specs(args.architectures, args.unet_seeds)

    all_rows: List[dict] = []
    for spec in specs:
        all_rows.extend(score_one_run(spec, items, gt_info))

    per_image = pd.DataFrame(all_rows)
    run_df, loss_df = summarize(per_image)
    mech = mechanism_table(loss_df)

    per_image.to_csv(OUT / "Skeleton_Hole_Sensitivity_Per_Image.csv", index=False)
    run_df.to_csv(OUT / "Skeleton_Hole_Sensitivity_Run_Level.csv", index=False)
    loss_df.to_csv(OUT / "Skeleton_Hole_Sensitivity_Loss_Level.csv", index=False)
    mech.to_csv(OUT / "Skeleton_Hole_Mechanism_Table.csv", index=False)

    for arch in per_image.architecture.unique():
        make_plot(loss_df, arch)

    manifest = {
        "dataset": "Crack500",
        "frozen_test_n": len(items),
        "threshold_policy": "Frozen validation-selected threshold from each run manifest; never reselected.",
        "foreground_connectivity": 8,
        "hole_connectivity": 4,
        "hole_fill_conditions": [condition_name(x) for x in HOLE_LIMITS],
        "symmetric_rule": "The same maximum enclosed-hole area is filled in predictions and ground truth.",
        "interpretation": "Sensitivity diagnostic only. The submitted raw skeleton metric remains the primary frozen metric.",
        "architectures": list(per_image.architecture.unique()),
        "losses": LOSS_ORDER,
        "unet_seeds_requested": args.unet_seeds,
    }
    (OUT / "skeleton_hole_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSkeleton-hole sensitivity complete.")
    print(mech.to_string(index=False))
    print(f"\nOutputs: {OUT}")
    print(f"Figure(s): {FIG_OUT}")


if __name__ == "__main__":
    main()
