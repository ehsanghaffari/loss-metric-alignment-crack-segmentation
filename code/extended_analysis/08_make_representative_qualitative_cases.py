#!/usr/bin/env python
"""Generate representative, publication-oriented qualitative cases for the IJPRT revision.

Changes from the first qualitative script:
  * avoids selecting the maximum-error outlier;
  * documents each selected image's percentile within the relevant distribution;
  * uses ROI crops so thin crack structure is visible at journal width;
  * uses colorblind-friendly blue/orange overlays rather than red/green;
  * displays boundary/skeleton lines slightly dilated for visibility only;
  * uses the hole-sensitivity output to select a representative Dice+clDice
    example whose skeleton-length error is sensitive to filling small holes.

Metrics are always calculated on the original masks/predictions. Display-line
thickening and ROI cropping affect visualization only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import os
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import skeletonize

PAPER_ROOT = Path(os.environ.get("CRACKSEG_ROOT", Path.cwd())).expanduser().resolve()
DATASET_ROOT = PAPER_ROOT / "Datasets" / "Crack500"
TEST_MASK_DIR = DATASET_ROOT / "testcrop" / "testcrop"
OUTPUT_BASE = Path(os.environ.get("CRACKSEG_OUTPUT", PAPER_ROOT / "Output")).expanduser().resolve()
FIGURES_ROOT = Path(os.environ.get("CRACKSEG_FIGURES", PAPER_ROOT / "Figures")).expanduser().resolve()
ORIGINAL_UNET_ROOT = OUTPUT_BASE / "OUTPUT Crack500 Full 100 Epochs"
SCORING_CSV = OUTPUT_BASE / "Extended_Analysis" / "Scoring" / "per_image_complete_results.csv"
HOLE_CSV = OUTPUT_BASE / "Extended_Analysis" / "Skeleton_Hole_Sensitivity" / "Skeleton_Hole_Sensitivity_Per_Image.csv"
OUT = FIGURES_ROOT / "Extended_Qualitative_Cases"
OUT_TABLE = OUTPUT_BASE / "Extended_Analysis" / "Qualitative"

STRUCT8 = np.ones((3, 3), dtype=bool)
EPS = 1e-12
GT_COLOR = np.array([0, 114, 178], dtype=np.float32)
PRED_COLOR = np.array([230, 159, 0], dtype=np.float32)
OVERLAP_COLOR = np.array([204, 121, 167], dtype=np.float32)


@dataclass(frozen=True)
class SplitItem:
    image_name: str
    mask_name: str

    @property
    def stem(self) -> str:
        return Path(self.image_name).stem


def read_split(path: Path) -> List[SplitItem]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[SplitItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"Bad split line: {line}")
        rows.append(SplitItem(parts[0], parts[1]))
    return rows


def resolve(base_dir: Path, listed_name: str) -> Path:
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


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


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
    raise ValueError(f"Unexpected mask encoding: {path}: {u[:20]}")


def load_prob(path: Path) -> np.ndarray:
    p = np.load(path, allow_pickle=False)
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError(f"Invalid probability map: {path}")
    return p.astype(np.float32, copy=False)


def run_threshold(run_dir: Path) -> float:
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if m.get("status") != "complete":
        raise RuntimeError(f"Incomplete run: {run_dir}")
    return float(m["best_val_threshold"])


def boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_and(mask, ~binary_erosion(mask, structure=STRUCT8, border_value=0))


def display_dilate(line_mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if not line_mask.any() or iterations <= 0:
        return line_mask
    return binary_dilation(line_mask, structure=STRUCT8, iterations=iterations)


def overlay_two(image: np.ndarray, gt_line: np.ndarray, pred_line: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32).copy()
    g = display_dilate(gt_line, 1)
    p = display_dilate(pred_line, 1)
    only_g = g & ~p
    only_p = p & ~g
    both = g & p
    out[only_g] = 0.25 * out[only_g] + 0.75 * GT_COLOR
    out[only_p] = 0.25 * out[only_p] + 0.75 * PRED_COLOR
    out[both] = 0.15 * out[both] + 0.85 * OVERLAP_COLOR
    return np.clip(out, 0, 255).astype(np.uint8)


def roi_from_masks(masks: Sequence[np.ndarray], pad: int = 30, min_size: int = 192) -> Tuple[int, int, int, int]:
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= m.astype(bool)
    h, w = union.shape
    if not union.any():
        return 0, h, 0, w
    ys, xs = np.where(union)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = max(0, y0 - pad); y1 = min(h, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(w, x1 + pad)

    def expand(lo: int, hi: int, maximum: int) -> Tuple[int, int]:
        size = hi - lo
        if size >= min_size:
            return lo, hi
        need = min_size - size
        left = need // 2
        right = need - left
        lo2 = max(0, lo - left)
        hi2 = min(maximum, hi + right)
        if hi2 - lo2 < min_size:
            if lo2 == 0:
                hi2 = min(maximum, min_size)
            elif hi2 == maximum:
                lo2 = max(0, maximum - min_size)
        return lo2, hi2

    y0, y1 = expand(y0, y1, h)
    x0, x1 = expand(x0, x1, w)
    return y0, y1, x0, x1


def crop(arr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = roi
    return arr[y0:y1, x0:x1]


def percentile_rank(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float((s <= value).mean() * 100.0)


def get_item_map() -> dict:
    items = read_split(ORIGINAL_UNET_ROOT / "test_files.txt")
    return {x.stem: x for x in items}


def prediction(loss: str, seed: int, stem: str) -> Tuple[np.ndarray, float]:
    run = ORIGINAL_UNET_ROOT / f"{loss}_seed{seed}"
    t = run_threshold(run)
    return load_prob(run / "probmaps_test" / f"{stem}.npy") >= t, t


def select_case_a(scoring: pd.DataFrame, holes: pd.DataFrame) -> dict:
    base = scoring[(scoring.architecture == "UNet_ResNet34") & (scoring.loss == "dice_cldice") & (scoring.seed == 0)].copy()
    raw = holes[(holes.architecture == "UNet_ResNet34") & (holes.loss == "dice_cldice") & (holes.seed == 0) & (holes.hole_condition == "raw")][["image", "skeleton_length_abs_error_pct"]].rename(columns={"skeleton_length_abs_error_pct": "hole_raw_error"})
    f64 = holes[(holes.architecture == "UNet_ResNet34") & (holes.loss == "dice_cldice") & (holes.seed == 0) & (holes.hole_condition == "fill_le_64px")][["image", "skeleton_length_abs_error_pct"]].rename(columns={"skeleton_length_abs_error_pct": "hole_fill64_error"})
    d = base.merge(raw, on="image").merge(f64, on="image")
    d["hole_reduction_pp"] = d.hole_raw_error - d.hole_fill64_error

    cl_cut = d.cldice.quantile(.75)
    err_target = d.skeleton_length_abs_error_pct.quantile(.75)
    err_cap = d.skeleton_length_abs_error_pct.quantile(.90)
    eligible = d[(d.cldice >= cl_cut) & (d.skeleton_length_abs_error_pct <= err_cap) & (d.hole_reduction_pp > 5)].copy()
    if eligible.empty:
        eligible = d[(d.cldice >= cl_cut) & (d.skeleton_length_abs_error_pct <= err_cap)].copy()
    if eligible.empty:
        eligible = d[d.cldice >= cl_cut].copy()
    eligible["selection_distance"] = (eligible.skeleton_length_abs_error_pct - err_target).abs()
    row = eligible.sort_values(["selection_distance", "hole_reduction_pp", "cldice"], ascending=[True, False, False]).iloc[0]
    return row.to_dict()


def select_case_b(scoring: pd.DataFrame) -> dict:
    d = scoring[(scoring.architecture == "UNet_ResNet34") & (scoring.loss == "dice_cldice") & (scoring.seed == 0)].copy()
    eligible = d[(d.fragmentation_abs_error <= 1) & (d.area_pred > d.area_gt) & (d.f1 >= .4)].copy()
    target = eligible.area_abs_error_pct.quantile(.75)
    eligible["selection_distance"] = (eligible.area_abs_error_pct - target).abs()
    row = eligible.sort_values(["selection_distance", "f1"], ascending=[True, False]).iloc[0]
    return row.to_dict()


def select_case_c(scoring: pd.DataFrame) -> dict:
    d = scoring[(scoring.architecture == "UNet_ResNet34") & (scoring.seed == 0) & (scoring.loss.isin(["dice_boundary", "dice_cldice"]))].copy()
    b = d[d.loss == "dice_boundary"].set_index("image")
    c = d[d.loss == "dice_cldice"].set_index("image")
    common = b.index.intersection(c.index)
    rows = []
    for image in common:
        rb, rc = b.loc[image], c.loc[image]
        rows.append({
            "image": image,
            "f1_boundary": rb.f1,
            "f1_cldice": rc.f1,
            "cldice_boundary": rb.cldice,
            "cldice_cldice": rc.cldice,
            "f1_gap": abs(rb.f1 - rc.f1),
            "cldice_gap": abs(rb.cldice - rc.cldice),
            "area_gt": rb.area_gt,
        })
    pair = pd.DataFrame(rows)
    area_cut = pair.area_gt.quantile(.50)
    eligible = pair[(pair.f1_gap <= .02) & (pair.area_gt >= area_cut)].copy()
    if eligible.empty:
        eligible = pair[pair.f1_gap <= .03].copy()
    target = eligible.cldice_gap.quantile(.75)
    eligible["selection_distance"] = (eligible.cldice_gap - target).abs()
    row = eligible.sort_values(["selection_distance", "f1_gap"], ascending=[True, True]).iloc[0]
    return row.to_dict()


def full_image_with_roi(ax, image: np.ndarray, roi: Tuple[int, int, int, int], title: str) -> None:
    y0, y1, x0, x1 = roi
    ax.imshow(image)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=1.8, edgecolor="white"))
    ax.set_title(title)
    ax.axis("off")


def render_single_case(case_label: str, row: dict, item: SplitItem, loss: str, seed: int, filename: str, note: str) -> dict:
    image = load_rgb(resolve(DATASET_ROOT, item.image_name))
    gt = load_mask(resolve(TEST_MASK_DIR, item.mask_name))
    pred, threshold = prediction(loss, seed, item.stem)
    roi = roi_from_masks([gt, pred])

    fig, ax = plt.subplots(1, 5, figsize=(13.5, 3.1), facecolor="white")
    full_image_with_roi(ax[0], image, roi, "Input (ROI boxed)")
    ax[1].imshow(crop(gt, roi), cmap="gray", vmin=0, vmax=1); ax[1].set_title("Ground truth")
    ax[2].imshow(crop(pred, roi), cmap="gray", vmin=0, vmax=1); ax[2].set_title(f"Prediction\n{loss.replace('_','+')}")
    ax[3].imshow(crop(overlay_two(image, boundary(gt), boundary(pred)), roi)); ax[3].set_title("Boundary overlay\nGT blue, Pred orange")
    ax[4].imshow(crop(overlay_two(image, skeletonize(gt), skeletonize(pred)), roi)); ax[4].set_title("Skeleton overlay\nGT blue, Pred orange")
    for a in ax[1:]:
        a.axis("off")
    fig.suptitle(case_label, fontsize=10)
    fig.text(.5, .01, note + "  Display lines are dilated by 1 px for visibility only.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, .94])
    path = OUT / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "case": case_label,
        "image": item.stem,
        "loss": loss,
        "seed": seed,
        "threshold": threshold,
        "roi_y0": roi[0], "roi_y1": roi[1], "roi_x0": roi[2], "roi_x1": roi[3],
        "f1": row.get("f1", np.nan),
        "cldice": row.get("cldice", np.nan),
        "area_abs_error_pct": row.get("area_abs_error_pct", np.nan),
        "skeleton_length_abs_error_pct": row.get("skeleton_length_abs_error_pct", np.nan),
        "fragmentation_abs_error": row.get("fragmentation_abs_error", np.nan),
        "file": str(path),
    }


def render_case_c(row: dict, item: SplitItem, scoring: pd.DataFrame) -> dict:
    image = load_rgb(resolve(DATASET_ROOT, item.image_name))
    gt = load_mask(resolve(TEST_MASK_DIR, item.mask_name))
    pb, tb = prediction("dice_boundary", 0, item.stem)
    pc, tc = prediction("dice_cldice", 0, item.stem)
    roi = roi_from_masks([gt, pb, pc])

    fig, ax = plt.subplots(2, 4, figsize=(11.5, 6.0), facecolor="white")
    full_image_with_roi(ax[0,0], image, roi, "Input (ROI boxed)")
    ax[0,1].imshow(crop(gt, roi), cmap="gray", vmin=0, vmax=1); ax[0,1].set_title("Ground truth")
    ax[0,2].imshow(crop(pb, roi), cmap="gray", vmin=0, vmax=1); ax[0,2].set_title(f"Dice+Boundary\nF1={row['f1_boundary']:.3f}")
    ax[0,3].imshow(crop(pc, roi), cmap="gray", vmin=0, vmax=1); ax[0,3].set_title(f"Dice+clDice\nF1={row['f1_cldice']:.3f}")
    ax[1,0].axis("off")
    ax[1,1].imshow(crop(image, roi)); ax[1,1].set_title("ROI input")
    ax[1,2].imshow(crop(overlay_two(image, skeletonize(gt), skeletonize(pb)), roi)); ax[1,2].set_title(f"Dice+Boundary skeleton\nclDice={row['cldice_boundary']:.3f}")
    ax[1,3].imshow(crop(overlay_two(image, skeletonize(gt), skeletonize(pc)), roi)); ax[1,3].set_title(f"Dice+clDice skeleton\nclDice={row['cldice_cldice']:.3f}")
    for i in range(2):
        for j in range(4):
            if not (i == 0 and j == 0):
                ax[i,j].axis("off")
    fig.suptitle("C: Similar strict F1, different topology score (representative ROI)", fontsize=10)
    fig.text(.5, .01, "GT blue, prediction orange; overlap purple. Display skeletons are dilated by 1 px for visibility only.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, .94])
    path = OUT / "Case_C_representative_similar_F1_different_topology_ROI.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "case": "C: Similar F1, different topology",
        "image": item.stem,
        "loss": "dice_boundary vs dice_cldice",
        "seed": 0,
        "threshold_boundary": tb,
        "threshold_cldice": tc,
        "f1_boundary": row["f1_boundary"],
        "f1_cldice": row["f1_cldice"],
        "cldice_boundary": row["cldice_boundary"],
        "cldice_cldice": row["cldice_cldice"],
        "f1_gap": row["f1_gap"],
        "cldice_gap": row["cldice_gap"],
        "file": str(path),
    }


def main() -> None:
    if not SCORING_CSV.exists():
        raise FileNotFoundError(f"Run scoring first: {SCORING_CSV}")
    if not HOLE_CSV.exists():
        raise FileNotFoundError(f"Run 06_skeleton_hole_sensitivity.py first: {HOLE_CSV}")

    OUT.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.mkdir(parents=True, exist_ok=True)

    scoring = pd.read_csv(SCORING_CSV)
    holes = pd.read_csv(HOLE_CSV)
    item_map = get_item_map()

    a = select_case_a(scoring, holes)
    b = select_case_b(scoring)
    c = select_case_c(scoring)

    dice0 = scoring[(scoring.architecture == "UNet_ResNet34") & (scoring.loss == "dice_cldice") & (scoring.seed == 0)].copy()
    a_len_pct = percentile_rank(dice0.skeleton_length_abs_error_pct, a["skeleton_length_abs_error_pct"])
    a_cl_pct = percentile_rank(dice0.cldice, a["cldice"])
    b_area_pct = percentile_rank(dice0.area_abs_error_pct, b["area_abs_error_pct"])

    selected: List[dict] = []
    selected.append(render_single_case(
        "A: High clDice with representative upper-quartile length error",
        a, item_map[a["image"]], "dice_cldice", 0,
        "Case_A_representative_hole_sensitive_length_error_ROI.png",
        f"clDice percentile={a_cl_pct:.0f}; length-error percentile={a_len_pct:.0f}; "
        f"raw-to-fill64 reduction={a.get('hole_reduction_pp', np.nan):.1f} pp."
    ))
    selected[-1].update({
        "cldice_percentile_within_loss_seed": a_cl_pct,
        "length_error_percentile_within_loss_seed": a_len_pct,
        "hole_fill64_error_reduction_pp": a.get("hole_reduction_pp", np.nan),
    })

    selected.append(render_single_case(
        "B: Low fragmentation error with representative over-segmentation",
        b, item_map[b["image"]], "dice_cldice", 0,
        "Case_B_representative_low_fragmentation_oversegmentation_ROI.png",
        f"area-error percentile among Dice+clDice seed-0 images={b_area_pct:.0f}; fragmentation error={int(b['fragmentation_abs_error'])}."
    ))
    selected[-1].update({"area_error_percentile_within_loss_seed": b_area_pct})

    selected.append(render_case_c(c, item_map[c["image"]], scoring))

    out_df = pd.DataFrame(selected)
    out_df.to_csv(OUT_TABLE / "Selected_Representative_Qualitative_Cases_v2.csv", index=False)

    selection_manifest = {
        "case_A_rule": "Dice+clDice seed0; clDice >= 75th percentile; skeleton-length error <=90th percentile; prefer positive <=64-px-hole sensitivity; choose error nearest 75th percentile.",
        "case_B_rule": "Dice+clDice seed0; fragmentation error <=1; over-segmenting; F1>=0.4; choose area error nearest 75th percentile.",
        "case_C_rule": "Dice+Boundary vs Dice+clDice seed0; F1 gap <=0.02; GT area >= median; choose clDice gap nearest 75th percentile.",
        "visualization": "ROI crop around union of GT and prediction(s). GT=blue, prediction=orange, overlap=purple. Boundary/skeleton display lines dilated by one pixel only for visibility.",
        "metrics": "All reported metrics use uncropped original masks/predictions and are not affected by display dilation.",
    }
    (OUT_TABLE / "qualitative_selection_manifest_v2.json").write_text(json.dumps(selection_manifest, indent=2), encoding="utf-8")

    print("Representative qualitative cases created:")
    print(out_df.to_string(index=False))
    print(f"\nFigures: {OUT}")
    print(f"Selection table: {OUT_TABLE}")


if __name__ == "__main__":
    main()
