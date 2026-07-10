"""
Paper 3 offline scoring suite for saved probability maps.

Inputs already produced by training:
  * OUTPUT Crack500/<run_id>/probmaps_val/*.npy
  * OUTPUT Crack500/<run_id>/probmaps_test/*.npy
  * OUTPUT DeepCrack/<run_id>/probmaps_val/*.npy
  * OUTPUT DeepCrack/<run_id>/probmaps_test/*.npy
  * each run manifest.json
  * dataset-level val_files.txt and test_files.txt created by the training scripts

Frozen protocol implemented:
  * threshold grid 0.01--0.99, step 0.01
  * comparator p >= t
  * fixed 0.5, validation-selected, ODS, and OIS overlap metrics
  * primary advanced metrics at validation-selected threshold:
      relaxed F1 at r={0,2,3,4,5}, boundary F1 theta=2,
      clDice, fragmentation error, area error, skeleton-length error
  * both per-image mean and dataset-global aggregation for F1/IoU
  * empty-mask rules are applied and case counts are reported

Dependencies:
    pip install numpy pillow scipy pandas scikit-image
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from skimage.morphology import skeletonize

# =============================================================================
# 1. PATHS TO EDIT ONLY IF YOUR FOLDERS MOVE
# =============================================================================
CRACK500_OUTPUT_ROOT = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Paper\Review Paper 3\OUTPUT Crack500"
)
DEEPCRACK_OUTPUT_ROOT = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Paper\Review Paper 3\OUTPUT DeepCrack"
)

CRACK500_ROOT = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Dataset\Crack500"
)
DEEPCRACK_ROOT = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Dataset\DeepCrack Dataset"
)

SCORING_OUTPUT_ROOT = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Paper\Review Paper 3\OFFLINE_SCORE_OUTPUT"
)

# =============================================================================
# 2. FROZEN SCORING CONSTANTS
# =============================================================================
THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100.0  # 0.01 ... 0.99
THRESHOLD_GRID_TEXT = "0.01:0.99:0.01"
THRESHOLD_COMPARATOR = "p >= t"
RELAXED_RADII = [0, 2, 3, 4, 5]
BOUNDARY_THETA = 2
EPS = 1e-12
STRUCT8 = np.ones((3, 3), dtype=bool)
LOSS_ORDER = [
    "bce",
    "focal",
    "bce_dice",
    "focal_tversky",
    "dice_boundary",
    "dice_cldice",
]

# Write per-image CSVs? Useful for supplementary distributions; can be large.
WRITE_PER_IMAGE_OVERLAP = True
WRITE_PER_IMAGE_ADVANCED = True


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    output_root: Path
    val_mask_dir: Path
    test_mask_dir: Path
    expected_val: int
    expected_test: int


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        name="Crack500",
        output_root=CRACK500_OUTPUT_ROOT,
        val_mask_dir=CRACK500_ROOT / "valcrop" / "valcrop",
        test_mask_dir=CRACK500_ROOT / "testcrop" / "testcrop",
        expected_val=348,
        expected_test=1124,
    ),
    DatasetSpec(
        name="DeepCrack",
        output_root=DEEPCRACK_OUTPUT_ROOT,
        val_mask_dir=DEEPCRACK_ROOT / "train_lab",
        test_mask_dir=DEEPCRACK_ROOT / "test_lab",
        expected_val=60,
        expected_test=237,
    ),
]


@dataclass(frozen=True)
class SplitItem:
    image_name: str
    mask_name: str

    @property
    def stem(self) -> str:
        return Path(self.image_name).stem


# =============================================================================
# 3. FILE I/O
# =============================================================================
def read_split_file(path: Path) -> List[SplitItem]:
    """Read training-script split list. First two tab-separated columns are
    image filename and mask filename; later SHA-256 columns are ignored here."""
    if not path.exists():
        raise FileNotFoundError(f"Missing split list: {path}")
    items: List[SplitItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Bad split-list line in {path}: {line}")
            items.append(SplitItem(parts[0], parts[1]))
    return items


def load_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")
    m = np.asarray(Image.open(mask_path).convert("L"))
    return m > 127


def load_prob(prob_path: Path) -> np.ndarray:
    if not prob_path.exists():
        raise FileNotFoundError(f"Missing probability map: {prob_path}")
    p = np.load(prob_path)
    if p.ndim != 2:
        raise ValueError(f"Probability map must be HxW, got {p.shape}: {prob_path}")
    return p.astype(np.float32, copy=False)


def discover_runs(output_root: Path) -> List[Path]:
    runs = []
    for p in output_root.iterdir():
        if p.is_dir() and (p / "manifest.json").exists():
            runs.append(p)

    def sort_key(run_dir: Path) -> Tuple[int, int, str]:
        try:
            m = json.loads((run_dir / "manifest.json").read_text())
            loss = m.get("loss", run_dir.name)
            seed = int(m.get("seed", 9999))
        except Exception:
            loss, seed = run_dir.name, 9999
        loss_i = LOSS_ORDER.index(loss) if loss in LOSS_ORDER else 999
        return loss_i, seed, run_dir.name

    return sorted(runs, key=sort_key)


def read_manifest(run_dir: Path) -> dict:
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 4. BASIC COUNTS AND EMPTY-MASK RULES
# =============================================================================
def threshold_pred(prob: np.ndarray, threshold: float) -> np.ndarray:
    return prob >= threshold


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return tp, fp, fn


def empty_case(pred: np.ndarray, gt: np.ndarray) -> str:
    p_any = bool(pred.any())
    g_any = bool(gt.any())
    if not p_any and not g_any:
        return "A_both_empty"
    if p_any and not g_any:
        return "B_gt_empty_pred_nonempty"
    if not p_any and g_any:
        return "C_gt_nonempty_pred_empty"
    return "D_both_nonempty"


def image_overlap_scores(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Per-image overlap metrics with frozen empty-mask rules."""
    case = empty_case(pred, gt)
    if case == "A_both_empty":
        return {"f1": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    if case in {"B_gt_empty_pred_nonempty", "C_gt_nonempty_pred_empty"}:
        return {"f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    tp, fp, fn = confusion_counts(pred, gt)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    return {
        "f1": float(f1),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def global_overlap_scores(tp: int, fp: int, fn: int) -> Dict[str, float]:
    if (2 * tp + fp + fn) == 0:
        f1 = 1.0
    else:
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
    if (tp + fp + fn) == 0:
        iou = 1.0
    else:
        iou = tp / (tp + fp + fn + EPS)
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp + EPS)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn + EPS)
    return {
        "f1_global": float(f1),
        "iou_global": float(iou),
        "precision_global": float(precision),
        "recall_global": float(recall),
    }


# =============================================================================
# 5. THRESHOLD SELECTION
# =============================================================================
def _hist_counts_for_threshold_grid(
    prob_dir: Path,
    mask_dir: Path,
    items: Sequence[SplitItem],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return pos_tail, all_tail, n_pos for thresholds 0.01..0.99.
    Histogram bins preserve comparator p >= t for t on the 0.01 grid."""
    edges = np.linspace(0.0, 1.0, 101)
    pos_hist = np.zeros(100, dtype=np.float64)
    all_hist = np.zeros(100, dtype=np.float64)
    n_pos = 0.0

    for item in items:
        prob = load_prob(prob_dir / f"{item.stem}.npy")
        gt = load_mask(mask_dir / item.mask_name)
        if prob.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for {item.image_name}: prob={prob.shape}, mask={gt.shape}"
            )
        pos_hist += np.histogram(prob[gt], bins=edges)[0]
        all_hist += np.histogram(prob, bins=edges)[0]
        n_pos += float(gt.sum())

    pos_tail = np.cumsum(pos_hist[::-1])[::-1]
    all_tail = np.cumsum(all_hist[::-1])[::-1]
    return pos_tail[1:100], all_tail[1:100], n_pos


def select_threshold_global(
    prob_dir: Path,
    mask_dir: Path,
    items: Sequence[SplitItem],
) -> Tuple[float, float]:
    """Dataset-global F1 argmax on the frozen threshold grid.
    Tie rule: F1 ties at four decimals -> closest to 0.5, then lower."""
    tp, predpos, n_pos = _hist_counts_for_threshold_grid(prob_dir, mask_dir, items)
    f1 = (2.0 * tp) / (predpos + n_pos + EPS)
    f1r = np.round(f1, 4)
    candidates = np.flatnonzero(f1r == f1r.max())
    order = np.lexsort((THRESHOLDS[candidates], np.abs(THRESHOLDS[candidates] - 0.5)))
    best_i = candidates[order[0]]
    return float(THRESHOLDS[best_i]), float(f1[best_i])


def select_threshold_per_image(prob: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Per-image F1 argmax for OIS, with the same threshold tie rule."""
    f1_values = []
    n_pos = int(gt.sum())
    for t in THRESHOLDS:
        pred = prob >= t
        predpos = int(pred.sum())
        tp = int(np.logical_and(pred, gt).sum())
        if n_pos == 0 and predpos == 0:
            f1 = 1.0
        elif n_pos == 0 and predpos > 0:
            f1 = 0.0
        elif n_pos > 0 and predpos == 0:
            f1 = 0.0
        else:
            f1 = (2.0 * tp) / (predpos + n_pos + EPS)
        f1_values.append(f1)
    f1_arr = np.asarray(f1_values, dtype=np.float64)
    f1r = np.round(f1_arr, 4)
    candidates = np.flatnonzero(f1r == f1r.max())
    order = np.lexsort((THRESHOLDS[candidates], np.abs(THRESHOLDS[candidates] - 0.5)))
    best_i = candidates[order[0]]
    return float(THRESHOLDS[best_i]), float(f1_arr[best_i])


# =============================================================================
# 6. OVERLAP SCORING
# =============================================================================
def score_overlap_dataset(
    *,
    dataset_name: str,
    run_id: str,
    loss: str,
    seed: int,
    prob_dir: Path,
    mask_dir: Path,
    items: Sequence[SplitItem],
    threshold_convention: str,
    threshold: Optional[float] = None,
    per_image_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[dict, List[dict]]:
    """Compute overlap metrics for one run and one threshold convention."""
    if threshold is None and per_image_thresholds is None:
        raise ValueError("Either threshold or per_image_thresholds must be provided.")

    per_image_rows: List[dict] = []
    tp_total = fp_total = fn_total = 0
    case_counts = {
        "A_both_empty": 0,
        "B_gt_empty_pred_nonempty": 0,
        "C_gt_nonempty_pred_empty": 0,
        "D_both_nonempty": 0,
    }
    f1s, ious, ps, rs = [], [], [], []
    thresholds_used = []

    for item in items:
        prob = load_prob(prob_dir / f"{item.stem}.npy")
        gt = load_mask(mask_dir / item.mask_name)
        t = per_image_thresholds[item.stem] if per_image_thresholds is not None else float(threshold)
        pred = threshold_pred(prob, t)
        tp, fp, fn = confusion_counts(pred, gt)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        scores = image_overlap_scores(pred, gt)
        case = empty_case(pred, gt)
        case_counts[case] += 1
        f1s.append(scores["f1"])
        ious.append(scores["iou"])
        ps.append(scores["precision"])
        rs.append(scores["recall"])
        thresholds_used.append(t)

        if WRITE_PER_IMAGE_OVERLAP:
            per_image_rows.append({
                "dataset": dataset_name,
                "run_id": run_id,
                "loss": loss,
                "seed": seed,
                "image": item.image_name,
                "mask": item.mask_name,
                "threshold_convention": threshold_convention,
                "threshold": round(t, 2),
                "case": case,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": scores["f1"],
                "iou": scores["iou"],
                "precision": scores["precision"],
                "recall": scores["recall"],
            })

    global_scores = global_overlap_scores(tp_total, fp_total, fn_total)
    summary = {
        "dataset": dataset_name,
        "run_id": run_id,
        "loss": loss,
        "seed": seed,
        "threshold_convention": threshold_convention,
        "threshold": round(float(threshold), 2) if threshold is not None else np.nan,
        "threshold_mean": float(np.mean(thresholds_used)),
        "threshold_min": float(np.min(thresholds_used)),
        "threshold_max": float(np.max(thresholds_used)),
        "n_images": len(items),
        "tp_global_count": tp_total,
        "fp_global_count": fp_total,
        "fn_global_count": fn_total,
        **global_scores,
        "f1_per_image_mean": float(np.mean(f1s)),
        "iou_per_image_mean": float(np.mean(ious)),
        "precision_per_image_mean": float(np.mean(ps)),
        "recall_per_image_mean": float(np.mean(rs)),
        **{f"case_{k}": v for k, v in case_counts.items()},
    }
    return summary, per_image_rows


# =============================================================================
# 7. ADVANCED METRICS AT VALIDATION-SELECTED THRESHOLD
# =============================================================================
def relaxed_f1(pred: np.ndarray, gt: np.ndarray, radius: int) -> float:
    case = empty_case(pred, gt)
    if case == "A_both_empty":
        return 1.0
    if case in {"B_gt_empty_pred_nonempty", "C_gt_nonempty_pred_empty"}:
        return 0.0

    dt_to_gt = distance_transform_edt(~gt)
    dt_to_pred = distance_transform_edt(~pred)
    matched_pred = int(np.logical_and(pred, dt_to_gt <= radius).sum())
    matched_gt = int(np.logical_and(gt, dt_to_pred <= radius).sum())
    precision = matched_pred / (int(pred.sum()) + EPS)
    recall = matched_gt / (int(gt.sum()) + EPS)
    return float((2.0 * precision * recall) / (precision + recall + EPS))


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=STRUCT8, border_value=0)
    return np.logical_and(mask, ~eroded)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, theta: int = BOUNDARY_THETA) -> float:
    case = empty_case(pred, gt)
    if case == "A_both_empty":
        return 1.0
    if case in {"B_gt_empty_pred_nonempty", "C_gt_nonempty_pred_empty"}:
        return 0.0

    pb = boundary_pixels(pred)
    gb = boundary_pixels(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0

    dt_to_gb = distance_transform_edt(~gb)
    dt_to_pb = distance_transform_edt(~pb)
    matched_pb = int(np.logical_and(pb, dt_to_gb <= theta).sum())
    matched_gb = int(np.logical_and(gb, dt_to_pb <= theta).sum())
    precision = matched_pb / (int(pb.sum()) + EPS)
    recall = matched_gb / (int(gb.sum()) + EPS)
    return float((2.0 * precision * recall) / (precision + recall + EPS))


def cldice_metric(pred: np.ndarray, gt: np.ndarray) -> float:
    case = empty_case(pred, gt)
    if case == "A_both_empty":
        return 1.0
    if case in {"B_gt_empty_pred_nonempty", "C_gt_nonempty_pred_empty"}:
        return 0.0

    sp = skeletonize(pred).astype(bool)
    sg = skeletonize(gt).astype(bool)
    if not sp.any() or not sg.any():
        return 0.0
    tprec = int(np.logical_and(sp, gt).sum()) / (int(sp.sum()) + EPS)
    tsens = int(np.logical_and(sg, pred).sum()) / (int(sg.sum()) + EPS)
    return float((2.0 * tprec * tsens) / (tprec + tsens + EPS))


def component_count(mask: np.ndarray) -> int:
    _, n = label(mask, structure=STRUCT8)
    return int(n)


def rel_error_pct(pred_sum: float, gt_sum: float) -> float:
    if gt_sum == 0:
        return float("nan")
    return float(abs(pred_sum - gt_sum) / gt_sum * 100.0)


def score_advanced_val_selected(
    *,
    dataset_name: str,
    run_id: str,
    loss: str,
    seed: int,
    prob_dir: Path,
    mask_dir: Path,
    items: Sequence[SplitItem],
    threshold: float,
) -> Tuple[dict, List[dict]]:
    rows: List[dict] = []
    case_counts = {
        "A_both_empty": 0,
        "B_gt_empty_pred_nonempty": 0,
        "C_gt_nonempty_pred_empty": 0,
        "D_both_nonempty": 0,
    }
    relaxed_values = {r: [] for r in RELAXED_RADII}
    boundary_values: List[float] = []
    cldice_values: List[float] = []
    frag_abs_values: List[int] = []

    area_pred_sum = 0.0
    area_gt_sum = 0.0
    length_pred_sum = 0.0
    length_gt_sum = 0.0

    for item in items:
        prob = load_prob(prob_dir / f"{item.stem}.npy")
        gt = load_mask(mask_dir / item.mask_name)
        pred = threshold_pred(prob, threshold)
        case = empty_case(pred, gt)
        case_counts[case] += 1

        r_scores = {r: relaxed_f1(pred, gt, r) for r in RELAXED_RADII}
        bf = boundary_f1(pred, gt, BOUNDARY_THETA)
        cd = cldice_metric(pred, gt)
        n_pred = component_count(pred)
        n_gt = component_count(gt)
        frag_abs = abs(n_pred - n_gt)

        area_pred = int(pred.sum())
        area_gt = int(gt.sum())
        skel_pred = skeletonize(pred).astype(bool)
        skel_gt = skeletonize(gt).astype(bool)
        length_pred = int(skel_pred.sum())
        length_gt = int(skel_gt.sum())

        area_pred_sum += area_pred
        area_gt_sum += area_gt
        length_pred_sum += length_pred
        length_gt_sum += length_gt

        for r, v in r_scores.items():
            relaxed_values[r].append(v)
        boundary_values.append(bf)
        cldice_values.append(cd)
        frag_abs_values.append(frag_abs)

        if WRITE_PER_IMAGE_ADVANCED:
            row = {
                "dataset": dataset_name,
                "run_id": run_id,
                "loss": loss,
                "seed": seed,
                "image": item.image_name,
                "mask": item.mask_name,
                "threshold_convention": "val_selected",
                "threshold": round(threshold, 2),
                "case": case,
                "boundary_f1_theta2": bf,
                "cldice": cd,
                "components_pred": n_pred,
                "components_gt": n_gt,
                "fragmentation_abs_error": frag_abs,
                "area_pred": area_pred,
                "area_gt": area_gt,
                "skeleton_length_pred": length_pred,
                "skeleton_length_gt": length_gt,
            }
            for r, v in r_scores.items():
                row[f"relaxed_f1_r{r}"] = v
            rows.append(row)

    summary = {
        "dataset": dataset_name,
        "run_id": run_id,
        "loss": loss,
        "seed": seed,
        "threshold_convention": "val_selected",
        "threshold": round(threshold, 2),
        "n_images": len(items),
        **{f"relaxed_f1_r{r}_per_image_mean": float(np.mean(vals))
           for r, vals in relaxed_values.items()},
        "boundary_f1_theta2_per_image_mean": float(np.mean(boundary_values)),
        "cldice_per_image_mean": float(np.mean(cldice_values)),
        "fragmentation_abs_error_mean": float(np.mean(frag_abs_values)),
        "fragmentation_abs_error_median": float(np.median(frag_abs_values)),
        "area_pred_sum": area_pred_sum,
        "area_gt_sum": area_gt_sum,
        "area_error_pct_dataset": rel_error_pct(area_pred_sum, area_gt_sum),
        "skeleton_length_pred_sum": length_pred_sum,
        "skeleton_length_gt_sum": length_gt_sum,
        "skeleton_length_error_pct_dataset": rel_error_pct(length_pred_sum, length_gt_sum),
        **{f"case_{k}": v for k, v in case_counts.items()},
    }
    return summary, rows


# =============================================================================
# 8. DATASET/RUN DRIVER
# =============================================================================
def verify_run_complete(run_dir: Path, n_val: int, n_test: int) -> None:
    manifest = read_manifest(run_dir)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: {run_dir}")
    if not (run_dir / "best.pt").exists():
        raise FileNotFoundError(f"Missing best.pt: {run_dir}")
    n_val_maps = len(list((run_dir / "probmaps_val").glob("*.npy")))
    n_test_maps = len(list((run_dir / "probmaps_test").glob("*.npy")))
    if n_val_maps != n_val:
        raise RuntimeError(f"{run_dir.name}: val maps {n_val_maps}, expected {n_val}")
    if n_test_maps != n_test:
        raise RuntimeError(f"{run_dir.name}: test maps {n_test_maps}, expected {n_test}")


def score_one_dataset(spec: DatasetSpec) -> Tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    print(f"\n=== {spec.name} ===")
    val_items = read_split_file(spec.output_root / "val_files.txt")
    test_items = read_split_file(spec.output_root / "test_files.txt")
    if len(val_items) != spec.expected_val:
        raise RuntimeError(f"{spec.name}: val list has {len(val_items)}, expected {spec.expected_val}")
    if len(test_items) != spec.expected_test:
        raise RuntimeError(f"{spec.name}: test list has {len(test_items)}, expected {spec.expected_test}")

    runs = discover_runs(spec.output_root)
    if not runs:
        raise RuntimeError(f"No run directories found in {spec.output_root}")
    print(f"found {len(runs)} run directories")

    threshold_rows: List[dict] = []
    overlap_summary_rows: List[dict] = []
    overlap_per_image_rows: List[dict] = []
    advanced_summary_rows: List[dict] = []
    advanced_per_image_rows: List[dict] = []

    for run_dir in runs:
        manifest = read_manifest(run_dir)
        run_id = manifest.get("run_id", run_dir.name)
        loss = manifest.get("loss", run_id.rsplit("_seed", 1)[0])
        seed = int(manifest.get("seed", 0))
        print(f"scoring {spec.name} / {run_id}")
        verify_run_complete(run_dir, len(val_items), len(test_items))

        val_prob_dir = run_dir / "probmaps_val"
        test_prob_dir = run_dir / "probmaps_test"

        # Validation-selected threshold, recomputed offline from released val maps.
        val_thr, val_f1 = select_threshold_global(val_prob_dir, spec.val_mask_dir, val_items)
        manifest_thr = manifest.get("best_val_threshold", None)
        threshold_rows.append({
            "dataset": spec.name,
            "run_id": run_id,
            "loss": loss,
            "seed": seed,
            "val_selected_threshold_recomputed": round(val_thr, 2),
            "val_selected_f1_recomputed": val_f1,
            "manifest_best_val_threshold": manifest_thr,
            "manifest_best_val_f1": manifest.get("best_val_f1", None),
            "threshold_grid": THRESHOLD_GRID_TEXT,
            "threshold_comparator": THRESHOLD_COMPARATOR,
            "threshold_match_manifest": (
                manifest_thr is not None and abs(float(manifest_thr) - val_thr) < 1e-9
            ),
        })

        # Oracle dataset-global threshold on test maps.
        ods_thr, ods_f1 = select_threshold_global(test_prob_dir, spec.test_mask_dir, test_items)

        # Oracle per-image thresholds on test maps.
        ois_thresholds: Dict[str, float] = {}
        for item in test_items:
            prob = load_prob(test_prob_dir / f"{item.stem}.npy")
            gt = load_mask(spec.test_mask_dir / item.mask_name)
            t_img, _ = select_threshold_per_image(prob, gt)
            ois_thresholds[item.stem] = t_img

        conventions = [
            ("fixed_0.5", 0.50, None),
            ("val_selected", val_thr, None),
            ("ODS_test_oracle", ods_thr, None),
            ("OIS_test_oracle", None, ois_thresholds),
        ]
        for conv_name, scalar_thr, per_image_thr in conventions:
            summary, per_image = score_overlap_dataset(
                dataset_name=spec.name,
                run_id=run_id,
                loss=loss,
                seed=seed,
                prob_dir=test_prob_dir,
                mask_dir=spec.test_mask_dir,
                items=test_items,
                threshold_convention=conv_name,
                threshold=scalar_thr,
                per_image_thresholds=per_image_thr,
            )
            if conv_name == "ODS_test_oracle":
                summary["ods_selected_f1_global"] = ods_f1
            overlap_summary_rows.append(summary)
            overlap_per_image_rows.extend(per_image)

        # Advanced metrics primary operating point: validation-selected threshold.
        adv_summary, adv_per_image = score_advanced_val_selected(
            dataset_name=spec.name,
            run_id=run_id,
            loss=loss,
            seed=seed,
            prob_dir=test_prob_dir,
            mask_dir=spec.test_mask_dir,
            items=test_items,
            threshold=val_thr,
        )
        advanced_summary_rows.append(adv_summary)
        advanced_per_image_rows.extend(adv_per_image)

    return (
        threshold_rows,
        overlap_summary_rows,
        overlap_per_image_rows,
        advanced_summary_rows,
        advanced_per_image_rows,
    )


# =============================================================================
# 9. OUTPUT SUMMARIES
# =============================================================================
def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"warning: no rows for {path.name}")
        return
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}")


def make_by_loss_summary(overlap_df: pd.DataFrame, advanced_df: pd.DataFrame) -> pd.DataFrame:
    """Compact manuscript-oriented mean/std table by dataset/loss.
    Crack500 has 3 seeds; DeepCrack has 1 seed, so std will be NaN there."""
    primary_overlap = overlap_df[overlap_df["threshold_convention"] == "val_selected"].copy()
    keep_overlap = [
        "f1_global",
        "iou_global",
        "precision_global",
        "recall_global",
        "f1_per_image_mean",
        "iou_per_image_mean",
    ]
    keep_advanced = [
        "relaxed_f1_r0_per_image_mean",
        "relaxed_f1_r2_per_image_mean",
        "relaxed_f1_r3_per_image_mean",
        "relaxed_f1_r4_per_image_mean",
        "relaxed_f1_r5_per_image_mean",
        "boundary_f1_theta2_per_image_mean",
        "cldice_per_image_mean",
        "fragmentation_abs_error_mean",
        "area_error_pct_dataset",
        "skeleton_length_error_pct_dataset",
    ]

    merged = primary_overlap.merge(
        advanced_df,
        on=["dataset", "run_id", "loss", "seed", "threshold_convention", "threshold", "n_images"],
        how="left",
        suffixes=("", "_adv"),
    )
    metric_cols = keep_overlap + keep_advanced
    grouped = merged.groupby(["dataset", "loss"], sort=False)[metric_cols]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=1).add_suffix("_std")
    out = pd.concat([mean_df, std_df], axis=1).reset_index()

    # Preserve loss order inside each dataset.
    out["loss_order"] = out["loss"].map({k: i for i, k in enumerate(LOSS_ORDER)})
    out = out.sort_values(["dataset", "loss_order"]).drop(columns=["loss_order"])
    return out


def main() -> None:
    t0 = time.time()
    SCORING_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_threshold_rows: List[dict] = []
    all_overlap_summary_rows: List[dict] = []
    all_overlap_per_image_rows: List[dict] = []
    all_advanced_summary_rows: List[dict] = []
    all_advanced_per_image_rows: List[dict] = []

    for spec in DATASETS:
        (
            threshold_rows,
            overlap_summary_rows,
            overlap_per_image_rows,
            advanced_summary_rows,
            advanced_per_image_rows,
        ) = score_one_dataset(spec)
        all_threshold_rows.extend(threshold_rows)
        all_overlap_summary_rows.extend(overlap_summary_rows)
        all_overlap_per_image_rows.extend(overlap_per_image_rows)
        all_advanced_summary_rows.extend(advanced_summary_rows)
        all_advanced_per_image_rows.extend(advanced_per_image_rows)

    write_csv(SCORING_OUTPUT_ROOT / "validation_thresholds.csv", all_threshold_rows)
    write_csv(SCORING_OUTPUT_ROOT / "overlap_summary_all_thresholds.csv", all_overlap_summary_rows)
    write_csv(SCORING_OUTPUT_ROOT / "advanced_summary_val_selected.csv", all_advanced_summary_rows)

    if WRITE_PER_IMAGE_OVERLAP:
        write_csv(SCORING_OUTPUT_ROOT / "per_image_overlap_all_thresholds.csv", all_overlap_per_image_rows)
    if WRITE_PER_IMAGE_ADVANCED:
        write_csv(SCORING_OUTPUT_ROOT / "per_image_advanced_val_selected.csv", all_advanced_per_image_rows)

    overlap_df = pd.DataFrame(all_overlap_summary_rows)
    advanced_df = pd.DataFrame(all_advanced_summary_rows)
    by_loss = make_by_loss_summary(overlap_df, advanced_df)
    by_loss.to_csv(SCORING_OUTPUT_ROOT / "summary_by_dataset_loss_val_selected.csv", index=False)
    print(f"wrote {SCORING_OUTPUT_ROOT / 'summary_by_dataset_loss_val_selected.csv'}")

    meta = {
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 2),
        "threshold_grid": THRESHOLD_GRID_TEXT,
        "threshold_comparator": THRESHOLD_COMPARATOR,
        "relaxed_radii": RELAXED_RADII,
        "boundary_theta_px": BOUNDARY_THETA,
        "outputs": [
            "validation_thresholds.csv",
            "overlap_summary_all_thresholds.csv",
            "advanced_summary_val_selected.csv",
            "per_image_overlap_all_thresholds.csv" if WRITE_PER_IMAGE_OVERLAP else None,
            "per_image_advanced_val_selected.csv" if WRITE_PER_IMAGE_ADVANCED else None,
            "summary_by_dataset_loss_val_selected.csv",
        ],
    }
    with open(SCORING_OUTPUT_ROOT / "scoring_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("\nAll offline scoring complete.")
    print(f"Output folder: {SCORING_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
