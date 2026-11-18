#!/usr/bin/env python
"""Score the architecture and seed extension and create revision-ready tables.

Outputs include:
- exact validation-selected threshold for every architecture/loss/seed;
- validation Brier score and 15-bin ECE (calibration context for extreme thresholds);
- complete test metrics for every loss, not only winners;
- per-image absolute area and skeleton-length errors;
- 5-seed U-Net summary (original seeds 0-2 + new seeds 3-4);
- 3-seed DeepLabV3+ summary;
- hierarchical paired-bootstrap confidence intervals for all U-Net loss pairs
  on global F1 and per-image clDice.

The original 72-run outputs are read-only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from scipy.stats import kendalltau
from skimage.morphology import skeletonize
from tqdm import tqdm

# =============================================================================
# 1. PATHS / CONSTANTS
# =============================================================================
PAPER_ROOT = Path(os.environ.get("PAPER3_ROOT", Path.cwd())).expanduser().resolve()
DATASET_BASE = Path(os.environ.get("PAPER3_DATASETS", PAPER_ROOT / "Datasets")).expanduser().resolve()
DATASET_ROOT = DATASET_BASE / "Crack500"
OUTPUT_BASE = Path(os.environ.get("PAPER3_OUTPUT", PAPER_ROOT / "Output")).expanduser().resolve()
VAL_DIR = DATASET_ROOT / "valcrop" / "valcrop"
TEST_DIR = DATASET_ROOT / "testcrop" / "testcrop"

ORIGINAL_UNET_ROOT = OUTPUT_BASE / "OUTPUT Crack500 Full 100 Epochs"
EXTRA_UNET_ROOT = OUTPUT_BASE / "Extension" / "UNet_ResNet34_Crack500_Seeds3_4"
DEEPLAB_ROOT = OUTPUT_BASE / "Extension" / "DeepLabV3Plus_ResNet50_Crack500"
OUT = OUTPUT_BASE / "Extended_Analysis" / "Scoring"
ORIGINAL_OFFLINE_ROOT = Path(os.environ.get("PAPER3_OFFLINE_SCORE_OUTPUT", PAPER_ROOT / "OFFLINE_SCORE_OUTPUT")).expanduser().resolve()

LOSS_ORDER = ["bce", "focal", "bce_dice", "focal_tversky", "dice_boundary", "dice_cldice"]
THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100.0
RELAXED_RADII = [0, 2, 3, 4, 5]
BOUNDARY_THETA = 2
STRUCT8 = np.ones((3, 3), dtype=bool)
EPS = 1e-12
EXPECTED_VAL = 348
EXPECTED_TEST = 1124


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    loss: str
    seed: int
    run_dir: Path


# =============================================================================
# 2. BASIC I/O / METRICS
# =============================================================================
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
    raise ValueError(f"Unexpected non-binary mask encoding: {path}; values={unique[:20]}")


def load_prob(path: Path) -> np.ndarray:
    p = np.load(path, allow_pickle=False)
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError(f"Invalid probability map: {path}")
    if p.min() < -1e-6 or p.max() > 1 + 1e-6:
        raise ValueError(f"Probability map outside [0,1]: {path}")
    return p.astype(np.float32, copy=False)


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int]:
    return (
        int(np.logical_and(pred, gt).sum()),
        int(np.logical_and(pred, ~gt).sum()),
        int(np.logical_and(~pred, gt).sum()),
    )


def global_scores(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Match the submitted v2 scorer's explicit empty-denominator rules."""
    f1 = 1.0 if (2 * tp + fp + fn) == 0 else (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
    iou = 1.0 if (tp + fp + fn) == 0 else tp / (tp + fp + fn + EPS)
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp + EPS)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn + EPS)
    return {
        "f1_global": float(f1),
        "iou_global": float(iou),
        "precision_global": float(precision),
        "recall_global": float(recall),
    }


def empty_case(pred: np.ndarray, gt: np.ndarray) -> str:
    pa, ga = bool(pred.any()), bool(gt.any())
    if not pa and not ga: return "A_both_empty"
    if pa and not ga: return "B_gt_empty_pred_nonempty"
    if not pa and ga: return "C_gt_nonempty_pred_empty"
    return "D_both_nonempty"


def per_image_overlap(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    case = empty_case(pred, gt)
    if case == "A_both_empty":
        return {"f1": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    if case != "D_both_nonempty":
        return {"f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    tp, fp, fn = confusion_counts(pred, gt)
    return {
        "f1": float(2 * tp / (2 * tp + fp + fn + EPS)),
        "iou": float(tp / (tp + fp + fn + EPS)),
        "precision": float(tp / (tp + fp + EPS)),
        "recall": float(tp / (tp + fn + EPS)),
    }


def relaxed_f1(pred: np.ndarray, gt: np.ndarray, radius: int) -> float:
    case = empty_case(pred, gt)
    if case == "A_both_empty": return 1.0
    if case != "D_both_nonempty": return 0.0
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    matched_pred = int(np.logical_and(pred, dt_gt <= radius).sum())
    matched_gt = int(np.logical_and(gt, dt_pred <= radius).sum())
    p = matched_pred / (int(pred.sum()) + EPS)
    r = matched_gt / (int(gt.sum()) + EPS)
    return float(2 * p * r / (p + r + EPS))


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=STRUCT8, border_value=0)
    return np.logical_and(mask, ~eroded)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, theta: int = BOUNDARY_THETA) -> float:
    case = empty_case(pred, gt)
    if case == "A_both_empty": return 1.0
    if case != "D_both_nonempty": return 0.0
    pb, gb = boundary_pixels(pred), boundary_pixels(gt)
    if not pb.any() and not gb.any(): return 1.0
    if not pb.any() or not gb.any(): return 0.0
    dt_g = distance_transform_edt(~gb)
    dt_p = distance_transform_edt(~pb)
    mp = int(np.logical_and(pb, dt_g <= theta).sum())
    mg = int(np.logical_and(gb, dt_p <= theta).sum())
    p = mp / (int(pb.sum()) + EPS)
    r = mg / (int(gb.sum()) + EPS)
    return float(2 * p * r / (p + r + EPS))


def cldice_metric(pred: np.ndarray, gt: np.ndarray) -> float:
    case = empty_case(pred, gt)
    if case == "A_both_empty": return 1.0
    if case != "D_both_nonempty": return 0.0
    sp = skeletonize(pred).astype(bool)
    sg = skeletonize(gt).astype(bool)
    if not sp.any() or not sg.any(): return 0.0
    tprec = int(np.logical_and(sp, gt).sum()) / (int(sp.sum()) + EPS)
    tsens = int(np.logical_and(sg, pred).sum()) / (int(sg.sum()) + EPS)
    return float(2 * tprec * tsens / (tprec + tsens + EPS))


def component_count(mask: np.ndarray) -> int:
    _, n = label(mask, structure=STRUCT8)
    return int(n)


def abs_rel_pct(pred_value: float, gt_value: float) -> float:
    if gt_value <= 0:
        return float("nan")
    return float(abs(pred_value - gt_value) / gt_value * 100.0)


def select_threshold(prob_dir: Path, mask_dir: Path, stems: Sequence[str]) -> Tuple[float, float]:
    edges = np.linspace(0.0, 1.0, 101)
    pos_hist = np.zeros(100, dtype=np.float64)
    all_hist = np.zeros(100, dtype=np.float64)
    n_pos = 0.0
    for stem in stems:
        p = load_prob(prob_dir / f"{stem}.npy")
        gt = load_mask(mask_dir / f"{stem}.png")
        if p.shape != gt.shape:
            raise ValueError(f"Shape mismatch {stem}: {p.shape} vs {gt.shape}")
        pos_hist += np.histogram(p[gt], bins=edges)[0]
        all_hist += np.histogram(p, bins=edges)[0]
        n_pos += gt.sum()
    pos_tail = np.cumsum(pos_hist[::-1])[::-1][1:100]
    all_tail = np.cumsum(all_hist[::-1])[::-1][1:100]
    f1 = 2 * pos_tail / (all_tail + n_pos + EPS)
    rounded = np.round(f1, 4)
    candidates = np.flatnonzero(rounded == rounded.max())
    order = np.lexsort((THRESHOLDS[candidates], np.abs(THRESHOLDS[candidates] - 0.5)))
    i = candidates[order[0]]
    return float(THRESHOLDS[i]), float(f1[i])


def calibration_metrics(prob_dir: Path, mask_dir: Path, stems: Sequence[str], bins: int = 15) -> Tuple[float, float]:
    brier_sum = 0.0
    n = 0
    count = np.zeros(bins, dtype=np.float64)
    p_sum = np.zeros(bins, dtype=np.float64)
    y_sum = np.zeros(bins, dtype=np.float64)
    for stem in stems:
        p = load_prob(prob_dir / f"{stem}.npy").ravel().astype(np.float64)
        y = load_mask(mask_dir / f"{stem}.png").ravel().astype(np.float64)
        brier_sum += np.square(p - y).sum()
        n += p.size
        idx = np.minimum((p * bins).astype(int), bins - 1)
        count += np.bincount(idx, minlength=bins)
        p_sum += np.bincount(idx, weights=p, minlength=bins)
        y_sum += np.bincount(idx, weights=y, minlength=bins)
    nonzero = count > 0
    conf = np.zeros(bins); acc = np.zeros(bins)
    conf[nonzero] = p_sum[nonzero] / count[nonzero]
    acc[nonzero] = y_sum[nonzero] / count[nonzero]
    ece = np.sum((count / max(n, 1)) * np.abs(acc - conf))
    return float(brier_sum / max(n, 1)), float(ece)


# =============================================================================
# 3. RUN DISCOVERY / PREFLIGHT
# =============================================================================
def list_stems(directory: Path, ext: str) -> List[str]:
    return sorted(p.stem for p in directory.glob(f"*{ext}"))


def discover_runs() -> List[RunSpec]:
    specs: List[RunSpec] = []
    for loss in LOSS_ORDER:
        for seed in [0, 1, 2]:
            specs.append(RunSpec("UNet_ResNet34", loss, seed, ORIGINAL_UNET_ROOT / f"{loss}_seed{seed}"))
        for seed in [3, 4]:
            specs.append(RunSpec("UNet_ResNet34", loss, seed, EXTRA_UNET_ROOT / f"{loss}_seed{seed}"))
        for seed in [0, 1, 2]:
            specs.append(RunSpec("DeepLabV3Plus_ResNet50", loss, seed, DEEPLAB_ROOT / f"{loss}_seed{seed}"))
    return specs


def verify_run(run: RunSpec, val_stems: Sequence[str], test_stems: Sequence[str]) -> dict:
    manifest_path = run.run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Incomplete run: {run.run_dir}")
    for folder, expected in [("probmaps_val", val_stems), ("probmaps_test", test_stems)]:
        actual = {p.stem for p in (run.run_dir / folder).glob("*.npy")}
        if actual != set(expected):
            raise RuntimeError(f"Probability-map set mismatch in {run.run_dir / folder}")
    return manifest


# =============================================================================
# 4. SCORE ONE RUN
# =============================================================================
def score_run(run: RunSpec, val_stems: Sequence[str], test_stems: Sequence[str]) -> Tuple[dict, List[dict]]:
    manifest = verify_run(run, val_stems, test_stems)
    val_prob = run.run_dir / "probmaps_val"
    test_prob = run.run_dir / "probmaps_test"

    threshold, val_f1 = select_threshold(val_prob, VAL_DIR, val_stems)
    recorded = manifest.get("best_val_threshold")
    if recorded is not None and abs(float(recorded) - threshold) > 1e-9:
        raise RuntimeError(f"Threshold provenance mismatch {run.run_dir}: manifest={recorded}, recomputed={threshold}")
    brier, ece = calibration_metrics(val_prob, VAL_DIR, val_stems)

    rows: List[dict] = []
    tp_total = fp_total = fn_total = 0
    relaxed = {r: [] for r in RELAXED_RADII}
    bf_values: List[float] = []
    cd_values: List[float] = []
    frag_values: List[float] = []
    per_f1: List[float] = []
    per_iou: List[float] = []
    area_pred_total = area_gt_total = 0
    len_pred_total = len_gt_total = 0
    area_err: List[float] = []
    len_err: List[float] = []

    for stem in tqdm(test_stems, desc=f"score {run.architecture[:7]} {run.loss} s{run.seed}", unit="img", leave=False):
        p = load_prob(test_prob / f"{stem}.npy")
        gt = load_mask(TEST_DIR / f"{stem}.png")
        pred = p >= threshold
        tp, fp, fn = confusion_counts(pred, gt)
        tp_total += tp; fp_total += fp; fn_total += fn
        ov = per_image_overlap(pred, gt)
        per_f1.append(ov["f1"]); per_iou.append(ov["iou"])
        rvals = {r: relaxed_f1(pred, gt, r) for r in RELAXED_RADII}
        for r, v in rvals.items(): relaxed[r].append(v)
        bf = boundary_f1(pred, gt)
        cd = cldice_metric(pred, gt)
        frag = abs(component_count(pred) - component_count(gt))
        sp = skeletonize(pred).astype(bool)
        sg = skeletonize(gt).astype(bool)
        ap, ag = int(pred.sum()), int(gt.sum())
        lp, lg = int(sp.sum()), int(sg.sum())
        area_pred_total += ap; area_gt_total += ag
        len_pred_total += lp; len_gt_total += lg
        ae = abs_rel_pct(ap, ag)
        le = abs_rel_pct(lp, lg)
        if np.isfinite(ae): area_err.append(ae)
        if np.isfinite(le): len_err.append(le)
        bf_values.append(bf); cd_values.append(cd); frag_values.append(frag)
        rows.append({
            "architecture": run.architecture, "loss": run.loss, "seed": run.seed, "image": stem,
            "threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "f1": ov["f1"], "iou": ov["iou"], "boundary_f1_theta2": bf,
            "cldice": cd, "fragmentation_abs_error": frag,
            "area_pred": ap, "area_gt": ag, "area_abs_error_pct": ae,
            "skeleton_length_pred": lp, "skeleton_length_gt": lg,
            "skeleton_length_abs_error_pct": le,
            **{f"relaxed_f1_r{r}": rvals[r] for r in RELAXED_RADII},
        })

    def dist_stats(values: Sequence[float], prefix: str) -> dict:
        a = np.asarray(values, dtype=float)
        return {
            f"{prefix}_mean": float(np.mean(a)), f"{prefix}_std": float(np.std(a, ddof=1)),
            f"{prefix}_median": float(np.median(a)), f"{prefix}_q25": float(np.quantile(a, .25)),
            f"{prefix}_q75": float(np.quantile(a, .75)),
        }

    summary = {
        "architecture": run.architecture, "loss": run.loss, "seed": run.seed,
        "validation_threshold": threshold, "validation_f1_recomputed": val_f1,
        "threshold_at_grid_boundary": int(threshold in {0.01, 0.99}),
        "validation_brier": brier, "validation_ece15": ece,
        **global_scores(tp_total, fp_total, fn_total),
        "f1_per_image_mean": float(np.mean(per_f1)), "iou_per_image_mean": float(np.mean(per_iou)),
        **{f"relaxed_f1_r{r}_mean": float(np.mean(v)) for r, v in relaxed.items()},
        "boundary_f1_theta2_mean": float(np.mean(bf_values)),
        "cldice_mean": float(np.mean(cd_values)),
        "fragmentation_abs_error_mean": float(np.mean(frag_values)),
        "area_error_pct_dataset_total": abs_rel_pct(area_pred_total, area_gt_total),
        "skeleton_length_error_pct_dataset_total": abs_rel_pct(len_pred_total, len_gt_total),
        **dist_stats(area_err, "area_abs_error_pct_per_image"),
        **dist_stats(len_err, "skeleton_length_abs_error_pct_per_image"),
    }
    return summary, rows


# =============================================================================
# 5. SUMMARIES / HIERARCHICAL BOOTSTRAP
# =============================================================================
def summarize_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in {"architecture", "loss", "seed"}]
    numeric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    out = []
    for (arch, loss), g in df.groupby(["architecture", "loss"], sort=False):
        row = {"architecture": arch, "loss": loss, "n_seeds": g["seed"].nunique()}
        for c in numeric_cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=1)
        out.append(row)
    return pd.DataFrame(out)


def hierarchical_bootstrap(per_image: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Paired hierarchical bootstrap across seeds and images for U-Net 5-seed runs.

    Each replicate samples 5 seeds with replacement. Within every selected seed,
    test images are sampled with replacement. Global F1 is recomputed from pooled
    counts within each selected seed and then averaged over sampled seeds, matching
    the manuscript's mean-across-seeds reporting. clDice is averaged over sampled
    images and then over sampled seeds.
    """
    rng = np.random.default_rng(seed)
    u = per_image[per_image.architecture == "UNet_ResNet34"].copy()
    seeds = sorted(u.seed.unique())
    images = sorted(u.image.unique())
    if seeds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Expected U-Net seeds 0..4, found {seeds}")
    n_img = len(images)
    image_index = {name: i for i, name in enumerate(images)}
    seed_index = {s: i for i, s in enumerate(seeds)}

    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for loss in LOSS_ORDER:
        arrays[loss] = {k: np.zeros((len(seeds), n_img), dtype=np.float64) for k in ["tp", "fp", "fn", "cldice"]}
        g = u[u.loss == loss]
        for row in g.itertuples(index=False):
            si = seed_index[row.seed]; ii = image_index[row.image]
            arrays[loss]["tp"][si, ii] = row.tp
            arrays[loss]["fp"][si, ii] = row.fp
            arrays[loss]["fn"][si, ii] = row.fn
            arrays[loss]["cldice"][si, ii] = row.cldice

    results = []
    for loss_a, loss_b in tqdm(list(combinations(LOSS_ORDER, 2)), desc="hierarchical bootstrap", unit="pair"):
        d_f1 = np.empty(n_boot, dtype=float)
        d_cd = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
            f1_a = []; f1_b = []; cd_a = []; cd_b = []
            for si in sampled_seeds:
                idx = rng.integers(0, n_img, size=n_img)
                def one(loss: str):
                    tp = arrays[loss]["tp"][si, idx].sum()
                    fp = arrays[loss]["fp"][si, idx].sum()
                    fn = arrays[loss]["fn"][si, idx].sum()
                    f1 = 2 * tp / (2 * tp + fp + fn + EPS)
                    cd = arrays[loss]["cldice"][si, idx].mean()
                    return f1, cd
                af, ac = one(loss_a); bf, bc = one(loss_b)
                f1_a.append(af); f1_b.append(bf); cd_a.append(ac); cd_b.append(bc)
            d_f1[b] = np.mean(f1_a) - np.mean(f1_b)
            d_cd[b] = np.mean(cd_a) - np.mean(cd_b)
        for metric, d in [("global_f1", d_f1), ("cldice", d_cd)]:
            lo, hi = np.quantile(d, [0.025, 0.975])
            results.append({
                "loss_A": loss_a, "loss_B": loss_b, "metric": metric,
                "A_minus_B_bootstrap_mean": float(np.mean(d)),
                "ci95_low": float(lo), "ci95_high": float(hi),
                "prob_A_greater_B": float(np.mean(d > 0)),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "bootstrap_replicates": n_boot,
                "seed_count": len(seeds), "image_count": n_img,
            })
    return pd.DataFrame(results)



def architecture_robustness_tables(run_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create direct second-architecture checks of the paper's principal findings."""
    metric_specs = {
        "global_f1": ("f1_global", False),
        "boundary_f1": ("boundary_f1_theta2_mean", False),
        "cldice": ("cldice_mean", False),
        "area_abs_error_median": ("area_abs_error_pct_per_image_median", True),
        "skeleton_length_abs_error_median": ("skeleton_length_abs_error_pct_per_image_median", True),
    }
    # Architecture-to-architecture comparisons use matched seeds 0,1,2 for both
    # networks. The additional U-Net seeds 3,4 are reserved for the separate
    # five-seed reliability analysis and do not unbalance the architecture check.
    matched = run_df[run_df["seed"].isin([0, 1, 2])].copy()
    mean_df = matched.groupby(["architecture", "loss"], as_index=False).mean(numeric_only=True)
    rank_rows = []
    tau_rows = []
    for label, (col, lower_better) in metric_specs.items():
        by_arch = {}
        for arch in sorted(mean_df.architecture.unique()):
            g = mean_df[mean_df.architecture == arch].copy()
            g["rank"] = g[col].rank(method="min", ascending=lower_better)
            by_arch[arch] = g.set_index("loss")["rank"]
            for r in g.itertuples(index=False):
                rank_rows.append({
                    "architecture": arch, "metric": label, "loss": r.loss,
                    "value": float(getattr(r, col)), "rank": int(getattr(r, "rank")),
                })
        if {"UNet_ResNet34", "DeepLabV3Plus_ResNet50"}.issubset(by_arch):
            a = by_arch["UNet_ResNet34"].reindex(LOSS_ORDER).to_numpy()
            b = by_arch["DeepLabV3Plus_ResNet50"].reindex(LOSS_ORDER).to_numpy()
            tau, p = kendalltau(a, b)
            tau_rows.append({
                "metric": label, "kendall_tau_between_architectures": float(tau),
                "p_value_descriptive_only": float(p), "n_losses": len(LOSS_ORDER),
                "comparison_seeds": "0,1,2",
                "interpretation": "descriptive rank agreement across six losses using matched seeds 0,1,2",
            })

    # Key pair from the submitted manuscript: Dice+Boundary vs Dice+clDice.
    key_rows = []
    for arch in sorted(matched.architecture.unique()):
        for metric in ["f1_global", "cldice_mean"]:
            for seed in [0, 1, 2]:
                g = matched[(matched.architecture == arch) & (matched.seed == seed)].set_index("loss")
                if {"dice_boundary", "dice_cldice"}.issubset(g.index):
                    a = float(g.loc["dice_boundary", metric])
                    b = float(g.loc["dice_cldice", metric])
                    key_rows.append({
                        "architecture": arch, "seed": int(seed), "metric": metric,
                        "dice_boundary": a, "dice_cldice": b,
                        "dice_boundary_minus_dice_cldice": a - b,
                        "winner": "dice_boundary" if a > b else ("dice_cldice" if b > a else "tie"),
                    })

    # Tolerance sensitivity under each architecture.
    tol_rows = []
    for arch in sorted(matched.architecture.unique()):
        garch = matched[matched.architecture == arch]
        for radius in RELAXED_RADII:
            col = f"relaxed_f1_r{radius}_mean"
            means = garch.groupby("loss")[col].mean().sort_values(ascending=False)
            for rank, (loss, value) in enumerate(means.items(), start=1):
                tol_rows.append({
                    "architecture": arch, "radius_px": radius, "loss": loss,
                    "mean_relaxed_f1": float(value), "rank": rank,
                })
    return pd.DataFrame(rank_rows), pd.DataFrame(tau_rows), pd.DataFrame(key_rows), pd.DataFrame(tol_rows)


def write_combined_threshold_table(run_df: pd.DataFrame) -> None:
    """Combine all 72 submitted thresholds with the 30 post-review thresholds."""
    old_path = ORIGINAL_OFFLINE_ROOT / "validation_thresholds.csv"
    if not old_path.exists():
        print(f"warning: original threshold table not found: {old_path}")
        return
    old = pd.read_csv(old_path).copy()
    old_out = pd.DataFrame({
        "phase": "submitted",
        "dataset": old["dataset"],
        "architecture": "UNet_ResNet34",
        "loss": old["loss"],
        "seed": old["seed"].astype(int),
        "validation_selected_threshold": old["val_selected_threshold_recomputed"],
        "validation_f1": old["val_selected_f1_recomputed"],
        "grid_boundary_0p01_or_0p99": old["val_selected_threshold_recomputed"].isin([0.01, 0.99]).astype(int),
    })
    new = run_df[((run_df.architecture == "DeepLabV3Plus_ResNet50") | (run_df.seed >= 3))].copy()
    new_out = pd.DataFrame({
        "phase": "post_review_extension",
        "dataset": "Crack500",
        "architecture": new["architecture"],
        "loss": new["loss"],
        "seed": new["seed"].astype(int),
        "validation_selected_threshold": new["validation_threshold"],
        "validation_f1": new["validation_f1_recomputed"],
        "grid_boundary_0p01_or_0p99": new["threshold_at_grid_boundary"].astype(int),
    })
    combined = pd.concat([old_out, new_out], ignore_index=True)
    combined = combined.sort_values(["phase", "dataset", "architecture", "loss", "seed"])
    combined.to_csv(OUT / "Supplementary_Table_All_102_Run_Thresholds.csv", index=False)


# =============================================================================
# 6. MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261113)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    val_stems = list_stems(VAL_DIR, ".jpg")
    test_stems = list_stems(TEST_DIR, ".jpg")
    if len(val_stems) != EXPECTED_VAL or len(test_stems) != EXPECTED_TEST:
        raise RuntimeError(f"Crack500 counts wrong: val={len(val_stems)}, test={len(test_stems)}")

    runs = discover_runs()
    summaries: List[dict] = []
    per_rows: List[dict] = []
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run.architecture} / {run.loss} / seed {run.seed}")
        summary, rows = score_run(run, val_stems, test_stems)
        summaries.append(summary); per_rows.extend(rows)
        pd.DataFrame(summaries).to_csv(OUT / "run_level_results_partial.csv", index=False)

    run_df = pd.DataFrame(summaries)
    per_df = pd.DataFrame(per_rows)
    run_df.to_csv(OUT / "run_level_complete_results.csv", index=False)
    per_df.to_csv(OUT / "per_image_complete_results.csv", index=False)

    # Threshold/calibration table requested by Reviewer 1.
    threshold_cols = [
        "architecture", "loss", "seed", "validation_threshold", "validation_f1_recomputed",
        "threshold_at_grid_boundary", "validation_brier", "validation_ece15"
    ]
    run_df[threshold_cols].to_csv(OUT / "Crack500_Thresholds_and_Calibration.csv", index=False)
    write_combined_threshold_table(run_df)

    # Complete all-loss table requested by Reviewer 1.
    seed_summary = summarize_across_seeds(run_df)
    seed_summary.to_csv(OUT / "Supplementary_Table_All_Losses_All_Metrics.csv", index=False)

    ranks, taus, key_pair, tol_ranks = architecture_robustness_tables(run_df)
    ranks.to_csv(OUT / "Architecture_Loss_Ranks.csv", index=False)
    taus.to_csv(OUT / "Architecture_Rank_Agreement_KendallTau.csv", index=False)
    key_pair.to_csv(OUT / "Architecture_Key_DiceBoundary_vs_DiceClDice.csv", index=False)
    tol_ranks.to_csv(OUT / "Architecture_Tolerance_Rank_Sensitivity.csv", index=False)

    # Compact geometric table emphasizing reviewer-requested image-level distributions.
    geo_cols = [c for c in seed_summary.columns if (
        c in {"architecture", "loss", "n_seeds"}
        or "area_error" in c or "skeleton_length_error" in c
        or "area_abs_error" in c or "skeleton_length_abs_error" in c
    )]
    seed_summary[geo_cols].to_csv(OUT / "Supplementary_Table_Per_Image_Geometric_Errors.csv", index=False)

    boot = hierarchical_bootstrap(per_df, args.bootstrap, args.bootstrap_seed)
    boot.to_csv(OUT / "UNet_5seed_Hierarchical_Paired_Bootstrap.csv", index=False)

    manifest = {
        "purpose": "architecture and seed extension scoring",
        "original_unet_seeds": [0, 1, 2],
        "new_unet_seeds": [3, 4],
        "deeplab_seeds": [0, 1, 2],
        "losses": LOSS_ORDER,
        "threshold_grid": "0.01:0.99:0.01",
        "comparator": "p >= t",
        "relaxed_radii_px": RELAXED_RADII,
        "boundary_tolerance_px": BOUNDARY_THETA,
        "per_image_geometry": "absolute relative percentage error; GT-empty excluded",
        "calibration": "validation-set Brier score and 15-bin ECE",
        "bootstrap": {
            "method": "paired hierarchical bootstrap over seeds and test images",
            "replicates": args.bootstrap,
            "seed": args.bootstrap_seed,
        },
    }
    (OUT / "scoring_extension_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. Outputs: {OUT}")


if __name__ == "__main__":
    main()
