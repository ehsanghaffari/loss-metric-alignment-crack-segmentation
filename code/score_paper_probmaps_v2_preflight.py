"""
Paper 3 offline scoring suite for saved probability maps.

Inputs already produced by training for Crack500, DeepCrack, CFD, and CrackTree260:
  * <dataset output>/<run_id>/probmaps_val/*.npy
  * <dataset output>/<run_id>/probmaps_test/*.npy
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
  * a complete preflight scan validates all four datasets and every saved
    probability map before any full scoring begins

Dependencies:
    pip install numpy pillow scipy pandas scikit-image matplotlib
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from skimage.morphology import skeletonize

# =============================================================================
# 1. PATHS TO EDIT ONLY IF YOUR FOLDERS MOVE
# =============================================================================
# All four v2 (no-early-stopping, full-100-epoch) output folders live under one base.
OUTPUT_BASE = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Paper\Review Paper 3\Output"
)
CRACK500_OUTPUT_ROOT     = OUTPUT_BASE / "OUTPUT Crack500 Full 100 Epochs"
DEEPCRACK_OUTPUT_ROOT    = OUTPUT_BASE / "OUTPUT DeepCrack Full 100 Epochs"
CFD_OUTPUT_ROOT          = OUTPUT_BASE / "OUTPUT CFD Full 100 Epochs"
CRACKTREE260_OUTPUT_ROOT = OUTPUT_BASE / "OUTPUT CrackTree260 Full 100 Epochs"

DATASET_BASE = Path(r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Dataset")
CRACK500_ROOT     = DATASET_BASE / "Crack500"
DEEPCRACK_ROOT    = DATASET_BASE / "DeepCrack Dataset"
CFD_ROOT          = DATASET_BASE / "CrackForest-dataset-master"
CRACKTREE260_ROOT = DATASET_BASE / "CrackTree260"

SCORING_OUTPUT_ROOT = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Paper\Review Paper 3\OFFLINE_SCORE_OUTPUT"
)

# =============================================================================
# 2. FROZEN SCORING CONSTANTS
# =============================================================================
THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100.0  # 0.01 ... 0.99
THRESHOLD_GRID_TEXT = "0.01:0.99:0.01"
THRESHOLD_COMPARATOR = "p >= t"
PROTOCOL_VERSION = "v2_no_early_stopping"
MANIFEST_THRESHOLD_ABS_TOL = 1e-9
# Allows a manifest value rounded to four decimals while still detecting a real mismatch.
MANIFEST_F1_ABS_TOL = 5e-5
# Ground-truth PNGs are treated as binary after validating that every pixel is
# either near the background endpoint or near the foreground endpoint. This
# safely handles near-binary encodings such as CFD masks with values {0,3,255}
# while still rejecting genuinely grayscale, antialiased, or multiclass masks.
MASK_BINARY_THRESHOLD = 127
MASK_NEAR_BINARY_LOW_MAX = 15
MASK_NEAR_BINARY_HIGH_MIN = 240
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0
BOOTSTRAP_TIE_ABS_TOL = 1e-12
# Preflight is intentionally comprehensive: it opens every mask and every
# validation/test probability map, validates shapes and numeric ranges,
# recomputes each run's validation threshold/F1, and smoke-tests every metric
# family before the expensive full scoring pass starts.
RUN_PREFLIGHT_BEFORE_SCORING = True
PREFLIGHT_REPORT_FILENAME = "preflight_report.json"
PREFLIGHT_PROBABILITY_ABS_TOL = 1e-6
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

# Provenance for any masks normalized from a validated near-binary encoding.
# Paths are stored uniquely so repeated loading across losses/seeds does not
# inflate the count in scoring_manifest.json.
_NEAR_BINARY_MASK_PATHS: set[str] = set()
_NEAR_BINARY_ENCODING_EXAMPLES: dict[str, list[int]] = {}
_WARNED_NEAR_BINARY_ENCODINGS: set[tuple[int, ...]] = set()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    output_root: Path
    val_mask_dir: Path
    test_mask_dir: Path
    expected_val: Optional[int]
    expected_test: Optional[int]


# NOTE: all four datasets now have 3 seeds x 6 losses = 18 runs each (72 total).
# expected_val / expected_test = None means "discover from the split file and just
# report the count" -- set the two new datasets' cardinalities once confirmed, so the
# reporting-checklist item A1 (exact split counts) can be stated in the manuscript.
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
    DatasetSpec(
        name="CFD",
        output_root=CFD_OUTPUT_ROOT,
        val_mask_dir=CFD_ROOT / "seg",
        test_mask_dir=CFD_ROOT / "seg",
        expected_val=None,
        expected_test=None,
    ),
    DatasetSpec(
        name="CrackTree260",
        output_root=CRACKTREE260_OUTPUT_ROOT,
        val_mask_dir=CRACKTREE260_ROOT / "gt",
        test_mask_dir=CRACKTREE260_ROOT / "gt",
        expected_val=None,
        expected_test=None,
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


def resolve_split_path(base_dir: Path, listed_name: str, file_role: str = "file") -> Path:
    """Resolve a filename recorded in a split list against its dataset folder."""
    raw_name = str(listed_name).strip().strip('"').strip("'")
    if not raw_name:
        raise ValueError(f"Empty {file_role} name in split list for base directory {base_dir}")

    listed_path = Path(raw_name.replace("\\", "/"))
    candidates: List[Path] = []

    def add_candidate(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    if listed_path.is_absolute():
        add_candidate(listed_path)
    else:
        add_candidate(base_dir / listed_path)

        base_parts_lower = [part.casefold() for part in base_dir.parts]
        listed_parts = list(listed_path.parts)
        listed_parts_lower = [part.casefold() for part in listed_parts]
        max_overlap = min(len(base_parts_lower), max(0, len(listed_parts_lower) - 1))
        for overlap in range(max_overlap, 0, -1):
            if base_parts_lower[-overlap:] == listed_parts_lower[:overlap]:
                remainder = listed_parts[overlap:]
                if remainder:
                    add_candidate(base_dir / Path(*remainder))
                break

        for parent in base_dir.parents:
            add_candidate(parent / listed_path)

        add_candidate(base_dir / listed_path.name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    attempted = "\n    ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve {file_role} listed as {listed_name!r} from base directory "
        f"{base_dir}. Attempted:\n    {attempted}"
    )


def load_mask(mask_path: Path) -> np.ndarray:
    """Load and validate a binary ground-truth mask."""
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    with Image.open(mask_path) as image:
        m = np.asarray(image.convert("L"))

    if m.ndim != 2:
        raise ValueError(f"Ground-truth mask must be HxW, got {m.shape}: {mask_path}")

    unique_values = np.unique(m)

    if np.all(np.isin(unique_values, np.array([0, 1], dtype=unique_values.dtype))):
        return m > 0

    if np.all(np.isin(unique_values, np.array([0, 255], dtype=unique_values.dtype))):
        return m > MASK_BINARY_THRESHOLD

    near_background = unique_values <= MASK_NEAR_BINARY_LOW_MAX
    near_foreground = unique_values >= MASK_NEAR_BINARY_HIGH_MIN
    if np.all(np.logical_or(near_background, near_foreground)):
        resolved_path = str(mask_path.resolve())
        _NEAR_BINARY_MASK_PATHS.add(resolved_path)
        _NEAR_BINARY_ENCODING_EXAMPLES.setdefault(
            resolved_path,
            [int(value) for value in unique_values[:20]],
        )

        encoding_key = tuple(int(value) for value in unique_values.tolist())
        if encoding_key not in _WARNED_NEAR_BINARY_ENCODINGS:
            _WARNED_NEAR_BINARY_ENCODINGS.add(encoding_key)
            preview = list(encoding_key[:20])
            suffix = "" if len(encoding_key) <= 20 else f" ... ({len(encoding_key)} unique values total)"
            print(
                "warning: validated near-binary mask encoding "
                f"{preview}{suffix}; binarizing at > {MASK_BINARY_THRESHOLD}. "
                f"First observed in: {mask_path}"
            )

        return m > MASK_BINARY_THRESHOLD

    preview = unique_values[:20].tolist()
    suffix = "" if unique_values.size <= 20 else f" ... ({unique_values.size} unique values total)"
    raise ValueError(
        f"Unexpected mask encoding in {mask_path}. Expected exact binary values "
        f"{{0,1}} or {{0,255}}, or validated near-binary endpoint values within "
        f"[0,{MASK_NEAR_BINARY_LOW_MAX}] and "
        f"[{MASK_NEAR_BINARY_HIGH_MIN},255]. Found {preview}{suffix}."
    )


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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution_name: str) -> str:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def resolve_git_provenance(script_path: Path) -> dict:
    """Resolve the commit from PAPER3_GIT_COMMIT first, then the local repository."""
    env_commit = os.environ.get("PAPER3_GIT_COMMIT", "").strip()
    if env_commit:
        return {
            "commit": env_commit,
            "source": "PAPER3_GIT_COMMIT",
            "repository_root": None,
            "working_tree_dirty": None,
        }

    try:
        root_result = subprocess.run(
            ["git", "-C", str(script_path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        repo_root = root_result.stdout.strip()
        commit_result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "-C", repo_root, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "commit": commit_result.stdout.strip(),
            "source": "git_rev_parse",
            "repository_root": repo_root,
            "working_tree_dirty": bool(status_result.stdout.strip()),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "commit": "UNSET",
            "source": "unavailable",
            "repository_root": None,
            "working_tree_dirty": None,
        }


def environment_versions() -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scipy": package_version("scipy"),
        "Pillow": package_version("Pillow"),
        "scikit_image": package_version("scikit-image"),
        "matplotlib": package_version("matplotlib"),
    }


def current_output_artifacts(root: Path, started_epoch: float) -> dict:
    """Hash CSV/JSON/PNG/PDF artifacts written during this scoring invocation."""
    artifacts = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "scoring_manifest.json":
            continue
        if path.suffix.lower() not in {".csv", ".json", ".png", ".pdf"}:
            continue
        if path.stat().st_mtime < started_epoch - 2.0:
            continue
        rel = path.relative_to(root).as_posix()
        artifacts[rel] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return artifacts


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
    edges = np.linspace(0.0, 1.0, 101)
    pos_hist = np.zeros(100, dtype=np.float64)
    all_hist = np.zeros(100, dtype=np.float64)
    n_pos = 0.0

    for item in items:
        prob = load_prob(prob_dir / f"{item.stem}.npy")
        gt = load_mask(resolve_split_path(mask_dir, item.mask_name, "mask"))
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
    tp, predpos, n_pos = _hist_counts_for_threshold_grid(prob_dir, mask_dir, items)
    f1 = (2.0 * tp) / (predpos + n_pos + EPS)
    f1r = np.round(f1, 4)
    candidates = np.flatnonzero(f1r == f1r.max())
    order = np.lexsort((THRESHOLDS[candidates], np.abs(THRESHOLDS[candidates] - 0.5)))
    best_i = candidates[order[0]]
    return float(THRESHOLDS[best_i]), float(f1[best_i])


def select_threshold_per_image(prob: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
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
        gt = load_mask(resolve_split_path(mask_dir, item.mask_name, "mask"))
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
        gt = load_mask(resolve_split_path(mask_dir, item.mask_name, "mask"))
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


def _preflight_load_probability(prob_path: Path, expected_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, float]:
    if not prob_path.exists():
        raise FileNotFoundError(f"Missing probability map: {prob_path}")
    try:
        p = np.load(prob_path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Could not load probability map {prob_path}: {exc}") from exc
    if p.ndim != 2:
        raise ValueError(f"Probability map must be HxW, got {p.shape}: {prob_path}")
    if tuple(p.shape) != tuple(expected_shape):
        raise ValueError(
            f"Shape mismatch: probability map {prob_path} has {p.shape}, "
            f"but its mask has {expected_shape}"
        )
    if not np.issubdtype(p.dtype, np.number):
        raise TypeError(f"Probability map must have a numeric dtype, got {p.dtype}: {prob_path}")
    if not np.isfinite(p).all():
        bad_count = int((~np.isfinite(p)).sum())
        raise ValueError(f"Probability map contains {bad_count} non-finite values: {prob_path}")
    p_min = float(np.min(p))
    p_max = float(np.max(p))
    if p_min < -PREFLIGHT_PROBABILITY_ABS_TOL or p_max > 1.0 + PREFLIGHT_PROBABILITY_ABS_TOL:
        raise ValueError(
            f"Probability map values fall outside [0,1]: min={p_min:.9g}, "
            f"max={p_max:.9g}, tolerance={PREFLIGHT_PROBABILITY_ABS_TOL:g}: {prob_path}"
        )
    return p.astype(np.float32, copy=False), p_min, p_max


def _preflight_threshold_from_histograms(pos_hist: np.ndarray, all_hist: np.ndarray, n_pos: float) -> Tuple[float, float]:
    pos_tail = np.cumsum(pos_hist[::-1])[::-1][1:100]
    all_tail = np.cumsum(all_hist[::-1])[::-1][1:100]
    f1 = (2.0 * pos_tail) / (all_tail + n_pos + EPS)
    f1_rounded = np.round(f1, 4)
    candidates = np.flatnonzero(f1_rounded == f1_rounded.max())
    order = np.lexsort((THRESHOLDS[candidates], np.abs(THRESHOLDS[candidates] - 0.5)))
    best_index = int(candidates[order[0]])
    return float(THRESHOLDS[best_index]), float(f1[best_index])


def _preflight_validate_split(*, dataset_name: str, split_name: str, mask_dir: Path, items: Sequence[SplitItem]) -> Tuple[List[np.ndarray], List[Path]]:
    if not items:
        raise RuntimeError(f"{dataset_name}: {split_name} split is empty")
    stems = [item.stem for item in items]
    duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicate_stems:
        raise RuntimeError(f"{dataset_name}: duplicate probability-map stems in {split_name} split: {duplicate_stems[:20]}")
    masks: List[np.ndarray] = []
    mask_paths: List[Path] = []
    for item in items:
        mask_path = resolve_split_path(mask_dir, item.mask_name, "mask")
        mask = load_mask(mask_path)
        if mask.ndim != 2:
            raise ValueError(f"{dataset_name}: {split_name} mask must be HxW, got {mask.shape}: {mask_path}")
        masks.append(mask)
        mask_paths.append(mask_path)
    return masks, mask_paths


def _preflight_verify_probability_file_set(*, dataset_name: str, run_id: str, split_name: str, prob_dir: Path, items: Sequence[SplitItem]) -> None:
    expected = {f"{item.stem}.npy" for item in items}
    actual = {path.name for path in prob_dir.glob("*.npy")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"{dataset_name}/{run_id}: {split_name} probability-map filename mismatch; "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )


def _preflight_smoke_test_metrics(prob: np.ndarray, gt: np.ndarray, threshold: float) -> dict:
    pred = threshold_pred(prob, threshold)
    overlap = image_overlap_scores(pred, gt)
    relaxed = {str(r): relaxed_f1(pred, gt, r) for r in RELAXED_RADII}
    boundary = boundary_f1(pred, gt, BOUNDARY_THETA)
    cldice_value = cldice_metric(pred, gt)
    components = component_count(pred)
    skeleton_length = int(skeletonize(pred).astype(bool).sum())
    ois_threshold, ois_f1 = select_threshold_per_image(prob, gt)
    values = list(overlap.values()) + list(relaxed.values()) + [boundary, cldice_value, float(components), float(skeleton_length), ois_threshold, ois_f1]
    if not all(np.isfinite(value) for value in values):
        raise RuntimeError("Metric smoke test produced a non-finite value")
    return {
        "threshold": threshold,
        "overlap": overlap,
        "relaxed_f1": relaxed,
        "boundary_f1_theta2": boundary,
        "cldice": cldice_value,
        "components_pred": components,
        "skeleton_length_pred": skeleton_length,
        "ois_threshold": ois_threshold,
        "ois_f1": ois_f1,
    }


def preflight_one_dataset(spec: DatasetSpec) -> dict:
    dataset_started = time.time()
    print(f"\n[PREFLIGHT] {spec.name}")
    if not spec.output_root.exists():
        raise FileNotFoundError(f"{spec.name}: missing output root: {spec.output_root}")
    if not spec.val_mask_dir.exists():
        raise FileNotFoundError(f"{spec.name}: missing validation mask directory: {spec.val_mask_dir}")
    if not spec.test_mask_dir.exists():
        raise FileNotFoundError(f"{spec.name}: missing test mask directory: {spec.test_mask_dir}")

    val_split_path = spec.output_root / "val_files.txt"
    test_split_path = spec.output_root / "test_files.txt"
    val_items = read_split_file(val_split_path)
    test_items = read_split_file(test_split_path)
    if spec.expected_val is not None and len(val_items) != spec.expected_val:
        raise RuntimeError(f"{spec.name}: val list has {len(val_items)}, expected {spec.expected_val}")
    if spec.expected_test is not None and len(test_items) != spec.expected_test:
        raise RuntimeError(f"{spec.name}: test list has {len(test_items)}, expected {spec.expected_test}")

    val_masks, val_mask_paths = _preflight_validate_split(dataset_name=spec.name, split_name="validation", mask_dir=spec.val_mask_dir, items=val_items)
    test_masks, test_mask_paths = _preflight_validate_split(dataset_name=spec.name, split_name="test", mask_dir=spec.test_mask_dir, items=test_items)
    runs = discover_runs(spec.output_root)
    if not runs:
        raise RuntimeError(f"{spec.name}: no run directories found in {spec.output_root}")

    run_rows: List[dict] = []
    dataset_probability_min = float("inf")
    dataset_probability_max = float("-inf")
    total_probability_maps = 0
    histogram_edges = np.linspace(0.0, 1.0, 101)

    for run_number, run_dir in enumerate(runs, start=1):
        manifest = read_manifest(run_dir)
        run_id = manifest.get("run_id", run_dir.name)
        loss = manifest.get("loss", run_id.rsplit("_seed", 1)[0])
        seed = int(manifest.get("seed", 0))
        print(f"    [{run_number}/{len(runs)}] {run_id}")
        verify_run_complete(run_dir, len(val_items), len(test_items))
        val_prob_dir = run_dir / "probmaps_val"
        test_prob_dir = run_dir / "probmaps_test"
        _preflight_verify_probability_file_set(dataset_name=spec.name, run_id=run_id, split_name="validation", prob_dir=val_prob_dir, items=val_items)
        _preflight_verify_probability_file_set(dataset_name=spec.name, run_id=run_id, split_name="test", prob_dir=test_prob_dir, items=test_items)

        pos_hist = np.zeros(100, dtype=np.float64)
        all_hist = np.zeros(100, dtype=np.float64)
        n_pos = 0.0
        run_probability_min = float("inf")
        run_probability_max = float("-inf")
        for item, gt in zip(val_items, val_masks):
            prob, p_min, p_max = _preflight_load_probability(val_prob_dir / f"{item.stem}.npy", gt.shape)
            pos_hist += np.histogram(prob[gt], bins=histogram_edges)[0]
            all_hist += np.histogram(prob, bins=histogram_edges)[0]
            n_pos += float(gt.sum())
            run_probability_min = min(run_probability_min, p_min)
            run_probability_max = max(run_probability_max, p_max)
            total_probability_maps += 1

        val_threshold, val_f1 = _preflight_threshold_from_histograms(pos_hist, all_hist, n_pos)
        manifest_threshold_raw = manifest.get("best_val_threshold")
        manifest_f1_raw = manifest.get("best_val_f1")
        if manifest_threshold_raw is None or manifest_f1_raw is None:
            raise RuntimeError(f"{spec.name}/{run_id}: manifest is missing validation provenance")
        manifest_threshold = float(manifest_threshold_raw)
        manifest_f1 = float(manifest_f1_raw)
        threshold_abs_diff = abs(manifest_threshold - val_threshold)
        f1_abs_diff = abs(manifest_f1 - val_f1)
        if threshold_abs_diff > MANIFEST_THRESHOLD_ABS_TOL:
            raise RuntimeError(f"{spec.name}/{run_id}: validation-threshold provenance mismatch")
        if f1_abs_diff > MANIFEST_F1_ABS_TOL:
            raise RuntimeError(f"{spec.name}/{run_id}: validation-F1 provenance mismatch")

        test_pos_hist = np.zeros(100, dtype=np.float64)
        test_all_hist = np.zeros(100, dtype=np.float64)
        test_n_pos = 0.0
        smoke_test: Optional[dict] = None
        for item_index, (item, gt) in enumerate(zip(test_items, test_masks)):
            prob, p_min, p_max = _preflight_load_probability(test_prob_dir / f"{item.stem}.npy", gt.shape)
            test_pos_hist += np.histogram(prob[gt], bins=histogram_edges)[0]
            test_all_hist += np.histogram(prob, bins=histogram_edges)[0]
            test_n_pos += float(gt.sum())
            run_probability_min = min(run_probability_min, p_min)
            run_probability_max = max(run_probability_max, p_max)
            total_probability_maps += 1
            if item_index == 0:
                smoke_test = _preflight_smoke_test_metrics(prob, gt, val_threshold)
        ods_threshold, ods_f1 = _preflight_threshold_from_histograms(test_pos_hist, test_all_hist, test_n_pos)
        dataset_probability_min = min(dataset_probability_min, run_probability_min)
        dataset_probability_max = max(dataset_probability_max, run_probability_max)
        run_rows.append({
            "run_id": run_id,
            "loss": loss,
            "seed": seed,
            "validation_threshold_recomputed": val_threshold,
            "validation_f1_recomputed": val_f1,
            "manifest_validation_threshold": manifest_threshold,
            "manifest_validation_f1": manifest_f1,
            "threshold_abs_diff": threshold_abs_diff,
            "f1_abs_diff": f1_abs_diff,
            "test_ods_threshold_smoke": ods_threshold,
            "test_ods_f1_smoke": ods_f1,
            "probability_min": run_probability_min,
            "probability_max": run_probability_max,
            "metric_smoke_test_first_test_image": smoke_test,
        })

    elapsed = time.time() - dataset_started
    return {
        "dataset": spec.name,
        "status": "PASS",
        "duration_seconds": round(elapsed, 2),
        "output_root": str(spec.output_root.resolve()),
        "validation_count": len(val_items),
        "test_count": len(test_items),
        "validation_mask_directory": str(spec.val_mask_dir.resolve()),
        "test_mask_directory": str(spec.test_mask_dir.resolve()),
        "validation_split_sha256": sha256_file(val_split_path),
        "test_split_sha256": sha256_file(test_split_path),
        "validation_mask_examples": [str(path.resolve()) for path in val_mask_paths[:3]],
        "test_mask_examples": [str(path.resolve()) for path in test_mask_paths[:3]],
        "run_count": len(runs),
        "probability_map_count": total_probability_maps,
        "probability_min": dataset_probability_min,
        "probability_max": dataset_probability_max,
        "runs": run_rows,
    }


def run_preflight_all_datasets(output_root: Path) -> dict:
    preflight_started_epoch = time.time()
    report = {"status": "RUNNING", "started_utc": utc_now_iso(), "protocol_version": PROTOCOL_VERSION, "datasets": [], "errors": []}
    for spec in DATASETS:
        try:
            report["datasets"].append(preflight_one_dataset(spec))
        except Exception as exc:
            report["errors"].append({"dataset": spec.name, "error_type": type(exc).__name__, "message": str(exc)})
            report["datasets"].append({"dataset": spec.name, "status": "FAIL", "error_type": type(exc).__name__, "message": str(exc)})
    report["finished_utc"] = utc_now_iso()
    report["duration_seconds"] = round(time.time() - preflight_started_epoch, 2)
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    report["normalized_near_binary_mask_count"] = len(_NEAR_BINARY_MASK_PATHS)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / PREFLIGHT_REPORT_FILENAME
    temp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    temp_path.replace(report_path)
    if report["errors"]:
        formatted = "\n".join(f"  - {row['dataset']}: {row['error_type']}: {row['message']}" for row in report["errors"])
        raise RuntimeError("PRE-SCORING PREFLIGHT FAILED. Full scoring was not started.\n" + formatted)
    return report


def score_one_dataset(spec: DatasetSpec) -> Tuple[List[dict], List[dict], List[dict], List[dict], List[dict], dict]:
    print(f"\n=== {spec.name} ===")
    val_split_path = spec.output_root / "val_files.txt"
    test_split_path = spec.output_root / "test_files.txt"
    val_items = read_split_file(val_split_path)
    test_items = read_split_file(test_split_path)
    runs = discover_runs(spec.output_root)
    if not runs:
        raise RuntimeError(f"No run directories found in {spec.output_root}")

    threshold_rows: List[dict] = []
    overlap_summary_rows: List[dict] = []
    overlap_per_image_rows: List[dict] = []
    advanced_summary_rows: List[dict] = []
    advanced_per_image_rows: List[dict] = []
    run_provenance: List[dict] = []

    for run_dir in runs:
        manifest_path = run_dir / "manifest.json"
        manifest = read_manifest(run_dir)
        run_id = manifest.get("run_id", run_dir.name)
        loss = manifest.get("loss", run_id.rsplit("_seed", 1)[0])
        seed = int(manifest.get("seed", 0))
        verify_run_complete(run_dir, len(val_items), len(test_items))
        val_prob_dir = run_dir / "probmaps_val"
        test_prob_dir = run_dir / "probmaps_test"

        val_thr, val_f1 = select_threshold_global(val_prob_dir, spec.val_mask_dir, val_items)
        manifest_thr = float(manifest.get("best_val_threshold"))
        manifest_f1 = float(manifest.get("best_val_f1"))
        threshold_abs_diff = abs(manifest_thr - val_thr)
        f1_abs_diff = abs(manifest_f1 - val_f1)
        if threshold_abs_diff > MANIFEST_THRESHOLD_ABS_TOL or f1_abs_diff > MANIFEST_F1_ABS_TOL:
            raise RuntimeError(f"{spec.name}/{run_id}: validation provenance mismatch")

        threshold_rows.append({
            "dataset": spec.name,
            "run_id": run_id,
            "loss": loss,
            "seed": seed,
            "val_selected_threshold_recomputed": round(val_thr, 2),
            "val_selected_f1_recomputed": val_f1,
            "manifest_best_val_threshold": manifest_thr,
            "manifest_best_val_f1": manifest_f1,
            "threshold_abs_diff": threshold_abs_diff,
            "f1_abs_diff": f1_abs_diff,
            "threshold_grid": THRESHOLD_GRID_TEXT,
            "threshold_comparator": THRESHOLD_COMPARATOR,
            "threshold_match_manifest": True,
            "f1_match_manifest": True,
        })
        run_provenance.append({
            "run_id": run_id,
            "loss": loss,
            "seed": seed,
            "run_directory": str(run_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "best_checkpoint_path": str((run_dir / "best.pt").resolve()),
            "validation_probability_maps": len(list(val_prob_dir.glob("*.npy"))),
            "test_probability_maps": len(list(test_prob_dir.glob("*.npy"))),
            "validated_threshold": val_thr,
            "validated_f1": val_f1,
        })

        ods_thr, ods_f1 = select_threshold_global(test_prob_dir, spec.test_mask_dir, test_items)
        ois_thresholds: Dict[str, float] = {}
        for item in test_items:
            prob = load_prob(test_prob_dir / f"{item.stem}.npy")
            gt = load_mask(resolve_split_path(spec.test_mask_dir, item.mask_name, "mask"))
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

    dataset_provenance = {
        "dataset": spec.name,
        "output_root": str(spec.output_root.resolve()),
        "validation_mask_directory": str(spec.val_mask_dir.resolve()),
        "test_mask_directory": str(spec.test_mask_dir.resolve()),
        "validation_split_file": str(val_split_path.resolve()),
        "validation_split_sha256": sha256_file(val_split_path),
        "test_split_file": str(test_split_path.resolve()),
        "test_split_sha256": sha256_file(test_split_path),
        "validation_count": len(val_items),
        "test_count": len(test_items),
        "run_count": len(run_provenance),
        "run_ids": [r["run_id"] for r in run_provenance],
        "runs": run_provenance,
    }
    return threshold_rows, overlap_summary_rows, overlap_per_image_rows, advanced_summary_rows, advanced_per_image_rows, dataset_provenance


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
    primary_overlap = overlap_df[overlap_df["threshold_convention"] == "val_selected"].copy()
    keep_overlap = ["f1_global", "iou_global", "precision_global", "recall_global", "f1_per_image_mean", "iou_per_image_mean"]
    keep_advanced = [
        "relaxed_f1_r0_per_image_mean", "relaxed_f1_r2_per_image_mean", "relaxed_f1_r3_per_image_mean",
        "relaxed_f1_r4_per_image_mean", "relaxed_f1_r5_per_image_mean", "boundary_f1_theta2_per_image_mean",
        "cldice_per_image_mean", "fragmentation_abs_error_mean", "area_error_pct_dataset", "skeleton_length_error_pct_dataset",
    ]
    merged = primary_overlap.merge(advanced_df, on=["dataset", "run_id", "loss", "seed", "threshold_convention", "threshold", "n_images"], how="left", suffixes=("", "_adv"))
    metric_cols = keep_overlap + keep_advanced
    grouped = merged.groupby(["dataset", "loss"], sort=False)[metric_cols]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=1).add_suffix("_std")
    out = pd.concat([mean_df, std_df], axis=1).reset_index()
    out["loss_order"] = out["loss"].map({k: i for i, k in enumerate(LOSS_ORDER)})
    return out.sort_values(["dataset", "loss_order"]).drop(columns=["loss_order"])


# =============================================================================
# 10. DERIVED ANALYSES + FIGURES
# =============================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

try:
    from scipy.stats import kendalltau
    _HAVE_KENDALL = True
except Exception:
    _HAVE_KENDALL = False

LOSS_LABEL = {"bce": "BCE", "focal": "Focal", "bce_dice": "BCE+Dice", "focal_tversky": "Focal Tversky", "dice_boundary": "Dice+Boundary", "dice_cldice": "Dice+clDice"}
SCORE_ORIENT = {
    "f1_global": +1, "iou_global": +1,
    "boundary_f1_theta2_per_image_mean": +1, "cldice_per_image_mean": +1,
    "fragmentation_abs_error_mean": -1, "area_error_pct_dataset": -1,
    "skeleton_length_error_pct_dataset": -1,
}
KENDALL_METRICS = list(SCORE_ORIENT.keys())
KENDALL_LABEL = {"f1_global": "F1", "iou_global": "IoU", "boundary_f1_theta2_per_image_mean": "BoundaryF1", "cldice_per_image_mean": "clDice", "fragmentation_abs_error_mean": "Frag", "area_error_pct_dataset": "Area", "skeleton_length_error_pct_dataset": "SkelLen"}


def _order_losses(losses):
    return sorted(losses, key=lambda l: LOSS_ORDER.index(l) if l in LOSS_ORDER else 999)


def _mean_over_seeds(df, value_cols):
    keep = ["dataset", "loss"] + [c for c in value_cols if c in df.columns]
    return df[keep].groupby(["dataset", "loss"], sort=False).mean(numeric_only=True)


def threshold_convention_deltas(overlap_df: pd.DataFrame) -> pd.DataFrame:
    piv = overlap_df.pivot_table(index=["dataset", "loss", "seed"], columns="threshold_convention", values="f1_global", aggfunc="first").reset_index()
    for c in ["fixed_0.5", "val_selected", "ODS_test_oracle", "OIS_test_oracle"]:
        if c not in piv.columns:
            piv[c] = np.nan
    piv["d_val_minus_fixed"] = piv["val_selected"] - piv["fixed_0.5"]
    piv["d_ODS_minus_val"] = piv["ODS_test_oracle"] - piv["val_selected"]
    piv["d_OIS_minus_fixed"] = piv["OIS_test_oracle"] - piv["fixed_0.5"]
    piv["d_OIS_minus_val"] = piv["OIS_test_oracle"] - piv["val_selected"]
    agg = piv.groupby(["dataset", "loss"], sort=False).agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns]
    agg["loss_o"] = agg["loss"].map(lambda l: LOSS_ORDER.index(l) if l in LOSS_ORDER else 999)
    return agg.sort_values(["dataset", "loss_o"]).drop(columns="loss_o")


def tolerance_ratios(advanced_df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"relaxed_f1_r{r}_per_image_mean" for r in RELAXED_RADII]
    means = _mean_over_seeds(advanced_df, cols).reset_index()
    rows = []
    for ds, sub in means.groupby("dataset", sort=False):
        r0 = sub.set_index("loss")["relaxed_f1_r0_per_image_mean"]
        r5 = sub.set_index("loss")["relaxed_f1_r5_per_image_mean"]
        top_loss = r0.idxmax()
        top_gain = float(r5[top_loss] - r0[top_loss])
        full_span_r0 = float(r0.max() - r0.min())
        rows.append({"dataset": ds, "top_loss_at_r0": top_loss, "top_relaxed_f1_r0": float(r0[top_loss]), "top_relaxed_f1_r5": float(r5[top_loss]), "top_gain_r0_to_r5": top_gain, "six_loss_span_at_r0": full_span_r0, "ratio_gain_over_span": (top_gain / full_span_r0) if full_span_r0 > 0 else np.nan})
    return pd.DataFrame(rows)


def kendall_tau_matrices(overlap_df, advanced_df):
    ov = overlap_df[overlap_df["threshold_convention"] == "val_selected"]
    ov_m = _mean_over_seeds(ov, ["f1_global", "iou_global"])
    ad_m = _mean_over_seeds(advanced_df, [c for c in KENDALL_METRICS if c not in ov_m.columns])
    wide = ov_m.join(ad_m, how="outer")
    out = {}
    for ds, sub in wide.groupby(level="dataset"):
        s = sub.droplevel("dataset")
        oriented = pd.DataFrame({m: SCORE_ORIENT[m] * s[m] for m in KENDALL_METRICS if m in s})
        mets = list(oriented.columns)
        M = np.full((len(mets), len(mets)), np.nan)
        for i, a in enumerate(mets):
            for j, b in enumerate(mets):
                va, vb = oriented[a].values, oriented[b].values
                ok = ~(np.isnan(va) | np.isnan(vb))
                if ok.sum() >= 3 and _HAVE_KENDALL:
                    M[i, j] = kendalltau(va[ok], vb[ok]).correlation
        out[ds] = pd.DataFrame(M, index=[KENDALL_LABEL[m] for m in mets], columns=[KENDALL_LABEL[m] for m in mets])
    return out


def aggregation_flips(overlap_df: pd.DataFrame) -> pd.DataFrame:
    ov = overlap_df[overlap_df["threshold_convention"] == "val_selected"]
    means = _mean_over_seeds(ov, ["f1_global", "f1_per_image_mean", "iou_global", "iou_per_image_mean"]).reset_index()
    rows = []
    for dataset, subset in means.groupby("dataset", sort=False):
        for global_col, image_col, metric in [("f1_global", "f1_per_image_mean", "F1"), ("iou_global", "iou_per_image_mean", "IoU")]:
            global_rank = subset.set_index("loss")[global_col].rank(ascending=False, method="min")
            image_rank = subset.set_index("loss")[image_col].rank(ascending=False, method="min")
            for loss in global_rank.index:
                if global_rank[loss] != image_rank[loss]:
                    rows.append({"dataset": dataset, "metric": metric, "loss": loss, "rank_global": int(global_rank[loss]), "rank_per_image": int(image_rank[loss])})
    return pd.DataFrame(rows)


def aggregation_flip_pairs(overlap_df: pd.DataFrame) -> pd.DataFrame:
    ov = overlap_df[overlap_df["threshold_convention"] == "val_selected"]
    means = _mean_over_seeds(ov, ["f1_global", "f1_per_image_mean", "iou_global", "iou_per_image_mean"]).reset_index()
    rows = []
    for dataset, subset in means.groupby("dataset", sort=False):
        indexed = subset.set_index("loss")
        ordered_losses = _order_losses(indexed.index)
        for metric, global_col, image_col in [("F1", "f1_global", "f1_per_image_mean"), ("IoU", "iou_global", "iou_per_image_mean")]:
            for loss_a, loss_b in combinations(ordered_losses, 2):
                global_a = float(indexed.loc[loss_a, global_col]); global_b = float(indexed.loc[loss_b, global_col])
                image_a = float(indexed.loc[loss_a, image_col]); image_b = float(indexed.loc[loss_b, image_col])
                global_delta = global_a - global_b; image_delta = image_a - image_b
                if global_delta * image_delta < 0.0:
                    rows.append({"dataset": dataset, "metric": metric, "loss_a": loss_a, "loss_b": loss_b, "global_value_a": global_a, "global_value_b": global_b, "global_delta_a_minus_b": global_delta, "per_image_value_a": image_a, "per_image_value_b": image_b, "per_image_delta_a_minus_b": image_delta})
    return pd.DataFrame(rows)


def _global_metric_from_counts(tp: float, fp: float, fn: float, metric: str) -> float:
    if metric == "F1":
        denominator = 2.0 * tp + fp + fn
        return 1.0 if denominator == 0.0 else float(2.0 * tp / denominator)
    if metric == "IoU":
        denominator = tp + fp + fn
        return 1.0 if denominator == 0.0 else float(tp / denominator)
    raise ValueError(f"Unsupported aggregation-bootstrap metric: {metric}")


def paired_image_bootstrap(per_image_overlap_df: pd.DataFrame, dataset: str, loss_a: str, loss_b: str, metric: str, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Optional[dict]:
    per_image_col = "f1" if metric == "F1" else "iou"
    df = per_image_overlap_df[(per_image_overlap_df["dataset"] == dataset) & (per_image_overlap_df["threshold_convention"] == "val_selected") & (per_image_overlap_df["loss"].isin([loss_a, loss_b]))].copy()
    if df.empty:
        return None
    seeds = sorted(set(df.loc[df["loss"] == loss_a, "seed"].astype(int).unique().tolist()))
    images = sorted(set(df.loc[(df["loss"] == loss_a) & (df["seed"] == seeds[0]), "image"].tolist()))
    n_images = len(images)
    arrays: Dict[Tuple[str, int], np.ndarray] = {}
    for loss in (loss_a, loss_b):
        for seed_value in seeds:
            subset = df[(df["loss"] == loss) & (df["seed"] == seed_value)].set_index("image").loc[images, ["tp", "fp", "fn", per_image_col]]
            arrays[(loss, seed_value)] = subset.to_numpy(dtype=np.float64)
    def aggregate_loss(loss: str, sampled_indices: np.ndarray) -> Tuple[float, float]:
        global_values = []; per_image_values = []
        for seed_value in seeds:
            sampled = arrays[(loss, seed_value)][sampled_indices]
            global_values.append(_global_metric_from_counts(float(sampled[:,0].sum()), float(sampled[:,1].sum()), float(sampled[:,2].sum()), metric))
            per_image_values.append(float(sampled[:,3].mean()))
        return float(np.mean(global_values)), float(np.mean(per_image_values))
    full_indices = np.arange(n_images, dtype=np.int64)
    global_a, image_a = aggregate_loss(loss_a, full_indices); global_b, image_b = aggregate_loss(loss_b, full_indices)
    rng = np.random.default_rng(seed)
    global_deltas = np.empty(n_boot); image_deltas = np.empty(n_boot)
    for i in range(n_boot):
        sampled_indices = rng.integers(0, n_images, size=n_images)
        ga, ia = aggregate_loss(loss_a, sampled_indices); gb, ib = aggregate_loss(loss_b, sampled_indices)
        global_deltas[i] = ga - gb; image_deltas[i] = ia - ib
    global_ci_low, global_ci_high = np.quantile(global_deltas, [0.025, 0.975]); image_ci_low, image_ci_high = np.quantile(image_deltas, [0.025, 0.975])
    return {"dataset": dataset, "metric": metric, "loss_a": loss_a, "loss_b": loss_b, "n_images": n_images, "n_seeds": len(seeds), "n_bootstrap": n_boot, "bootstrap_seed": seed, "global_value_a": global_a, "global_value_b": global_b, "global_delta_a_minus_b": global_a-global_b, "global_delta_ci95_low": float(global_ci_low), "global_delta_ci95_high": float(global_ci_high), "per_image_value_a": image_a, "per_image_value_b": image_b, "per_image_delta_a_minus_b": image_a-image_b, "per_image_delta_ci95_low": float(image_ci_low), "per_image_delta_ci95_high": float(image_ci_high), "ordering_reverses_in_full_sample": bool((global_a-global_b)*(image_a-image_b)<0.0)}


def seed_variance_bands(overlap_df, advanced_df, thresholds={"cldice_per_image_mean":0.02,"f1_global":0.03,"skeleton_length_error_pct_dataset":5.0}):
    ov = overlap_df[overlap_df["threshold_convention"] == "val_selected"]
    metrics = list(thresholds.keys())
    parts = []
    for src in (ov, advanced_df):
        cols = [m for m in metrics if m in src.columns]
        if cols:
            parts.append(src[["dataset", "loss", "seed"] + cols])
    perseed = parts[0]
    for p in parts[1:]:
        perseed = perseed.merge(p, on=["dataset", "loss", "seed"], how="outer")
    rows = []
    for (ds, loss), sub in perseed.groupby(["dataset", "loss"], sort=False):
        for m in metrics:
            if m in sub.columns:
                rows.append({"dataset": ds, "loss": loss, "metric": m, "seed_sd": float(np.nanstd(sub[m].values, ddof=1))})
    sd = pd.DataFrame(rows)
    band = sd.groupby(["dataset", "metric"])["seed_sd"].max().rename("dataset_seed_band_max").reset_index()
    band["manuscript_threshold"] = band["metric"].map(thresholds)
    band["band_exceeds_threshold"] = band["dataset_seed_band_max"] > band["manuscript_threshold"]
    return sd, band


def _savefig(fig, out_dir, name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_tolerance_rank(advanced_df, out_dir):
    if not _HAVE_MPL: return
    cols = [f"relaxed_f1_r{r}_per_image_mean" for r in RELAXED_RADII]
    means = _mean_over_seeds(advanced_df, cols)
    dsets = list(dict.fromkeys(advanced_df["dataset"])); ncol=2; nrow=int(np.ceil(len(dsets)/ncol))
    fig, axes = plt.subplots(nrow,ncol,figsize=(5.2*ncol,3.6*nrow),squeeze=False)
    for k,ds in enumerate(dsets):
        ax=axes[k//ncol][k%ncol]; sub=means.loc[ds]
        ranks=pd.DataFrame({r:sub[f"relaxed_f1_r{r}_per_image_mean"].rank(ascending=False,method="min") for r in RELAXED_RADII})
        for loss in _order_losses(sub.index):
            ax.plot(RELAXED_RADII,[ranks.loc[loss,r] for r in RELAXED_RADII],marker="o",label=LOSS_LABEL.get(loss,loss))
        ax.set_title(ds); ax.set_xlabel("tolerance radius r (px)"); ax.set_ylabel("rank (1=best)"); ax.set_yticks(range(1,7)); ax.invert_yaxis(); ax.set_xticks(RELAXED_RADII); ax.grid(alpha=0.3)
    h,l=axes[0][0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.04)); fig.tight_layout(); _savefig(fig,out_dir,"fig4_tolerance_rank")


def fig_kendall_heatmap(tau_by_ds,out_dir):
    if not _HAVE_MPL: return
    dsets=list(tau_by_ds.keys()); ncol=2; nrow=int(np.ceil(len(dsets)/ncol)); fig,axes=plt.subplots(nrow,ncol,figsize=(4.8*ncol,4.2*nrow),squeeze=False)
    for k,ds in enumerate(dsets):
        ax=axes[k//ncol][k%ncol]; M=tau_by_ds[ds]; im=ax.imshow(M.values,vmin=-1,vmax=1,cmap="RdBu_r"); ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns,rotation=45,ha="right",fontsize=7); ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index,fontsize=7); ax.set_title(ds,fontsize=9)
    fig.colorbar(im,ax=axes,shrink=0.6,label="Kendall tau (score-oriented)"); _savefig(fig,out_dir,"fig5_kendall_tau")


def fig_seed_variance(overlap_df,advanced_df,out_dir):
    if not _HAVE_MPL: return
    ov=overlap_df[overlap_df["threshold_convention"]=="val_selected"]; panels=[("f1_global",ov,"Global F1",False),("cldice_per_image_mean",advanced_df,"clDice",False),("skeleton_length_error_pct_dataset",advanced_df,"Skeleton-length error %",True)]
    dsets=list(dict.fromkeys(overlap_df["dataset"]))
    for ds in dsets:
        fig,axes=plt.subplots(1,3,figsize=(13,3.6))
        for ax,(metric,src,title,is_err) in zip(axes,panels):
            sub=src[src["dataset"]==ds]; g=sub.groupby("loss")[metric].agg(["mean","std"]).reindex(_order_losses(sub["loss"].unique())); x=range(len(g)); ax.bar(x,g["mean"].values,yerr=g["std"].values,capsize=3); ax.set_xticks(list(x)); ax.set_xticklabels([LOSS_LABEL.get(l,l) for l in g.index],rotation=30,ha="right",fontsize=8); ax.set_title(title); ax.grid(axis="y",alpha=0.3)
        fig.suptitle(f"{ds} — seed variance (mean ± SD, 3 seeds)"); fig.tight_layout(); _savefig(fig,out_dir,f"fig6_seed_variance_{ds}")


def main() -> None:
    t0 = time.time()
    started_utc = utc_now_iso()
    script_path = Path(__file__).resolve()
    script_sha256_start = sha256_file(script_path)
    git_provenance = resolve_git_provenance(script_path)
    SCORING_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    preflight_report: Optional[dict] = None
    if RUN_PREFLIGHT_BEFORE_SCORING:
        preflight_report = run_preflight_all_datasets(SCORING_OUTPUT_ROOT)

    all_threshold_rows=[]; all_overlap_summary_rows=[]; all_overlap_per_image_rows=[]; all_advanced_summary_rows=[]; all_advanced_per_image_rows=[]; dataset_provenance_rows=[]
    for spec in DATASETS:
        threshold_rows, overlap_summary_rows, overlap_per_image_rows, advanced_summary_rows, advanced_per_image_rows, dataset_provenance = score_one_dataset(spec)
        all_threshold_rows.extend(threshold_rows); all_overlap_summary_rows.extend(overlap_summary_rows); all_overlap_per_image_rows.extend(overlap_per_image_rows); all_advanced_summary_rows.extend(advanced_summary_rows); all_advanced_per_image_rows.extend(advanced_per_image_rows); dataset_provenance_rows.append(dataset_provenance)

    write_csv(SCORING_OUTPUT_ROOT / "validation_thresholds.csv", all_threshold_rows)
    write_csv(SCORING_OUTPUT_ROOT / "overlap_summary_all_thresholds.csv", all_overlap_summary_rows)
    write_csv(SCORING_OUTPUT_ROOT / "advanced_summary_val_selected.csv", all_advanced_summary_rows)
    if WRITE_PER_IMAGE_OVERLAP: write_csv(SCORING_OUTPUT_ROOT / "per_image_overlap_all_thresholds.csv", all_overlap_per_image_rows)
    if WRITE_PER_IMAGE_ADVANCED: write_csv(SCORING_OUTPUT_ROOT / "per_image_advanced_val_selected.csv", all_advanced_per_image_rows)

    overlap_df=pd.DataFrame(all_overlap_summary_rows); advanced_df=pd.DataFrame(all_advanced_summary_rows); by_loss=make_by_loss_summary(overlap_df,advanced_df); by_loss.to_csv(SCORING_OUTPUT_ROOT/"summary_by_dataset_loss_val_selected.csv",index=False)
    per_image_overlap_df=pd.DataFrame(all_overlap_per_image_rows); fig_dir=SCORING_OUTPUT_ROOT/"figures"
    write_csv(SCORING_OUTPUT_ROOT/"threshold_convention_deltas.csv",threshold_convention_deltas(overlap_df).to_dict("records"))
    write_csv(SCORING_OUTPUT_ROOT/"tolerance_ratios.csv",tolerance_ratios(advanced_df).to_dict("records"))
    tau_by_ds=kendall_tau_matrices(overlap_df,advanced_df)
    for ds,matrix in tau_by_ds.items(): matrix.to_csv(SCORING_OUTPUT_ROOT/f"kendall_tau_{ds}.csv")
    write_csv(SCORING_OUTPUT_ROOT/"aggregation_flips.csv",aggregation_flips(overlap_df).to_dict("records"))
    flip_pairs=aggregation_flip_pairs(overlap_df); write_csv(SCORING_OUTPUT_ROOT/"aggregation_flip_pairs.csv",flip_pairs.to_dict("records"))
    boot_rows=[]
    if not flip_pairs.empty:
        for row in flip_pairs.itertuples(index=False):
            result=paired_image_bootstrap(per_image_overlap_df,dataset=row.dataset,loss_a=row.loss_a,loss_b=row.loss_b,metric=row.metric,n_boot=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
            if result is not None: boot_rows.append(result)
    if boot_rows: write_csv(SCORING_OUTPUT_ROOT/"aggregation_bootstrap.csv",boot_rows)
    seed_sd,seed_band=seed_variance_bands(overlap_df,advanced_df); write_csv(SCORING_OUTPUT_ROOT/"seed_variance_per_loss.csv",seed_sd.to_dict("records")); write_csv(SCORING_OUTPUT_ROOT/"seed_variance_bands.csv",seed_band.to_dict("records"))
    if _HAVE_MPL:
        fig_tolerance_rank(advanced_df,fig_dir); fig_kendall_heatmap(tau_by_ds,fig_dir); fig_seed_variance(overlap_df,advanced_df,fig_dir)

    script_sha256_finish=sha256_file(script_path)
    if script_sha256_finish!=script_sha256_start: raise RuntimeError("The scoring script changed while scoring was in progress.")
    meta={"study":"What to Optimize and How to Measure It: Loss-Metric Alignment for Deep Crack Segmentation","study_component":"controlled demonstration offline scoring","protocol_version":PROTOCOL_VERSION,"started_utc":started_utc,"finished_utc":utc_now_iso(),"duration_seconds":round(time.time()-t0,2),"script":{"path":str(script_path),"sha256":script_sha256_start,"git":git_provenance},"environment":environment_versions(),"preflight":{"enabled":RUN_PREFLIGHT_BEFORE_SCORING,"status":preflight_report.get("status") if preflight_report is not None else "DISABLED"},"scoring_protocol":{"threshold_grid":THRESHOLD_GRID_TEXT,"threshold_comparator":THRESHOLD_COMPARATOR,"threshold_tie_rule":"F1 equal after rounding to four decimals: closest to 0.5, then lower threshold","threshold_conventions":["fixed_0.5_deployable","validation_selected_global_deployable","ODS_test_oracle","OIS_test_oracle_per_image"],"relaxed_f1_radii_px_native_resolution":RELAXED_RADII,"boundary_f1_theta_px_native_resolution":BOUNDARY_THETA,"empty_mask_cases":{"A_both_empty":1.0,"B_gt_empty_pred_nonempty":0.0,"C_gt_nonempty_pred_empty":0.0,"D_both_nonempty":"normal metric definition"}},"metric_implementations":{"skeletonization":"skimage.morphology.skeletonize on binarized masks; no pruning","component_labeling":"scipy.ndimage.label with a 3x3 all-ones structure (8-connectivity)","area_error":"absolute dataset-level relative error of summed positive-pixel counts","skeleton_length_error":"absolute dataset-level relative error of summed unpruned skeleton-pixel counts"},"datasets":dataset_provenance_rows,"scored_run_count":sum(d["run_count"] for d in dataset_provenance_rows)}
    manifest_path=SCORING_OUTPUT_ROOT/"scoring_manifest.json"; temp_manifest_path=manifest_path.with_suffix(".json.tmp")
    with open(temp_manifest_path,"w",encoding="utf-8") as f: json.dump(meta,f,indent=2); f.write("\n")
    temp_manifest_path.replace(manifest_path)
    print("\nAll offline scoring complete.")
    print(f"Output folder: {SCORING_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
