#!/usr/bin/env python
"""Paper 3 full-budget sensitivity experiment: 4 datasets x 6 losses x 3 seeds.

Datasets: Crack500, DeepCrack, CFD, and CrackTree260.
Total runs: 72, trained back-to-back from scratch in separate full-budget output folders.

Purpose
-------
This script removes early stopping entirely. Every run trains for exactly 100 epochs.
The best checkpoint is still selected over all 100 epochs using validation-selected
Dataset-global F1 on the frozen 0.01-0.99 threshold grid. The final epoch checkpoint
is also saved as ``last.pt`` so late-training behavior remains auditable.

Everything else is held fixed relative to the prior Paper 3 pipelines: architecture,
preprocessing, data splits, seeds, augmentations, optimizer, scheduler, loss formulas,
threshold grid, tie rules, native-resolution inference, and float32 probability-map export.

The new output folders are intentionally separate from the original patience-20 runs,
so the original 72-run experiment remains untouched for paired sensitivity analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import PIL
from PIL import Image
import scipy
from scipy.ndimage import distance_transform_edt as edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise SystemExit(
        "segmentation_models_pytorch is required:\n"
        "pip install torch torchvision segmentation-models-pytorch scipy numpy pillow"
    ) from exc

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================
DATASETS: Dict[str, Dict[str, object]] = {
    "Crack500": {
        "source_type": "paired_directories",
        "root": r"E:\Paper03\Crack500",
        "train_image_dir": r"E:\Paper03\Crack500\traincrop\traincrop",
        "train_mask_dir": r"E:\Paper03\Crack500\traincrop\traincrop",
        "val_image_dir": r"E:\Paper03\Crack500\valcrop\valcrop",
        "val_mask_dir": r"E:\Paper03\Crack500\valcrop\valcrop",
        "test_image_dir": r"E:\Paper03\Crack500\testcrop\testcrop",
        "test_mask_dir": r"E:\Paper03\Crack500\testcrop\testcrop",
        "image_ext": ".jpg",
        "mask_ext": ".png",
        "output_root": r"E:\Paper03\OUTPUT Crack500 Full 100 Epochs",
        "expected_train": 1896,
        "expected_val": 348,
        "expected_test": 1124,
    },

    "DeepCrack": {
        "source_type": "split_csv",
        "root": r"E:\Paper03\DeepCrack Dataset",
        "train_csv": r"E:\Paper03\DeepCrack Dataset\splits_paper3\train.csv",
        "val_csv": r"E:\Paper03\DeepCrack Dataset\splits_paper3\val.csv",
        "test_csv": r"E:\Paper03\DeepCrack Dataset\splits_paper3\test.csv",
        "output_root": r"E:\Paper03\OUTPUT DeepCrack Full 100 Epochs",
        "expected_train": 240,
        "expected_val": 60,
        "expected_test": 237,
    },

    "CFD": {
        "source_type": "split_csv",
        "root": r"E:\Paper03\CrackForest-dataset-master",
        "train_csv": r"E:\Paper03\CrackForest-dataset-master\splits_paper3\train.csv",
        "val_csv": r"E:\Paper03\CrackForest-dataset-master\splits_paper3\val.csv",
        "test_csv": r"E:\Paper03\CrackForest-dataset-master\splits_paper3\test.csv",
        "output_root": r"E:\Paper03\OUTPUT CFD Full 100 Epochs",
        "expected_train": 82,
        "expected_val": 12,
        "expected_test": 24,
    },

    "CrackTree260": {
        "source_type": "split_csv",
        "root": r"E:\Paper03\CrackTree260",
        "train_csv": r"E:\Paper03\CrackTree260\splits_paper3\train.csv",
        "val_csv": r"E:\Paper03\CrackTree260\splits_paper3\val.csv",
        "test_csv": r"E:\Paper03\CrackTree260\splits_paper3\test.csv",
        "output_root": r"E:\Paper03\OUTPUT CrackTree260 Full 100 Epochs",
        "expected_train": 182,
        "expected_val": 26,
        "expected_test": 52,
    },
}

# Separate output roots preserve the original patience-20 runs for auditability.

LOSS_ORDER = ["bce", "focal", "bce_dice", "focal_tversky", "dice_boundary", "dice_cldice"]
SEEDS = [0, 1, 2]

CROP_SIZE = 448
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
USE_AMP = True
DETERMINISTIC = False
STRIDE = 32
HASH_SPLIT_FILES = True

# Version/provenance stamp for the full-budget, no-early-stopping protocol.
PROTOCOL_VERSION = "v2_no_early_stopping"
GIT_COMMIT_ENV_VAR = "PAPER3_GIT_COMMIT"

FOCAL_GAMMA = 2.0
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
FT_EXPONENT = 0.75
CLDICE_ALPHA = 0.3
CLDICE_ITERS = 5
BOUNDARY_ALPHA_FLOOR = 0.01
BOUNDARY_ALPHA_STEP = 0.01

SELECT_METRIC = "val_f1_grid_selected_global"
THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100.0
THRESHOLD_COMPARATOR = "p >= t"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = USE_AMP and DEVICE.type == "cuda"
EPS = 1e-6

@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    sample_id: str

# =============================================================================
# 2. REPRODUCIBILITY AND ENVIRONMENT
# =============================================================================
def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@lru_cache(maxsize=1)
def script_sha256() -> str:
    """Return the SHA-256 hash of the exact script file being executed."""
    return sha256_file(Path(__file__).resolve())


@lru_cache(maxsize=1)
def git_commit_id() -> str:
    """Return the repository commit for this script, or an explicit fallback.

    Set the PAPER3_GIT_COMMIT environment variable to a verified commit hash when
    launching outside a Git working tree. No commit identifier is invented.
    """
    override = os.environ.get(GIT_COMMIT_ENV_VAR, "").strip()
    if override:
        return override
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = result.stdout.strip()
        return commit if commit else "unavailable_empty_git_output"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable_not_in_git_repository"


def provenance_info() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": script_sha256(),
        "git_commit_id": git_commit_id(),
        "git_commit_source": (
            f"environment:{GIT_COMMIT_ENV_VAR}"
            if os.environ.get(GIT_COMMIT_ENV_VAR, "").strip()
            else "git rev-parse HEAD or explicit unavailable fallback"
        ),
    }


def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "segmentation_models_pytorch": getattr(smp, "__version__", "unknown"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "amp_enabled": AMP_ENABLED,
        "deterministic": DETERMINISTIC,
        "reproducibility_note": "seed-controlled, bitwise deterministic" if DETERMINISTIC else "seed-controlled, not bitwise deterministic",
    }

# =============================================================================
# 3. SPLITS, HASHES, AND LEAKAGE CHECKS
# =============================================================================
def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_csv(csv_path: Path, root: Path) -> List[Sample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    samples: List[Sample] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"image", "mask"}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{csv_path} must contain image and mask columns. Found: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            image_value = (row.get("image") or "").strip()
            mask_value = (row.get("mask") or "").strip()
            if not image_value or not mask_value:
                raise ValueError(f"Blank image or mask path in {csv_path}, row {row_number}")
            image_path = resolve_path(root, image_value).resolve()
            mask_path = resolve_path(root, mask_value).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image in {csv_path}, row {row_number}: {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask in {csv_path}, row {row_number}: {mask_path}")
            sample_id = image_path.stem
            csv_id = (row.get("id") or "").strip()
            if csv_id and csv_id != sample_id:
                raise ValueError(
                    f"CSV id must equal image stem for scorer compatibility. "
                    f"In {csv_path}, row {row_number}: id='{csv_id}', stem='{sample_id}'."
                )
            samples.append(Sample(image_path, mask_path, sample_id))
    if not samples:
        raise ValueError(f"No samples found in {csv_path}")
    ids = [s.sample_id for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image stems in {csv_path}; probability-map names would collide.")
    return samples


def read_directory_split(
    image_dir: Path,
    mask_dir: Path,
    image_ext: str,
    mask_ext: str,
) -> List[Sample]:
    """Read a split stored as image/mask directories paired by filename stem."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    images = sorted(image_dir.glob(f"*{image_ext}"))
    if not images:
        raise FileNotFoundError(f"No '{image_ext}' images found in {image_dir}")

    samples: List[Sample] = []
    missing_masks: List[str] = []
    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}{mask_ext}"
        if not mask_path.exists():
            missing_masks.append(mask_path.name)
            continue
        samples.append(
            Sample(
                image=image_path.resolve(),
                mask=mask_path.resolve(),
                sample_id=image_path.stem,
            )
        )

    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} masks are missing in {mask_dir}; "
            f"examples: {missing_masks[:5]}"
        )

    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Duplicate image stems in {image_dir}; probability-map names would collide."
        )
    return samples


def validate_no_split_leakage(
    dataset_name: str,
    train: Sequence[Sample],
    val: Sequence[Sample],
    test: Sequence[Sample],
) -> dict:
    """Validate true split leakage while allowing reused filename stems.

    DeepCrack can contain different files in different source directories that
    share the same filename stem (for example, ``11289-1``). A stem collision
    across splits is not leakage by itself because:

    1. the physical image paths differ;
    2. validation and test probability maps are stored in separate folders; and
    3. stems are already required to be unique *within* each split.

    The stem overlaps are therefore recorded for auditability but only exact
    path overlap, byte-identical image content, or an identical image-mask pair
    across splits causes a hard failure.
    """
    splits = {"train": list(train), "val": list(val), "test": list(test)}

    image_path_sets = {
        split_name: {sample.image.resolve() for sample in samples}
        for split_name, samples in splits.items()
    }
    mask_path_sets = {
        split_name: {sample.mask.resolve() for sample in samples}
        for split_name, samples in splits.items()
    }
    id_sets = {
        split_name: {sample.sample_id for sample in samples}
        for split_name, samples in splits.items()
    }

    def pairwise_overlap(sets: Dict[str, set], stringify: bool = False) -> dict:
        pairs = {
            "train_val": sets["train"] & sets["val"],
            "train_test": sets["train"] & sets["test"],
            "val_test": sets["val"] & sets["test"],
        }
        if stringify:
            return {key: sorted(str(value) for value in values) for key, values in pairs.items()}
        return {key: sorted(values) for key, values in pairs.items()}

    image_path_overlap = pairwise_overlap(image_path_sets, stringify=True)
    mask_path_overlap = pairwise_overlap(mask_path_sets, stringify=True)
    sample_id_overlap = pairwise_overlap(id_sets)

    if any(image_path_overlap.values()) or any(mask_path_overlap.values()):
        raise ValueError(
            f"Exact file-path leakage detected for {dataset_name}: "
            f"image_paths={image_path_overlap}, mask_paths={mask_path_overlap}"
        )

    hashes: Dict[str, List[Tuple[str, str, str]]] = {}
    for split_name, samples in splits.items():
        hashes[split_name] = [
            (sha256_file(sample.image), sha256_file(sample.mask), sample.sample_id)
            for sample in samples
        ]

    image_hash_sets = {
        split_name: {image_hash for image_hash, _mask_hash, _sample_id in rows}
        for split_name, rows in hashes.items()
    }
    pair_hash_sets = {
        split_name: {(image_hash, mask_hash) for image_hash, mask_hash, _sample_id in rows}
        for split_name, rows in hashes.items()
    }

    image_sha256_overlap = pairwise_overlap(image_hash_sets)
    pair_sha256_overlap_raw = pairwise_overlap(pair_hash_sets)
    pair_sha256_overlap = {
        key: [f"{image_hash}:{mask_hash}" for image_hash, mask_hash in values]
        for key, values in pair_sha256_overlap_raw.items()
    }

    if any(image_sha256_overlap.values()):
        raise ValueError(
            f"Byte-identical image leakage detected for {dataset_name}: "
            f"{image_sha256_overlap}"
        )
    if any(pair_sha256_overlap.values()):
        raise ValueError(
            f"Byte-identical image-mask pair leakage detected for {dataset_name}: "
            f"{pair_sha256_overlap}"
        )

    return {
        "image_path_overlap": image_path_overlap,
        "mask_path_overlap": mask_path_overlap,
        "sample_id_overlap_allowed": sample_id_overlap,
        "image_sha256_overlap": image_sha256_overlap,
        "image_mask_pair_sha256_overlap": pair_sha256_overlap,
        "decision_rule": (
            "Cross-split filename-stem reuse is recorded but allowed. "
            "Exact path overlap, byte-identical images, and identical image-mask pairs are rejected."
        ),
    }


def write_split_lists(output_root: Path, root: Path, splits: Dict[str, Sequence[Sample]]) -> None:
    for split_name, samples in splits.items():
        lines = []
        for sample in samples:
            try:
                image_text = str(sample.image.relative_to(root))
            except ValueError:
                image_text = str(sample.image)
            try:
                mask_text = str(sample.mask.relative_to(root))
            except ValueError:
                mask_text = str(sample.mask)
            if HASH_SPLIT_FILES:
                lines.append("\t".join([image_text, mask_text, sha256_file(sample.image), sha256_file(sample.mask)]))
            else:
                lines.append(f"{image_text}\t{mask_text}")
        (output_root / f"{split_name}_files.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_expected_counts(dataset_name: str, config: Dict[str, object], train: Sequence[Sample], val: Sequence[Sample], test: Sequence[Sample]) -> None:
    expected = {"train": int(config["expected_train"]), "val": int(config["expected_val"]), "test": int(config["expected_test"])}
    if any(v < 0 for v in expected.values()):
        raise ValueError(f"Expected counts are not frozen for {dataset_name}: {expected}")
    actual = {"train": len(train), "val": len(val), "test": len(test)}
    if actual != expected:
        raise ValueError(f"{dataset_name} split counts mismatch. Expected {expected}; found {actual}")

# =============================================================================
# 4. IMAGES, MASKS, CROPS, AND AUGMENTATION
# =============================================================================
def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    unique = np.unique(mask)
    binary = mask > 0 if np.all(np.isin(unique, [0, 1])) else mask > 127
    return binary.astype(np.float32)


def normalize(image: np.ndarray) -> np.ndarray:
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def compute_sdf(mask: np.ndarray) -> np.ndarray:
    foreground = mask.astype(bool)
    if not foreground.any() or foreground.all():
        return np.zeros(mask.shape, dtype=np.float32)
    return (edt(~foreground) - edt(foreground)).astype(np.float32)


def pad_to_multiple(array: np.ndarray, stride: int = STRIDE) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = array.shape[:2]
    pad_bottom = (stride - height % stride) % stride
    pad_right = (stride - width % stride) % stride
    if pad_bottom == 0 and pad_right == 0:
        return array, (height, width)
    pad_spec = ((0, pad_bottom), (0, pad_right)) + (((0, 0),) if array.ndim == 3 else ())
    return np.pad(array, pad_spec, mode="reflect"), (height, width)


class TrainCropDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], needs_sdf: bool):
        self.samples = list(samples)
        self.needs_sdf = needs_sdf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = load_image(sample.image)
        mask = load_mask(sample.mask)
        height, width = mask.shape
        pad_bottom = max(0, CROP_SIZE - height)
        pad_right = max(0, CROP_SIZE - width)
        if pad_bottom or pad_right:
            image = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, pad_bottom), (0, pad_right)), mode="constant")
            height, width = mask.shape

        top = random.randint(0, height - CROP_SIZE)
        left = random.randint(0, width - CROP_SIZE)
        image = image[top:top + CROP_SIZE, left:left + CROP_SIZE]
        mask = mask[top:top + CROP_SIZE, left:left + CROP_SIZE]

        if random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            image, mask = image[::-1, :], mask[::-1, :]
        k = random.randint(0, 3)
        if k:
            image, mask = np.rot90(image, k, (0, 1)), np.rot90(mask, k, (0, 1))

        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        image = np.clip((np.clip(image * brightness, 0, 1) - 0.5) * contrast + 0.5, 0, 1)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        sdf = compute_sdf(mask) if self.needs_sdf else np.zeros_like(mask, dtype=np.float32)

        x = torch.from_numpy(normalize(image)).permute(2, 0, 1).float()
        y = torch.from_numpy(mask).unsqueeze(0).float()
        d = torch.from_numpy(sdf).unsqueeze(0).float()
        return x, y, d

# =============================================================================
# 5. LOSSES — MATCHING CRACK500
# =============================================================================
def soft_dice_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    inter = (probabilities * targets).sum(dims)
    denom = probabilities.sum(dims) + targets.sum(dims)
    dice = (2 * inter + EPS) / (denom + EPS)
    return 1 - dice.mean()


class BCELoss(nn.Module):
    needs_sdf = False
    def forward(self, logits, target, sdf=None, epoch=0):
        return F.binary_cross_entropy_with_logits(logits, target)


class FocalLoss(nn.Module):
    needs_sdf = False
    def forward(self, logits, target, sdf=None, epoch=0):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        return (((1 - pt) ** FOCAL_GAMMA) * bce).mean()


class BCEDiceLoss(nn.Module):
    needs_sdf = False
    def forward(self, logits, target, sdf=None, epoch=0):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return bce + soft_dice_loss(torch.sigmoid(logits), target)


class FocalTverskyLoss(nn.Module):
    needs_sdf = False
    def forward(self, logits, target, sdf=None, epoch=0):
        p = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (p * target).sum(dims)
        fp = (p * (1 - target)).sum(dims)
        fn = ((1 - p) * target).sum(dims)
        ti = (tp + EPS) / (tp + TVERSKY_ALPHA * fp + TVERSKY_BETA * fn + EPS)
        return ((1 - ti) ** FT_EXPONENT).mean()


class DiceBoundaryLoss(nn.Module):
    needs_sdf = True
    def forward(self, logits, target, sdf=None, epoch=0):
        if sdf is None:
            raise ValueError("DiceBoundaryLoss requires SDF maps")
        p = torch.sigmoid(logits)
        alpha = max(BOUNDARY_ALPHA_FLOOR, 1.0 - BOUNDARY_ALPHA_STEP * epoch)
        boundary = (p * sdf).mean()
        return alpha * soft_dice_loss(p, target) + (1 - alpha) * boundary


def soft_erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)

def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)

def soft_open(x):
    return soft_dilate(soft_erode(x))

def soft_skel(x, iters=CLDICE_ITERS):
    skel = F.relu(x - soft_open(x))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class DiceClDiceLoss(nn.Module):
    needs_sdf = False
    def forward(self, logits, target, sdf=None, epoch=0):
        p = torch.sigmoid(logits)
        dice = soft_dice_loss(p, target)
        sp, st = soft_skel(p), soft_skel(target)
        dims = (1, 2, 3)
        tprec = ((sp * target).sum(dims) + EPS) / (sp.sum(dims) + EPS)
        tsens = ((st * p).sum(dims) + EPS) / (st.sum(dims) + EPS)
        cldice = (2 * tprec * tsens) / (tprec + tsens + EPS)
        return (1 - CLDICE_ALPHA) * dice + CLDICE_ALPHA * (1 - cldice.mean())


LOSS_REGISTRY = {
    "bce": BCELoss,
    "focal": FocalLoss,
    "bce_dice": BCEDiceLoss,
    "focal_tversky": FocalTverskyLoss,
    "dice_boundary": DiceBoundaryLoss,
    "dice_cldice": DiceClDiceLoss,
}

# =============================================================================
# 6. MODEL, FULL-IMAGE INFERENCE, AND MODEL SELECTION
# =============================================================================
def build_model() -> nn.Module:
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)


@torch.no_grad()
def infer_full(model: nn.Module, image_path: Path) -> np.ndarray:
    image = load_image(image_path)
    padded, (height, width) = pad_to_multiple(image)
    x = torch.from_numpy(normalize(padded)).permute(2, 0, 1)[None].float().to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
        probability = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    return probability[:height, :width].astype(np.float32, copy=False)


@torch.no_grad()
def validate_f1_grid(
    model: nn.Module,
    samples: Sequence[Sample],
) -> Tuple[float, float, float]:
    """Return selected F1, threshold, and pooled predicted-foreground fraction."""
    model.eval()
    edges = np.linspace(0.0, 1.0, 101)
    pos_hist = np.zeros(100, dtype=np.float64)
    all_hist = np.zeros(100, dtype=np.float64)
    n_pos = 0.0
    for sample in samples:
        probability = infer_full(model, sample.image)
        truth = load_mask(sample.mask) > 0.5
        if probability.shape != truth.shape:
            raise ValueError(f"Shape mismatch for {sample.sample_id}: {probability.shape} vs {truth.shape}")
        pos_hist += np.histogram(probability[truth], bins=edges)[0]
        all_hist += np.histogram(probability, bins=edges)[0]
        n_pos += float(truth.sum())
    pos_tail = np.cumsum(pos_hist[::-1])[::-1]
    all_tail = np.cumsum(all_hist[::-1])[::-1]
    tp = pos_tail[1:100]
    predpos = all_tail[1:100]
    f1 = (2.0 * tp) / (predpos + n_pos + EPS)
    f1r = np.round(f1, 4)
    candidates = np.flatnonzero(f1r == f1r.max())
    order = np.lexsort((THRESHOLDS[candidates], np.abs(THRESHOLDS[candidates] - 0.5)))
    best = candidates[order[0]]

    total_pixels = float(all_hist.sum())
    if total_pixels <= 0:
        raise RuntimeError("Validation histogram contains zero pixels")
    predicted_foreground_fraction = float(predpos[best] / total_pixels)
    return (
        float(f1[best]),
        float(THRESHOLDS[best]),
        predicted_foreground_fraction,
    )


@torch.no_grad()
def save_probability_maps(model: nn.Module, samples: Sequence[Sample], output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for sample in samples:
        probability = infer_full(model, sample.image)
        np.save(output_dir / f"{sample.sample_id}.npy", probability.astype(np.float32), allow_pickle=False)

# =============================================================================
# 7. MANIFESTS, RESUME, AND SUMMARY
# =============================================================================
def frozen_config() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "arch": "unet_resnet34_imagenet",
        "crop": CROP_SIZE,
        "batch": BATCH_SIZE,
        "workers": NUM_WORKERS,
        "drop_last": True,
        "epochs_max": EPOCHS,
        "early_stopping": False,
        "training_budget_policy": "all runs train exactly 100 epochs",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "select_metric": SELECT_METRIC,
        "focal_gamma": FOCAL_GAMMA,
        "tversky_alpha": TVERSKY_ALPHA,
        "tversky_beta": TVERSKY_BETA,
        "ft_exponent": FT_EXPONENT,
        "cldice_alpha": CLDICE_ALPHA,
        "cldice_iters": CLDICE_ITERS,
        "cldice_alpha_note": "pre-specified protocol value, not a paper default",
        "boundary_alpha_floor": BOUNDARY_ALPHA_FLOOR,
        "boundary_alpha_step": BOUNDARY_ALPHA_STEP,
        "boundary_schedule": (
            "dice_weight_alpha=max(0.01, 1-0.01*epoch); "
            "boundary_weight=1-alpha"
        ),
        "boundary_schedule_stopping_interaction": (
            "Full-budget training exposes Dice+Boundary to later boundary-dominant "
            "weights than patience-20 training; its v1-v2 delta is not a pure "
            "stopping-time contrast."
        ),
        "boundary_epoch_indexing": "zero-based, identical to Crack500",
        "boundary_sdf": "unnormalized (Kervadec original), frozen",
        "threshold_grid": "0.01:0.99:0.01",
        "threshold_comparator": THRESHOLD_COMPARATOR,
        "probmap_dtype": "float32",
        "training_resize": "none",
        "evaluation_resize": "none",
        "evaluation_padding": "reflect_to_multiple_of_32",
        "persistent_workers": False,
        "post_freeze_extension": True,
        "sensitivity_experiment": "full_100_epochs_no_early_stopping",
        "original_patience_runs_preserved": True,
    }


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def map_names(directory: Path) -> set[str]:
    return {p.stem for p in directory.glob("*.npy")} if directory.exists() else set()


def run_is_complete(run_dir: Path, val_samples: Sequence[Sample], test_samples: Sequence[Sample]) -> bool:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("status") != "complete":
        return False
    if manifest.get("early_stopping") is not False:
        return False
    if manifest.get("epochs_trained") != EPOCHS:
        return False
    if manifest.get("config", {}).get("protocol_version") != PROTOCOL_VERSION:
        return False
    if manifest.get("provenance", {}).get("script_sha256") != script_sha256():
        return False
    if (
        not (run_dir / "best.pt").exists()
        or not (run_dir / "last.pt").exists()
        or not (run_dir / "train_log.csv").exists()
    ):
        return False
    expected_val = {s.sample_id for s in val_samples}
    expected_test = {s.sample_id for s in test_samples}
    return map_names(run_dir / "probmaps_val") == expected_val and map_names(run_dir / "probmaps_test") == expected_test


def rebuild_summary(output_root: Path) -> None:
    rows = []
    for loss_name in LOSS_ORDER:
        for seed in SEEDS:
            path = output_root / f"{loss_name}_seed{seed}" / "manifest.json"
            if not path.exists():
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows.append([
                manifest.get("run_id"),
                manifest.get("loss"),
                manifest.get("seed"),
                manifest.get("status"),
                manifest.get("best_epoch"),
                manifest.get("best_val_f1"),
                manifest.get("best_val_threshold"),
                manifest.get("best_val_pred_foreground_fraction"),
                manifest.get("final_epoch"),
                manifest.get("final_val_f1"),
                manifest.get("final_val_threshold"),
                manifest.get("final_val_pred_foreground_fraction"),
                manifest.get("epochs_trained"),
                manifest.get("early_stopping"),
            ])
    with (output_root / "runs_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "run_id",
            "loss",
            "seed",
            "status",
            "best_epoch",
            "best_val_f1",
            "best_val_threshold",
            "best_val_pred_foreground_fraction",
            "final_epoch",
            "final_val_f1",
            "final_val_threshold",
            "final_val_pred_foreground_fraction",
            "epochs_trained",
            "early_stopping",
        ])
        writer.writerows(rows)

# =============================================================================
# 8. ONE RUN
# =============================================================================
def train_one_run(
    dataset_name: str,
    output_root: Path,
    loss_name: str,
    seed: int,
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    force: bool,
) -> None:
    run_id = f"{loss_name}_seed{seed}"
    run_dir = output_root / run_id
    manifest_path = run_dir / "manifest.json"

    if run_is_complete(run_dir, val_samples, test_samples) and not force:
        print(f"[skip] {dataset_name} / {run_id} verified complete")
        return

    if run_dir.exists():
        print(f"[redo] {dataset_name} / {run_id} will be trained from scratch")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    manifest = {
        "run_id": run_id,
        "dataset": dataset_name,
        "loss": loss_name,
        "seed": seed,
        "status": "started",
        "early_stopping": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "config": frozen_config(),
        "provenance": provenance_info(),
        "env": env_info(),
    }
    write_json(manifest_path, manifest)

    loss_function = LOSS_REGISTRY[loss_name]()
    dataset = TrainCropDataset(train_samples, needs_sdf=loss_function.needs_sdf)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=False,
        drop_last=True,
    )

    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    best_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    best_val_pred_foreground_fraction = float("nan")
    final_val_f1 = float("nan")
    final_val_threshold = float("nan")
    final_val_pred_foreground_fraction = float("nan")

    log_path = run_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            "epoch",
            "train_loss",
            "val_f1",
            "val_thr",
            "val_pred_foreground_fraction",
            "is_best",
            "lr",
            "seconds",
        ])

    print(
        f"\n[run ] {dataset_name} / {run_id} starting on {DEVICE}; "
        f"full {EPOCHS}-epoch budget, no early stopping"
    )

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        running = 0.0
        batches = 0

        for images, masks, sdfs in loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            sdfs = sdfs.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                logits = model(images)
                loss = loss_function(logits, masks, sdf=sdfs, epoch=epoch)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss in {dataset_name}/{run_id}, "
                    f"epoch {epoch}: {loss.item()}"
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            batches += 1

        if batches == 0:
            raise RuntimeError(
                f"Training loader produced zero batches for {dataset_name}/{run_id}"
            )

        scheduler.step()
        train_loss = running / batches
        (
            val_f1,
            val_threshold,
            val_pred_foreground_fraction,
        ) = validate_f1_grid(model, val_samples)
        final_val_f1 = val_f1
        final_val_threshold = val_threshold
        final_val_pred_foreground_fraction = val_pred_foreground_fraction

        is_best = False
        if round(val_f1, 4) > round(best_f1, 4):
            best_f1 = val_f1
            best_epoch = epoch
            best_threshold = val_threshold
            best_val_pred_foreground_fraction = val_pred_foreground_fraction
            torch.save(model.state_dict(), run_dir / "best.pt")
            is_best = True

        elapsed = time.time() - start
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                epoch,
                f"{train_loss:.5f}",
                f"{val_f1:.5f}",
                f"{val_threshold:.2f}",
                f"{val_pred_foreground_fraction:.8f}",
                int(is_best),
                f"{scheduler.get_last_lr()[0]:.2e}",
                f"{elapsed:.1f}",
            ])

        print(
            f"  {dataset_name}/{run_id} epoch {epoch:03d} "
            f"loss {train_loss:.4f} val_f1 {val_f1:.4f} "
            f"thr {val_threshold:.2f} "
            f"pred_fg {val_pred_foreground_fraction:.6f}"
            f"{'  [best]' if is_best else ''}"
        )

    torch.save(model.state_dict(), run_dir / "last.pt")

    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"No best checkpoint created for {dataset_name}/{run_id}")

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    save_probability_maps(model, val_samples, run_dir / "probmaps_val")
    save_probability_maps(model, test_samples, run_dir / "probmaps_test")

    manifest.update({
        "status": "complete",
        "early_stopping": False,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_f1, 5),
        "best_val_threshold": round(best_threshold, 2),
        "best_val_pred_foreground_fraction": round(
            best_val_pred_foreground_fraction, 8
        ),
        "final_epoch": EPOCHS - 1,
        "final_val_f1": round(final_val_f1, 5),
        "final_val_threshold": round(final_val_threshold, 2),
        "final_val_pred_foreground_fraction": round(
            final_val_pred_foreground_fraction, 8
        ),
        "epochs_trained": EPOCHS,
        "best_checkpoint": "best.pt",
        "final_checkpoint": "last.pt",
        "probability_maps_checkpoint": "best.pt",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    write_json(manifest_path, manifest)

    if not run_is_complete(run_dir, val_samples, test_samples):
        raise RuntimeError(
            f"Post-save artifact verification failed for {dataset_name}/{run_id}"
        )

    print(
        f"[done] {dataset_name} / {run_id} trained all {EPOCHS} epochs; "
        f"best_val_f1 {best_f1:.4f} at epoch {best_epoch} "
        f"(threshold {best_threshold:.2f}); "
        f"final_val_f1 {final_val_f1:.4f}"
    )

# =============================================================================
# 9. DATASET PREPARATION AND MAIN
# =============================================================================
def check_configuration(selected: Sequence[str]) -> None:
    for dataset_name in selected:
        config = DATASETS[dataset_name]
        source_type = str(config.get("source_type", ""))

        common_keys = ("root", "output_root")
        if source_type == "split_csv":
            required_keys = common_keys + ("train_csv", "val_csv", "test_csv")
        elif source_type == "paired_directories":
            required_keys = common_keys + (
                "train_image_dir",
                "train_mask_dir",
                "val_image_dir",
                "val_mask_dir",
                "test_image_dir",
                "test_mask_dir",
                "image_ext",
                "mask_ext",
            )
        else:
            raise SystemExit(
                f"Unsupported source_type for {dataset_name}: {source_type!r}"
            )

        for key in required_keys:
            if key not in config or "FILL_ME" in str(config[key]):
                raise SystemExit(
                    f"Fill in DATASETS['{dataset_name}']['{key}'] before running"
                )

        for key in ("expected_train", "expected_val", "expected_test"):
            if int(config[key]) < 0:
                raise SystemExit(
                    f"Freeze and enter DATASETS['{dataset_name}']['{key}'] before running"
                )


def write_json_atomic(path: Path, payload: dict) -> None:
    write_json(path, payload)


def prepare_dataset(
    dataset_name: str,
) -> Tuple[Path, List[Sample], List[Sample], List[Sample]]:
    config = DATASETS[dataset_name]
    source_type = str(config["source_type"])
    root = Path(str(config["root"])).resolve()
    output_root = Path(str(config["output_root"])).resolve()

    if not root.exists():
        raise FileNotFoundError(f"{dataset_name} root does not exist: {root}")

    source_manifest: dict
    if source_type == "split_csv":
        train_csv = Path(str(config["train_csv"])).resolve()
        val_csv = Path(str(config["val_csv"])).resolve()
        test_csv = Path(str(config["test_csv"])).resolve()

        train = read_split_csv(train_csv, root)
        val = read_split_csv(val_csv, root)
        test = read_split_csv(test_csv, root)
        source_manifest = {
            "source_type": source_type,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
        }
    else:
        train_image_dir = Path(str(config["train_image_dir"])).resolve()
        train_mask_dir = Path(str(config["train_mask_dir"])).resolve()
        val_image_dir = Path(str(config["val_image_dir"])).resolve()
        val_mask_dir = Path(str(config["val_mask_dir"])).resolve()
        test_image_dir = Path(str(config["test_image_dir"])).resolve()
        test_mask_dir = Path(str(config["test_mask_dir"])).resolve()
        image_ext = str(config["image_ext"])
        mask_ext = str(config["mask_ext"])

        train = read_directory_split(
            train_image_dir, train_mask_dir, image_ext, mask_ext
        )
        val = read_directory_split(val_image_dir, val_mask_dir, image_ext, mask_ext)
        test = read_directory_split(
            test_image_dir, test_mask_dir, image_ext, mask_ext
        )
        source_manifest = {
            "source_type": source_type,
            "train_image_dir": str(train_image_dir),
            "train_mask_dir": str(train_mask_dir),
            "val_image_dir": str(val_image_dir),
            "val_mask_dir": str(val_mask_dir),
            "test_image_dir": str(test_image_dir),
            "test_mask_dir": str(test_mask_dir),
            "image_ext": image_ext,
            "mask_ext": mask_ext,
        }

    validate_expected_counts(dataset_name, config, train, val, test)
    leakage = validate_no_split_leakage(dataset_name, train, val, test)
    allowed_id_overlap = leakage["sample_id_overlap_allowed"]
    if any(allowed_id_overlap.values()):
        counts = {key: len(values) for key, values in allowed_id_overlap.items()}
        print(
            f"[info] {dataset_name}: reused filename stems across split directories "
            f"were found and allowed after path/hash validation: {counts}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    write_split_lists(
        output_root,
        root,
        {"train": train, "val": val, "test": test},
    )

    write_json_atomic(
        output_root / "dataset_manifest.json",
        {
            "dataset": dataset_name,
            "root": str(root),
            **source_manifest,
            "counts": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
            "split_leakage_check": leakage,
            "split_hashes_written": HASH_SPLIT_FILES,
            "probability_map_key": "image filename stem",
            "protocol_version": PROTOCOL_VERSION,
            "provenance": provenance_info(),
            "experiment": "full_100_epochs_no_early_stopping",
            "early_stopping": False,
            "epochs_per_run": EPOCHS,
            "original_patience_runs_preserved": True,
            "post_freeze_replication_extension": True,
            "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        },
    )
    return output_root, train, val, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Paper 3 on Crack500, DeepCrack, CFD, and CrackTree260: "
            "6 losses x 3 seeds, exactly 100 epochs per run, no early stopping."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        choices=LOSS_ORDER,
        default=LOSS_ORDER,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=SEEDS,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and retrain selected completed runs from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_configuration(args.datasets)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP enabled: {AMP_ENABLED}")
    print(f"Datasets: {args.datasets}")
    print(f"Losses: {args.losses}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs per run: {EPOCHS}")
    print("Early stopping: DISABLED")
    print(
        "Output policy: separate Full 100 Epochs folders; original runs are untouched"
    )

    requested_runs = len(args.datasets) * len(args.losses) * len(args.seeds)
    print(f"Requested run count: {requested_runs}")

    for dataset_name in args.datasets:
        output_root, train, val, test = prepare_dataset(dataset_name)
        print(
            f"\n{dataset_name}: train={len(train)} | "
            f"val={len(val)} | test={len(test)}"
        )

        for loss_name in args.losses:
            for seed in args.seeds:
                train_one_run(
                    dataset_name,
                    output_root,
                    loss_name,
                    seed,
                    train,
                    val,
                    test,
                    args.force,
                )
                rebuild_summary(output_root)
        rebuild_summary(output_root)

    print("\nAll requested full-budget runs are complete.")


if __name__ == "__main__":
    main()
