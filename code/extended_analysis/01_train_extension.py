#!/usr/bin/env python
"""Additional training experiments for the loss-metric alignment study.

This script performs ONLY the architecture and seed extension.
It never writes into the original 72-run output folders.

New training matrix
-------------------
A) Architecture replication on Crack500:
   DeepLabV3+ / ImageNet ResNet-50 x 6 losses x seeds {0,1,2} = 18 runs.
B) Additional seed replication on Crack500:
   Original U-Net / ImageNet ResNet-34 x 6 losses x seeds {3,4} = 12 runs.

Total new runs = 30.

All training/scoring-relevant settings are inherited from the submitted v2 study:
448x448 random crops; no resizing; AdamW 1e-4; weight decay 1e-4;
cosine schedule; batch size 8; 100 epochs; no early stopping; ImageNet
normalization; identical augmentation; identical six losses/hyperparameters;
validation dataset-global F1 model selection over thresholds 0.01..0.99;
full-image native-resolution inference; float32 probability-map export.

The extension is explicitly marked as post-review in every manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
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
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise SystemExit(
        "segmentation_models_pytorch is required. Install with:\n"
        "pip install torch torchvision segmentation-models-pytorch scipy numpy pillow tqdm"
    ) from exc

# =============================================================================
# 1. PATHS AND FROZEN SETTINGS
# =============================================================================
PAPER_ROOT = Path(os.environ.get("PAPER3_ROOT", Path.cwd())).expanduser().resolve()
DATASET_BASE = Path(os.environ.get("PAPER3_DATASETS", PAPER_ROOT / "Datasets")).expanduser().resolve()
DATASET_ROOT = DATASET_BASE / "Crack500"
OUTPUT_BASE = Path(os.environ.get("PAPER3_OUTPUT", PAPER_ROOT / "Output")).expanduser().resolve()
EXTENSION_OUTPUT_ROOT = OUTPUT_BASE / "Extension"

CRACK500 = {
    "root": DATASET_ROOT,
    "train_image_dir": DATASET_ROOT / "traincrop" / "traincrop",
    "train_mask_dir": DATASET_ROOT / "traincrop" / "traincrop",
    "val_image_dir": DATASET_ROOT / "valcrop" / "valcrop",
    "val_mask_dir": DATASET_ROOT / "valcrop" / "valcrop",
    "test_image_dir": DATASET_ROOT / "testcrop" / "testcrop",
    "test_mask_dir": DATASET_ROOT / "testcrop" / "testcrop",
    "image_ext": ".jpg",
    "mask_ext": ".png",
    "expected_train": 1896,
    "expected_val": 348,
    "expected_test": 1124,
}

EXPERIMENTS: Dict[str, dict] = {
    "deeplab": {
        "architecture": "deeplabv3plus_resnet50_imagenet",
        "output_root": EXTENSION_OUTPUT_ROOT / "DeepLabV3Plus_ResNet50_Crack500",
        "seeds": [0, 1, 2],
    },
    "unet_extra": {
        "architecture": "unet_resnet34_imagenet",
        "output_root": EXTENSION_OUTPUT_ROOT / "UNet_ResNet34_Crack500_Seeds3_4",
        "seeds": [3, 4],
    },
}

LOSS_ORDER = ["bce", "focal", "bce_dice", "focal_tversky", "dice_boundary", "dice_cldice"]
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

# Exact submitted-study loss settings.
FOCAL_GAMMA = 2.0
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
FT_EXPONENT = 0.75
CLDICE_ALPHA = 0.3
CLDICE_ITERS = 5
BOUNDARY_ALPHA_FLOOR = 0.01
BOUNDARY_ALPHA_STEP = 0.01

THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100.0
THRESHOLD_COMPARATOR = "p >= t"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EPS = 1e-6

PROTOCOL_VERSION = "v2_no_early_stopping_post_review_extension_v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = USE_AMP and DEVICE.type == "cuda"


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    sample_id: str


# =============================================================================
# 2. REPRODUCIBILITY / I/O
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


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def script_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


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
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "amp_enabled": AMP_ENABLED,
        "deterministic": DETERMINISTIC,
        "reproducibility_note": (
            "seed-controlled, bitwise deterministic" if DETERMINISTIC
            else "seed-controlled, not bitwise deterministic"
        ),
    }


def read_directory_split(image_dir: Path, mask_dir: Path, image_ext: str, mask_ext: str) -> List[Sample]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
    images = sorted(image_dir.glob(f"*{image_ext}"))
    if not images:
        raise FileNotFoundError(f"No {image_ext} files in {image_dir}")
    samples: List[Sample] = []
    missing: List[str] = []
    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}{mask_ext}"
        if not mask_path.exists():
            missing.append(mask_path.name)
            continue
        samples.append(Sample(image_path.resolve(), mask_path.resolve(), image_path.stem))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} masks; examples: {missing[:5]}")
    ids = [s.sample_id for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate stems in {image_dir}")
    return samples


def prepare_crack500(output_root: Path) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    train = read_directory_split(CRACK500["train_image_dir"], CRACK500["train_mask_dir"], ".jpg", ".png")
    val = read_directory_split(CRACK500["val_image_dir"], CRACK500["val_mask_dir"], ".jpg", ".png")
    test = read_directory_split(CRACK500["test_image_dir"], CRACK500["test_mask_dir"], ".jpg", ".png")
    actual = {"train": len(train), "val": len(val), "test": len(test)}
    expected = {"train": 1896, "val": 348, "test": 1124}
    if actual != expected:
        raise ValueError(f"Crack500 split mismatch: expected {expected}; found {actual}")

    # Published directories are distinct. Reject accidental exact path reuse.
    for a_name, a, b_name, b in [
        ("train", train, "val", val), ("train", train, "test", test), ("val", val, "test", test)
    ]:
        overlap = {s.image for s in a} & {s.image for s in b}
        if overlap:
            raise ValueError(f"Exact image-path leakage between {a_name} and {b_name}: {list(overlap)[:3]}")

    output_root.mkdir(parents=True, exist_ok=True)
    for split_name, samples in {"train": train, "val": val, "test": test}.items():
        lines = []
        for s in tqdm(samples, desc=f"hash {split_name}", unit="file", leave=False):
            image_rel = s.image.relative_to(DATASET_ROOT)
            mask_rel = s.mask.relative_to(DATASET_ROOT)
            if HASH_SPLIT_FILES:
                lines.append("\t".join([str(image_rel), str(mask_rel), sha256_file(s.image), sha256_file(s.mask)]))
            else:
                lines.append(f"{image_rel}\t{mask_rel}")
        (output_root / f"{split_name}_files.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return train, val, test


# =============================================================================
# 3. IMAGE PIPELINE — IDENTICAL TO SUBMITTED V2 STUDY
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
    h, w = array.shape[:2]
    pb = (stride - h % stride) % stride
    pr = (stride - w % stride) % stride
    if pb == 0 and pr == 0:
        return array, (h, w)
    pad_spec = ((0, pb), (0, pr)) + (((0, 0),) if array.ndim == 3 else ())
    return np.pad(array, pad_spec, mode="reflect"), (h, w)


class TrainCropDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], needs_sdf: bool):
        self.samples = list(samples)
        self.needs_sdf = needs_sdf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = load_image(sample.image)
        mask = load_mask(sample.mask)
        h, w = mask.shape
        pb = max(0, CROP_SIZE - h)
        pr = max(0, CROP_SIZE - w)
        if pb or pr:
            image = np.pad(image, ((0, pb), (0, pr), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, pb), (0, pr)), mode="constant")
            h, w = mask.shape

        top = random.randint(0, h - CROP_SIZE)
        left = random.randint(0, w - CROP_SIZE)
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
# 4. LOSSES — EXACT SUBMITTED V2 DEFINITIONS
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
# 5. MODELS / VALIDATION / PROBABILITY MAPS
# =============================================================================
def build_model(architecture: str) -> nn.Module:
    if architecture == "unet_resnet34_imagenet":
        return smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=3, classes=1, activation=None,
        )
    if architecture == "deeplabv3plus_resnet50_imagenet":
        return smp.DeepLabV3Plus(
            encoder_name="resnet50", encoder_weights="imagenet",
            in_channels=3, classes=1, activation=None,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


@torch.no_grad()
def infer_full(model: nn.Module, image_path: Path) -> np.ndarray:
    image = load_image(image_path)
    padded, (h, w) = pad_to_multiple(image)
    x = torch.from_numpy(normalize(padded)).permute(2, 0, 1)[None].float().to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
        prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    return prob[:h, :w].astype(np.float32, copy=False)


@torch.no_grad()
def validate_f1_grid(model: nn.Module, samples: Sequence[Sample]) -> Tuple[float, float, float]:
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
    return float(f1[best]), float(THRESHOLDS[best]), float(predpos[best] / total_pixels)


@torch.no_grad()
def save_probability_maps(model: nn.Module, samples: Sequence[Sample], output_dir: Path, desc: str) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for sample in tqdm(samples, desc=desc, unit="image", leave=False):
        probability = infer_full(model, sample.image)
        np.save(output_dir / f"{sample.sample_id}.npy", probability.astype(np.float32), allow_pickle=False)


def map_names(directory: Path) -> set[str]:
    return {p.stem for p in directory.glob("*.npy")} if directory.exists() else set()


def frozen_config(architecture: str) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "post_review_extension": True,
        "extension_reason": "architecture and seed robustness extension",
        "architecture": architecture,
        "crop": CROP_SIZE,
        "batch": BATCH_SIZE,
        "workers": NUM_WORKERS,
        "drop_last": True,
        "epochs": EPOCHS,
        "early_stopping": False,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "lr_schedule": "CosineAnnealingLR(T_max=100)",
        "focal_gamma": FOCAL_GAMMA,
        "tversky_alpha": TVERSKY_ALPHA,
        "tversky_beta": TVERSKY_BETA,
        "ft_exponent": FT_EXPONENT,
        "cldice_weight": CLDICE_ALPHA,
        "cldice_soft_skeleton_iterations": CLDICE_ITERS,
        "boundary_dice_weight_schedule": "max(0.01, 1-0.01*epoch), zero-based epoch",
        "boundary_sdf": "unnormalized signed distance map; zero for empty/full masks",
        "threshold_grid": "0.01:0.99:0.01",
        "threshold_comparator": THRESHOLD_COMPARATOR,
        "training_resize": "none",
        "evaluation_resize": "none",
        "evaluation_padding": "reflect_to_multiple_of_32_then_crop_back",
        "probability_map_dtype": "float32",
        "augmentation": {
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.5,
            "rotation": "uniform k*90 degrees, k in {0,1,2,3}",
            "brightness_factor": "U(0.8,1.2)",
            "contrast_factor": "U(0.8,1.2)",
        },
        "normalization": "ImageNet mean/std",
    }


def run_is_complete(run_dir: Path, architecture: str, val_samples: Sequence[Sample], test_samples: Sequence[Sample]) -> bool:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if m.get("status") != "complete" or m.get("epochs_trained") != EPOCHS or m.get("early_stopping") is not False:
        return False
    if m.get("config", {}).get("protocol_version") != PROTOCOL_VERSION:
        return False
    if m.get("config", {}).get("architecture") != architecture:
        return False
    required = [run_dir / "best.pt", run_dir / "last.pt", run_dir / "train_log.csv"]
    if not all(p.exists() for p in required):
        return False
    return (
        map_names(run_dir / "probmaps_val") == {s.sample_id for s in val_samples}
        and map_names(run_dir / "probmaps_test") == {s.sample_id for s in test_samples}
    )


def train_one_run(
    *, architecture: str, output_root: Path, loss_name: str, seed: int,
    train_samples: Sequence[Sample], val_samples: Sequence[Sample],
    test_samples: Sequence[Sample], force: bool,
) -> None:
    run_id = f"{loss_name}_seed{seed}"
    run_dir = output_root / run_id
    manifest_path = run_dir / "manifest.json"

    if run_is_complete(run_dir, architecture, val_samples, test_samples) and not force:
        print(f"[skip] {architecture} / {run_id} verified complete")
        return
    if run_dir.exists():
        if not force:
            print(f"[redo] incomplete {architecture} / {run_id} will be restarted from scratch")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    manifest = {
        "run_id": run_id,
        "dataset": "Crack500",
        "loss": loss_name,
        "seed": seed,
        "status": "started",
        "early_stopping": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
        "config": frozen_config(architecture),
        "provenance": {"script_sha256": script_sha256(), "script": str(Path(__file__).resolve())},
        "environment": env_info(),
    }
    write_json(manifest_path, manifest)

    loss_fn = LOSS_REGISTRY[loss_name]()
    train_ds = TrainCropDataset(train_samples, needs_sdf=loss_fn.needs_sdf)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
        worker_init_fn=seed_worker, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"), persistent_workers=False, drop_last=True,
    )

    model = build_model(architecture).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    best_f1 = -1.0
    best_epoch = -1
    best_thr = 0.5
    best_pred_fg = float("nan")
    final_f1 = final_thr = final_pred_fg = float("nan")

    log_path = run_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_f1", "val_thr", "val_pred_foreground_fraction",
            "is_best", "lr", "seconds"
        ])

    epoch_bar = tqdm(range(EPOCHS), desc=f"{architecture[:12]} {run_id}", unit="epoch")
    for epoch in epoch_bar:
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
                loss = loss_fn(logits, masks, sdf=sdfs, epoch=epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {run_id}, epoch {epoch}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            batches += 1
        if batches == 0:
            raise RuntimeError("Training loader produced zero batches")
        scheduler.step()
        train_loss = running / batches
        val_f1, val_thr, pred_fg = validate_f1_grid(model, val_samples)
        final_f1, final_thr, final_pred_fg = val_f1, val_thr, pred_fg

        is_best = False
        if round(val_f1, 4) > round(best_f1, 4):
            best_f1, best_epoch, best_thr, best_pred_fg = val_f1, epoch, val_thr, pred_fg
            torch.save(model.state_dict(), run_dir / "best.pt")
            is_best = True

        elapsed = time.time() - start
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{val_f1:.6f}", f"{val_thr:.2f}",
                f"{pred_fg:.8f}", int(is_best), f"{scheduler.get_last_lr()[0]:.8e}", f"{elapsed:.2f}"
            ])
        epoch_bar.set_postfix(loss=f"{train_loss:.4f}", valF1=f"{val_f1:.4f}", thr=f"{val_thr:.2f}")

    torch.save(model.state_dict(), run_dir / "last.pt")
    if not (run_dir / "best.pt").exists():
        raise RuntimeError(f"No best checkpoint for {run_id}")
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=DEVICE), strict=True)
    save_probability_maps(model, val_samples, run_dir / "probmaps_val", f"{run_id} val maps")
    save_probability_maps(model, test_samples, run_dir / "probmaps_test", f"{run_id} test maps")

    manifest.update({
        "status": "complete",
        "best_epoch": best_epoch,
        "best_val_f1": round(best_f1, 6),
        "best_val_threshold": round(best_thr, 2),
        "best_val_pred_foreground_fraction": round(best_pred_fg, 8),
        "final_epoch": EPOCHS - 1,
        "final_val_f1": round(final_f1, 6),
        "final_val_threshold": round(final_thr, 2),
        "final_val_pred_foreground_fraction": round(final_pred_fg, 8),
        "epochs_trained": EPOCHS,
        "best_checkpoint": "best.pt",
        "final_checkpoint": "last.pt",
        "probability_maps_checkpoint": "best.pt",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    write_json(manifest_path, manifest)
    if not run_is_complete(run_dir, architecture, val_samples, test_samples):
        raise RuntimeError(f"Post-save verification failed for {run_id}")
    print(f"[done] {architecture} / {run_id}: best val F1={best_f1:.4f}, threshold={best_thr:.2f}, epoch={best_epoch}")


def rebuild_summary(output_root: Path) -> None:
    rows = []
    for run_dir in sorted(output_root.glob("*_seed*")):
        path = run_dir / "manifest.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "run_id": m.get("run_id"), "architecture": m.get("config", {}).get("architecture"),
            "loss": m.get("loss"), "seed": m.get("seed"), "status": m.get("status"),
            "best_epoch": m.get("best_epoch"), "best_val_f1": m.get("best_val_f1"),
            "best_val_threshold": m.get("best_val_threshold"),
            "best_val_pred_foreground_fraction": m.get("best_val_pred_foreground_fraction"),
            "epochs_trained": m.get("epochs_trained"),
        })
    if rows:
        with (output_root / "runs_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def preflight() -> Tuple[List[Sample], List[Sample], List[Sample]]:
    print("=" * 78)
    print("LOSS-METRIC ALIGNMENT - EXTENSION TRAINING PREFLIGHT")
    print("=" * 78)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Torch: {torch.__version__}; CUDA build: {torch.version.cuda}; AMP: {AMP_ENABLED}")
    print(f"Crack500 root: {DATASET_ROOT}")
    print(f"Revision output: {EXTENSION_OUTPUT_ROOT}")
    for key, p in CRACK500.items():
        if isinstance(p, Path) and key != "root" and "output" not in key:
            if not p.exists():
                raise FileNotFoundError(f"Missing required path {key}: {p}")
    # Hash/split lists are written separately under each architecture root so each extension is auditable.
    first_root = EXPERIMENTS["deeplab"]["output_root"]
    train, val, test = prepare_crack500(first_root)
    second_root = EXPERIMENTS["unet_extra"]["output_root"]
    second_root.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        shutil.copy2(first_root / f"{split_name}_files.txt", second_root / f"{split_name}_files.txt")
    print(f"Counts verified: train={len(train)}, val={len(val)}, test={len(test)}")
    # Build both models once to catch package/version problems before a long run.
    for exp_name, spec in EXPERIMENTS.items():
        model = build_model(spec["architecture"])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model OK: {exp_name} -> {spec['architecture']} ({n_params:,} parameters)")
        del model
    print("Preflight PASSED.")
    return train, val, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["all", "deeplab", "unet_extra"], default="all")
    parser.add_argument("--losses", nargs="+", choices=LOSS_ORDER, default=LOSS_ORDER)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, val, test = preflight()
    if args.preflight_only:
        return

    selected = ["deeplab", "unet_extra"] if args.experiment == "all" else [args.experiment]
    requested = sum(len(EXPERIMENTS[e]["seeds"]) * len(args.losses) for e in selected)
    print(f"\nNew runs requested: {requested}")
    print("Original 72-run folders are not writable by this script.\n")

    run_counter = 0
    for exp_name in selected:
        spec = EXPERIMENTS[exp_name]
        output_root: Path = spec["output_root"]
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(output_root / "extension_manifest.json", {
            "experiment": exp_name,
            "architecture": spec["architecture"],
            "dataset": "Crack500",
            "seeds": spec["seeds"],
            "losses": args.losses,
            "post_review_extension": True,
            "reason": "architecture and seed robustness extension",
            "original_submitted_run_matrix_unchanged": True,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        for loss_name in args.losses:
            for seed in spec["seeds"]:
                run_counter += 1
                print(f"\n===== NEW RUN {run_counter}/{requested}: {exp_name} / {loss_name} / seed {seed} =====")
                train_one_run(
                    architecture=spec["architecture"], output_root=output_root,
                    loss_name=loss_name, seed=seed, train_samples=train,
                    val_samples=val, test_samples=test, force=args.force,
                )
                rebuild_summary(output_root)

    print("\nAll requested extension training runs are complete.")


if __name__ == "__main__":
    main()
