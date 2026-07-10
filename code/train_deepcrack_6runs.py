
"""
DeepCrack controlled demonstration: all 6 training runs back-to-back.
Paper 3 protocol freeze v1.0: 6 losses x 1 seed, one fixed backbone.

This is the DeepCrack counterpart to train_crack500_18runs.py.

DeepCrack protocol commitments:
  * Published 300/237 train/test split is preserved.
  * Validation is a fixed 20% subset (60 images) drawn once from the 300
    training images using VAL_SPLIT_SEED.
  * Remaining 240 images are used for training.
  * U-Net + ResNet-34 (ImageNet init), sigmoid/logit output, one backbone.
  * No resizing anywhere. Training uses 448x448 random crops with constant
    zero padding if an image is smaller. Validation/test inference runs on
    full native-resolution images, reflect-padded to a multiple of 32 and
    cropped back before anything is stored.
  * Best checkpoint is selected by validation-selected dataset-global F1:
    for each epoch, F1 is maximized over the frozen 0.01-0.99 threshold grid.
  * Binarization comparator is frozen as p >= t.
  * Saved artifacts per run: best checkpoint, train log, manifest.json,
    float32 probability maps for every validation and test image.
  * Re-running the script skips runs verified complete.

Dependencies:
    pip install torch segmentation_models_pytorch scipy numpy pillow
"""

import csv
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt as edt
from torch.utils.data import DataLoader, Dataset

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise SystemExit(
        "segmentation_models_pytorch is required: "
        "pip install segmentation_models_pytorch"
    ) from e


# =========================================================================
# 1. PATHS
# =========================================================================
DEEPCRACK_ROOT = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Dataset\DeepCrack Dataset"
)

TRAINVAL_IMG_DIR  = DEEPCRACK_ROOT / "train_img"  # 300 published training images
TRAINVAL_MASK_DIR = DEEPCRACK_ROOT / "train_lab"  # 300 published training masks

TEST_IMG_DIR  = DEEPCRACK_ROOT / "test_img"       # 237 test images
TEST_MASK_DIR = DEEPCRACK_ROOT / "test_lab"       # 237 test masks

OUTPUT_ROOT = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Paper\Review Paper 3\OUTPUT DeepCrack"
)

# Accept common image/mask extensions. Pairing is by identical file stem.
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


# =========================================================================
# 2. FROZEN PROTOCOL CONSTANTS
# =========================================================================
LOSS_ORDER = [
    "bce",
    "focal",
    "bce_dice",
    "focal_tversky",
    "dice_boundary",
    "dice_cldice",
]

# Protocol default for DeepCrack: one seed. Do not expand to [0,1,2] after
# seeing any test-set scoring. If compute allows and the decision is made
# before any test-set scoring, document the upgrade before running.
SEEDS = [0]

# DeepCrack: preserve the published 300/237 train/test split. Draw one fixed
# 20% validation subset from the 300 training images.
VAL_SPLIT_SEED = 20260705
EXPECTED_COUNTS = {"train_pool": 300, "train": 240, "val": 60, "test": 237}
STRICT_SPLIT_COUNTS = True
HASH_SPLIT_FILES = True

CROP_SIZE     = 448
BATCH_SIZE    = 8
EPOCHS        = 100
PATIENCE      = 20
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4
USE_AMP       = True
DETERMINISTIC = False
STRIDE        = 32

# Loss hyperparameters, fixed before training.
FOCAL_GAMMA   = 2.0
TVERSKY_ALPHA = 0.3
TVERSKY_BETA  = 0.7
FT_EXPONENT   = 0.75
CLDICE_ALPHA  = 0.3
CLDICE_ITERS  = 5
BOUNDARY_ALPHA_FLOOR = 0.01
BOUNDARY_ALPHA_STEP  = 0.01

# Model-selection criterion.
SELECT_METRIC = "val_f1_grid_selected_global"
THRESHOLD_GRID_DESCRIPTION = "0.01:0.99:0.01"
THRESHOLD_COMPARATOR = "p >= t"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = USE_AMP and DEVICE.type == "cuda"


# =========================================================================
# 3. Utilities
# =========================================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Explicitly seed Python random and NumPy inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "amp_enabled": AMP_ENABLED,
        "deterministic": DETERMINISTIC,
        "reproducibility_note": (
            "seed-controlled, bitwise deterministic"
            if DETERMINISTIC
            else "seed-controlled, not bitwise deterministic"
        ),
    }


def _find_by_stem(directory: Path, exts) -> dict:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    out = {}
    duplicates = []
    for ext in exts:
        for p in directory.glob(f"*{ext}"):
            key = p.stem
            if key in out:
                duplicates.append((key, out[key].name, p.name))
            else:
                out[key] = p

    # Also catch uppercase extensions on Windows or mixed archives.
    for ext in exts:
        for p in directory.glob(f"*{ext.upper()}"):
            key = p.stem
            if key in out:
                if out[key].resolve() != p.resolve():
                    duplicates.append((key, out[key].name, p.name))
            else:
                out[key] = p

    if duplicates:
        raise RuntimeError(
            f"Duplicate stems in {directory}. Example: {duplicates[:3]}"
        )
    return out


def list_pairs(img_dir: Path, mask_dir: Path):
    """Pair images to masks by identical file stem. Fails loudly on gaps."""
    imgs = _find_by_stem(img_dir, IMG_EXTS)
    masks = _find_by_stem(mask_dir, MASK_EXTS)

    if not imgs:
        raise FileNotFoundError(f"No images found in {img_dir}")
    if not masks:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    missing_masks = sorted(set(imgs) - set(masks))
    missing_images = sorted(set(masks) - set(imgs))
    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} image(s) missing masks in {mask_dir}; "
            f"examples: {missing_masks[:5]}"
        )
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} mask(s) missing images in {img_dir}; "
            f"examples: {missing_images[:5]}"
        )

    return [(imgs[stem], masks[stem]) for stem in sorted(imgs)]


def split_train_val(pairs):
    """Draw fixed 60-image validation subset from the 300 DeepCrack training images."""
    rng = random.Random(VAL_SPLIT_SEED)
    pairs = list(pairs)
    rng.shuffle(pairs)
    val_pairs = sorted(pairs[:EXPECTED_COUNTS["val"]], key=lambda x: x[0].name)
    train_pairs = sorted(pairs[EXPECTED_COUNTS["val"]:], key=lambda x: x[0].name)
    return train_pairs, val_pairs


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split_lists(splits: dict) -> None:
    """Persist exact filenames and optional SHA-256 hashes for each split."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, pairs in splits.items():
        lines = []
        for ip, mp in pairs:
            if HASH_SPLIT_FILES:
                lines.append(
                    f"{ip.name}\t{mp.name}\t{_sha256(ip)}\t{_sha256(mp)}"
                )
            else:
                lines.append(f"{ip.name}\t{mp.name}")
        (OUTPUT_ROOT / f"{name}_files.txt").write_text("\n".join(lines) + "\n")


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L"))
    return (m > 127).astype(np.float32)


def normalize(img: np.ndarray) -> np.ndarray:
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def compute_sdf(mask: np.ndarray) -> np.ndarray:
    """Signed distance to the crack boundary: positive outside, negative inside."""
    m = mask.astype(bool)
    if not m.any() or m.all():
        return np.zeros(mask.shape, dtype=np.float32)
    return (edt(~m) - edt(m)).astype(np.float32)


def pad_to_multiple(img: np.ndarray, stride: int = STRIDE):
    """Reflect-pad HxWx3 or HxW array so both dimensions divide stride."""
    h, w = img.shape[:2]
    ph = (stride - h % stride) % stride
    pw = (stride - w % stride) % stride
    if ph == 0 and pw == 0:
        return img, (h, w)

    pad_spec = ((0, ph), (0, pw)) + (((0, 0),) if img.ndim == 3 else ())
    return np.pad(img, pad_spec, mode="reflect"), (h, w)


# =========================================================================
# 4. Dataset
# =========================================================================
class DeepCrackTrainSet(Dataset):
    """448x448 random crops from native-resolution images."""

    def __init__(self, pairs, needs_sdf: bool):
        self.pairs = pairs
        self.needs_sdf = needs_sdf

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img, mask = load_image(ip), load_mask(mp)

        h, w = mask.shape
        ph, pw = max(0, CROP_SIZE - h), max(0, CROP_SIZE - w)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
            h, w = mask.shape

        top = random.randint(0, h - CROP_SIZE)
        left = random.randint(0, w - CROP_SIZE)
        img = img[top:top + CROP_SIZE, left:left + CROP_SIZE]
        mask = mask[top:top + CROP_SIZE, left:left + CROP_SIZE]

        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1, :], mask[::-1, :]
        k = random.randint(0, 3)
        if k:
            img, mask = np.rot90(img, k, (0, 1)), np.rot90(mask, k, (0, 1))

        b = random.uniform(0.8, 1.2)
        c = random.uniform(0.8, 1.2)
        img = np.clip((np.clip(img * b, 0, 1) - 0.5) * c + 0.5, 0, 1)

        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        sdf = compute_sdf(mask) if self.needs_sdf else np.zeros_like(mask)

        x = torch.from_numpy(normalize(img)).permute(2, 0, 1).float()
        y = torch.from_numpy(mask).unsqueeze(0).float()
        d = torch.from_numpy(sdf).unsqueeze(0).float()
        return x, y, d


# =========================================================================
# 5. Losses
# =========================================================================
EPS = 1e-6


def soft_dice_loss(probs, target):
    dims = (1, 2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
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
        return ((1 - pt) ** FOCAL_GAMMA * bce).mean()


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
        p = torch.sigmoid(logits)
        a = max(BOUNDARY_ALPHA_FLOOR, 1.0 - BOUNDARY_ALPHA_STEP * epoch)
        boundary = (p * sdf).mean()
        return a * soft_dice_loss(p, target) + (1 - a) * boundary


def _soft_erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)


def _soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skel(x, iters=CLDICE_ITERS):
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
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


def frozen_config() -> dict:
    return {
        "dataset": "DeepCrack",
        "arch": "unet_resnet34_imagenet",
        "train_pool": EXPECTED_COUNTS["train_pool"],
        "train": EXPECTED_COUNTS["train"],
        "val": EXPECTED_COUNTS["val"],
        "test": EXPECTED_COUNTS["test"],
        "val_split_seed": VAL_SPLIT_SEED,
        "crop": CROP_SIZE,
        "batch": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "patience": PATIENCE,
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
        "boundary_sdf": "unnormalized (Kervadec original), frozen",
        "threshold_grid": THRESHOLD_GRID_DESCRIPTION,
        "threshold_comparator": THRESHOLD_COMPARATOR,
        "probmap_dtype": "float32",
    }


# =========================================================================
# 6. Full-image inference and validation metric
# =========================================================================
@torch.no_grad()
def infer_full(model, img_path: Path) -> np.ndarray:
    img = load_image(img_path)
    padded, (h, w) = pad_to_multiple(img)
    x = torch.from_numpy(normalize(padded)).permute(2, 0, 1)[None].float()
    x = x.to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
        prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    return prob[:h, :w]


@torch.no_grad()
def validate_f1_grid(model, pairs):
    """Dataset-global F1 maximized over threshold grid 0.01..0.99."""
    model.eval()
    edges = np.linspace(0.0, 1.0, 101)
    pos_hist = np.zeros(100, dtype=np.float64)
    all_hist = np.zeros(100, dtype=np.float64)
    n_pos = 0.0

    for ip, mp in pairs:
        prob = infer_full(model, ip)
        gt = load_mask(mp) > 0.5
        pos_hist += np.histogram(prob[gt], bins=edges)[0]
        all_hist += np.histogram(prob, bins=edges)[0]
        n_pos += float(gt.sum())

    pos_tail = np.cumsum(pos_hist[::-1])[::-1]
    all_tail = np.cumsum(all_hist[::-1])[::-1]
    thresholds = np.arange(1, 100) / 100.0

    tp = pos_tail[1:100]
    predpos = all_tail[1:100]
    f1 = (2.0 * tp) / (predpos + n_pos + EPS)

    f1r = np.round(f1, 4)
    cand = np.flatnonzero(f1r == f1r.max())

    # Tie rule: threshold closest to 0.5, then lower threshold.
    order = np.lexsort((thresholds[cand], np.abs(thresholds[cand] - 0.5)))
    best = cand[order[0]]
    return float(f1[best]), float(thresholds[best])


@torch.no_grad()
def save_probmaps(model, pairs, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    model.eval()
    for ip, _ in pairs:
        prob = infer_full(model, ip)
        np.save(out_dir / f"{ip.stem}.npy", prob.astype(np.float32))


# =========================================================================
# 7. Run bookkeeping
# =========================================================================
def run_is_complete(run_dir: Path, n_val: int, n_test: int) -> bool:
    mp = run_dir / "manifest.json"
    if not mp.exists():
        return False
    try:
        manifest = json.loads(mp.read_text())
    except json.JSONDecodeError:
        return False
    if manifest.get("status") != "complete":
        return False
    if not (run_dir / "best.pt").exists():
        return False
    if len(list((run_dir / "probmaps_val").glob("*.npy"))) != n_val:
        return False
    if len(list((run_dir / "probmaps_test").glob("*.npy"))) != n_test:
        return False
    return True


def rebuild_summary() -> None:
    rows = []
    for loss_name in LOSS_ORDER:
        for seed in SEEDS:
            mp = OUTPUT_ROOT / f"{loss_name}_seed{seed}" / "manifest.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            rows.append([
                m.get("run_id"),
                m.get("loss"),
                m.get("seed"),
                m.get("status"),
                m.get("best_epoch"),
                m.get("best_val_f1"),
                m.get("best_val_threshold"),
                m.get("epochs_trained"),
            ])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "runs_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id",
            "loss",
            "seed",
            "status",
            "best_epoch",
            "best_val_f1",
            "best_val_threshold",
            "epochs_trained",
        ])
        w.writerows(rows)


# =========================================================================
# 8. One run = one loss + one seed
# =========================================================================
def train_one_run(loss_name: str, seed: int, train_pairs, val_pairs, test_pairs) -> None:
    run_id = f"{loss_name}_seed{seed}"
    run_dir = OUTPUT_ROOT / run_id
    manifest_path = run_dir / "manifest.json"

    if run_is_complete(run_dir, len(val_pairs), len(test_pairs)):
        print(f"[skip] {run_id} verified complete")
        return

    if manifest_path.exists():
        print(f"[redo] {run_id} has a manifest but failed verification; re-running")

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    manifest = {
        "run_id": run_id,
        "dataset": "DeepCrack",
        "loss": loss_name,
        "seed": seed,
        "status": "started",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": frozen_config(),
        "env": env_info(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    loss_fn = LOSS_REGISTRY[loss_name]()
    dataset = DeepCrackTrainSet(train_pairs, needs_sdf=loss_fn.needs_sdf)

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    best_f1, best_epoch, best_thr, epochs_no_improve = -1.0, -1, 0.5, 0
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss",
            "val_f1",
            "val_thr",
            "lr",
            "seconds",
        ])

    print(f"[run ] {run_id} starting on {DEVICE}")
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        running = 0.0

        for x, y, d in loader:
            x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
            optim.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                loss = loss_fn(model(x), y, sdf=d, epoch=epoch)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item()

        sched.step()
        train_loss = running / max(1, len(loader))

        val_f1, val_thr = validate_f1_grid(model, val_pairs)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.5f}",
                f"{val_f1:.5f}",
                f"{val_thr:.2f}",
                f"{sched.get_last_lr()[0]:.2e}",
                f"{time.time() - t0:.1f}",
            ])

        print(
            f"  {run_id} epoch {epoch:03d} "
            f"loss {train_loss:.4f} val_f1 {val_f1:.4f} thr {val_thr:.2f}"
        )

        # Epoch tie rule: compare F1 rounded to four decimals; earliest
        # epoch keeps the checkpoint on a tie.
        if round(val_f1, 4) > round(best_f1, 4):
            best_f1, best_epoch, best_thr, epochs_no_improve = (
                val_f1,
                epoch,
                val_thr,
                0,
            )
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  {run_id} early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=DEVICE))
    save_probmaps(model, val_pairs, run_dir / "probmaps_val")
    save_probmaps(model, test_pairs, run_dir / "probmaps_test")

    manifest.update({
        "status": "complete",
        "best_epoch": best_epoch,
        "best_val_f1": round(best_f1, 5),
        "best_val_threshold": round(best_thr, 2),
        "epochs_trained": epoch + 1,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"[done] {run_id} best_val_f1 {best_f1:.4f} "
        f"thr {best_thr:.2f} epoch {best_epoch}"
    )


# =========================================================================
# 9. Main
# =========================================================================
def main():
    trainval_pairs = list_pairs(TRAINVAL_IMG_DIR, TRAINVAL_MASK_DIR)
    test_pairs = list_pairs(TEST_IMG_DIR, TEST_MASK_DIR)

    train_pairs, val_pairs = split_train_val(trainval_pairs)

    counts = {
        "train_pool": len(trainval_pairs),
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
    }
    print(f"pairs found: {counts}  expected: {EXPECTED_COUNTS}")

    if STRICT_SPLIT_COUNTS and counts != EXPECTED_COUNTS:
        raise SystemExit(
            "DeepCrack counts do not match the frozen protocol "
            f"({EXPECTED_COUNTS}). Fix the directories or document the "
            "deviation before training."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_split_lists({
        "deepcrack_train": train_pairs,
        "deepcrack_val": val_pairs,
        "deepcrack_test": test_pairs,
    })

    for loss_name in LOSS_ORDER:
        for seed in SEEDS:
            train_one_run(loss_name, seed, train_pairs, val_pairs, test_pairs)
            rebuild_summary()

    rebuild_summary()
    print(
        "All DeepCrack runs complete. Probability maps are ready for the "
        "offline scoring suite."
    )


if __name__ == "__main__":
    main()
