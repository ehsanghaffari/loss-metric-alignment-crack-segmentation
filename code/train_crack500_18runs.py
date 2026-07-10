"""
Crack500 controlled demonstration: all 18 training runs back-to-back.
Paper 3 protocol (rev. 3): 6 losses x 3 seeds, one fixed backbone.
Script rev. 5. Changes from rev. 4:
  * epoch tie rule now literally matches the freeze document: epochs are
    compared on F1 ROUNDED TO FOUR DECIMALS, so a later epoch improving
    only past the fourth decimal does not displace the earlier checkpoint,
  * threshold_grid and threshold_comparator recorded as machine-readable
    manifest fields.
Script rev. 4. Changes from rev. 3 (protocol review):
  * model selection is now threshold-free: best epoch by dataset-global F1
    maximized over the frozen 0.01-0.99 grid on the validation set (a
    fixed-0.5 rule would bias checkpoints toward losses calibrated near
    0.5, contradicting the paper's own thesis),
  * the argmax threshold at the best epoch is recorded per run as the
    deployable validation-selected threshold (val_thr column + manifest),
  * binarization comparator frozen as p >= t,
  * deterministic tie-breaking: earliest epoch; threshold closest to 0.5,
    then the lower threshold.
Script rev. 3. Changes from rev. 2:
  * explicit DataLoader generator per run: shuffle order and worker seeds
    are pinned to the run seed itself, independent of prior global RNG
    consumption (e.g. model init),
  * boundary-loss comment softened (early stopping limits, not guarantees,
    the influence of the boundary-dominant late regime),
  * python/scipy/pillow versions logged in every manifest,
  * optional SHA-256 hashes of every image and mask in the split lists
    (HASH_SPLIT_FILES, default True) for verifiable dataset provenance.
Script rev. 2, after external code review. Changes from rev. 1:
  * probability maps saved as float32 (released core artifact; no
    quantization questions),
  * split counts hard-enforced (1896/348/1124) via STRICT_SPLIT_COUNTS,
  * exact train/val/test filename lists written to the output root,
  * "started" manifest written before training; resume now verifies
    best.pt and probmap counts before skipping a run,
  * runs_summary.csv rebuilt from manifests (no duplicate rows on resume),
  * AMP gated to CUDA only; DETERMINISTIC flag added (default False:
    runs are seed-controlled, not bitwise deterministic, and the manifest
    records this),
  * stale probmap directories cleared before saving,
  * environment versions recorded in every manifest,
  * clDice alpha relabeled as a pre-specified protocol value (Shit et al.
    2021 study alpha in [0.1, 0.5] without a single canonical default).

Runs, in order:
    for loss in [bce, focal, bce_dice, focal_tversky, dice_boundary, dice_cldice]:
        for seed in [0, 1, 2]:
            train -> select best checkpoint on val F1@0.5 ->
            save probability maps for VAL and TEST at native resolution.

Protocol commitments implemented here (see plan rev. 3, Section 3):
  * U-Net + ResNet-34 (ImageNet init), sigmoid/logit output, one backbone.
  * No resizing anywhere. Training uses 448x448 random crops (constant zero
    padding if a patch is smaller). Validation/test inference runs on full
    native-resolution images, padded to a multiple of 32 and cropped back
    before anything is stored.
  * AdamW + cosine schedule, fixed epochs with early stopping.
  * Loss hyperparameters fixed before training (below), never tuned per
    dataset.
  * Saved artifacts per run: best checkpoint, per-epoch log, manifest.json,
    float32 probability maps for every val and test image (val maps are
    required later for the validation-selected threshold convention).
  * Re-running the script skips runs verified complete, so it can resume
    after an interruption.

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import scipy
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
# 1. FILL THESE IN  (the only block you need to edit)
# =========================================================================
CRACK500_ROOT = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Dataset\Crack500"
)

TRAIN_IMG_DIR  = CRACK500_ROOT / "traincrop" / "traincrop"   # 1,896 training images
TRAIN_MASK_DIR = CRACK500_ROOT / "traincrop" / "traincrop"   # 1,896 training masks

VAL_IMG_DIR    = CRACK500_ROOT / "valcrop" / "valcrop"       # 348 validation images
VAL_MASK_DIR   = CRACK500_ROOT / "valcrop" / "valcrop"       # 348 validation masks

TEST_IMG_DIR   = CRACK500_ROOT / "testcrop" / "testcrop"     # 1,124 test images
TEST_MASK_DIR  = CRACK500_ROOT / "testcrop" / "testcrop"     # 1,124 test masks

IMG_EXT  = ".jpg"    # image file extension
MASK_EXT = ".png"    # mask filename = image stem + MASK_EXT

OUTPUT_ROOT = Path(
    r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Paper\Review Paper 3\OUTPUT Crack500"
)

# =========================================================================
# 2. FROZEN PROTOCOL CONSTANTS
#    Adjust only before the week-1 protocol freeze, then record every value
#    in the protocol document and never change them mid-study.
# =========================================================================
LOSS_ORDER = ["bce", "focal", "bce_dice", "focal_tversky",
              "dice_boundary", "dice_cldice"]
SEEDS = [0, 1, 2]

# Published Crack500 patch-level partition; enforced before any training.
# Overriding STRICT_SPLIT_COUNTS is an explicit protocol deviation and must
# be documented as such.
STRICT_SPLIT_COUNTS = True
EXPECTED_COUNTS = {"train": 1896, "val": 348, "test": 1124}
HASH_SPLIT_FILES = True   # record SHA-256 of every image and mask in the
                          # split lists (verifiable dataset provenance;
                          # costs seconds once at startup)

CROP_SIZE    = 448       # training crop (no resizing, ever)
BATCH_SIZE   = 8
EPOCHS       = 100       # cosine schedule length
PATIENCE     = 20        # early stop on best val F1@0.5
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
USE_AMP      = True      # mixed precision (active on CUDA only)
DETERMINISTIC = False    # False = seed-controlled but not bitwise
                         # deterministic (recorded in manifest). True =
                         # cudnn deterministic mode, slower.
STRIDE       = 32        # encoder downsampling factor (pad multiple)

# Loss hyperparameters, fixed before training (record in protocol doc)
FOCAL_GAMMA   = 2.0      # Lin et al. 2017, unweighted binary focal
TVERSKY_ALPHA = 0.3      # FP weight  (Abraham & Khan 2019)
TVERSKY_BETA  = 0.7      # FN weight -> recall-oriented
FT_EXPONENT   = 0.75     # focal Tversky: (1 - TI)^(1/gamma), gamma = 4/3
CLDICE_ALPHA  = 0.3      # PRE-SPECIFIED PROTOCOL VALUE. Shit et al. 2021
                         # study alpha in [0.1, 0.5] with no single
                         # canonical default; 0.3 is our frozen mid-range
                         # choice and must be described as such, not as a
                         # paper default.
CLDICE_ITERS  = 5        # soft-skeletonization iterations (report per plan)
# Kervadec boundary loss: L = a*Dice + (1-a)*Boundary,
# a = max(0.01, 1 - 0.01*epoch)  (paper's default rebalancing schedule).
# FROZEN DECISION: the SDF stays unnormalized, as in the original paper.
# Known property: the background term is numerically dominant for thin
# cracks and can push predictions conservative late in training; early
# stopping on val F1 limits the influence of the boundary-dominant
# late-training regime, without guaranteeing it.
# Do not revisit this after seeing results.
BOUNDARY_ALPHA_FLOOR = 0.01
BOUNDARY_ALPHA_STEP  = 0.01

# Model-selection criterion (a protocol variable in its own right; frozen):
# best epoch = highest validation-selected dataset-global F1, i.e. for each
# epoch, F1 is maximized over the frozen threshold grid (0.01 to 0.99, step
# 0.01) on the validation set. Threshold-free selection avoids biasing
# checkpoints toward losses calibrated near 0.5. Binarization comparator
# frozen: a pixel is predicted positive iff p >= t. Ties at four decimals:
# earliest epoch; threshold closest to 0.5, then the lower threshold. The
# best epoch's argmax threshold is recorded as the run's deployable
# validation-selected threshold (canonically recomputed offline from the
# released validation maps).
SELECT_METRIC = "val_f1_grid_selected_global"

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
        # Seed variance is studied across seeds; bitwise determinism within
        # a run is not claimed. Recorded in the manifest.
        torch.backends.cudnn.benchmark = True


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
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "cpu"),
        "amp_enabled": AMP_ENABLED,
        "deterministic": DETERMINISTIC,
        "reproducibility_note": ("seed-controlled, bitwise deterministic"
                                 if DETERMINISTIC else
                                 "seed-controlled, not bitwise deterministic"),
    }


def list_pairs(img_dir: Path, mask_dir: Path):
    """Pair images to masks by identical file stem. Fails loudly on gaps."""
    imgs = sorted(img_dir.glob(f"*{IMG_EXT}"))
    if not imgs:
        raise FileNotFoundError(f"No '{IMG_EXT}' images found in {img_dir}")
    pairs = []
    missing = []
    for p in imgs:
        m = mask_dir / (p.stem + MASK_EXT)
        if m.exists():
            pairs.append((p, m))
        else:
            missing.append(m.name)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} masks missing in {mask_dir}, "
            f"e.g. {missing[:3]}"
        )
    return pairs


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split_lists(splits: dict) -> None:
    """Persist the exact filenames used per split (reproducibility claim).
    With HASH_SPLIT_FILES, each line also carries the SHA-256 of the image
    and of the mask, making dataset provenance verifiable byte-for-byte."""
    for name, pairs in splits.items():
        lines = []
        for ip, mp in pairs:
            if HASH_SPLIT_FILES:
                lines.append(
                    f"{ip.name}\t{mp.name}\t{_sha256(ip)}\t{_sha256(mp)}")
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
    """Signed distance to the crack boundary: positive outside the crack,
    negative inside (Kervadec convention). Zeros for empty/full masks so the
    boundary term vanishes on crack-free crops."""
    m = mask.astype(bool)
    if not m.any() or m.all():
        return np.zeros(mask.shape, dtype=np.float32)
    return (edt(~m) - edt(m)).astype(np.float32)


def pad_to_multiple(img: np.ndarray, stride: int = STRIDE):
    """Reflect-pad HxWx3 (or HxW) array so both dims divide `stride`.
    Returns (padded, (orig_h, orig_w)). Predictions are cropped back."""
    h, w = img.shape[:2]
    ph = (stride - h % stride) % stride
    pw = (stride - w % stride) % stride
    if ph == 0 and pw == 0:
        return img, (h, w)
    pad_spec = ((0, ph), (0, pw)) + (((0, 0),) if img.ndim == 3 else ())
    return np.pad(img, pad_spec, mode="reflect"), (h, w)


# =========================================================================
# 4. Dataset (training crops with augmentation)
# =========================================================================
class Crack500TrainSet(Dataset):
    """448x448 random crops from native-resolution patches.
    Augmentation: pad-if-small (constant zero for image and mask), random
    crop, h/v flip, 90-degree rotations, brightness/contrast jitter.
    Geometry is applied identically to image and mask; the SDF for the
    boundary loss is computed AFTER all geometry, on the final mask."""

    def __init__(self, pairs, needs_sdf: bool):
        self.pairs = pairs
        self.needs_sdf = needs_sdf

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img, mask = load_image(ip), load_mask(mp)

        # pad up to CROP_SIZE if a native patch is smaller in either dim
        h, w = mask.shape
        ph, pw = max(0, CROP_SIZE - h), max(0, CROP_SIZE - w)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
            h, w = mask.shape

        # random crop
        top = random.randint(0, h - CROP_SIZE)
        left = random.randint(0, w - CROP_SIZE)
        img = img[top:top + CROP_SIZE, left:left + CROP_SIZE]
        mask = mask[top:top + CROP_SIZE, left:left + CROP_SIZE]

        # flips and 90-degree rotations
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1, :], mask[::-1, :]
        k = random.randint(0, 3)
        if k:
            img, mask = np.rot90(img, k, (0, 1)), np.rot90(mask, k, (0, 1))

        # photometric jitter (image only)
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
#    Every loss exposes: needs_sdf (bool) and
#    forward(logits, target, sdf, epoch) -> scalar
# =========================================================================
EPS = 1e-6


def soft_dice_loss(probs, target):
    """Per-sample soft Dice, averaged over the batch (frozen convention)."""
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
    """Unweighted binary focal loss, gamma = FOCAL_GAMMA."""
    needs_sdf = False
    def forward(self, logits, target, sdf=None, epoch=0):
        bce = F.binary_cross_entropy_with_logits(logits, target,
                                                 reduction="none")
        pt = torch.exp(-bce)
        return ((1 - pt) ** FOCAL_GAMMA * bce).mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice summed at 1:1 (true unweighted sum; report as such)."""
    needs_sdf = False
    def forward(self, logits, target, sdf=None, epoch=0):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return bce + soft_dice_loss(torch.sigmoid(logits), target)


class FocalTverskyLoss(nn.Module):
    """(1 - TI)^FT_EXPONENT with TI = TP/(TP + a*FP + b*FN); a=0.3, b=0.7.
    FT_EXPONENT = 0.75 realizes (1 - TI)^(1/gamma) with gamma = 4/3
    (Abraham & Khan 2019); state this definition in the manuscript."""
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
    """Kervadec-style: a*Dice + (1-a)*mean(prob * SDF), with the paper's
    default rebalancing schedule a = max(0.01, 1 - 0.01*epoch).
    FROZEN: SDF is unnormalized, as in the original (see constant block)."""
    needs_sdf = True
    def forward(self, logits, target, sdf=None, epoch=0):
        p = torch.sigmoid(logits)
        a = max(BOUNDARY_ALPHA_FLOOR, 1.0 - BOUNDARY_ALPHA_STEP * epoch)
        boundary = (p * sdf).mean()
        return a * soft_dice_loss(p, target) + (1 - a) * boundary


# ---- soft skeletonization for clDice (Shit et al. 2021) -----------------
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
    """(1 - CLDICE_ALPHA)*Dice + CLDICE_ALPHA*(1 - soft clDice).
    Computed on probability maps (loss side); iterations = CLDICE_ITERS."""
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
    "bce":            BCELoss,
    "focal":          FocalLoss,
    "bce_dice":       BCEDiceLoss,
    "focal_tversky":  FocalTverskyLoss,
    "dice_boundary":  DiceBoundaryLoss,
    "dice_cldice":    DiceClDiceLoss,
}


def frozen_config() -> dict:
    return {
        "arch": "unet_resnet34_imagenet",
        "crop": CROP_SIZE, "batch": BATCH_SIZE,
        "epochs_max": EPOCHS, "patience": PATIENCE,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "select_metric": SELECT_METRIC,
        "focal_gamma": FOCAL_GAMMA,
        "tversky_alpha": TVERSKY_ALPHA, "tversky_beta": TVERSKY_BETA,
        "ft_exponent": FT_EXPONENT,
        "cldice_alpha": CLDICE_ALPHA, "cldice_iters": CLDICE_ITERS,
        "cldice_alpha_note": "pre-specified protocol value, not a paper default",
        "boundary_alpha_floor": BOUNDARY_ALPHA_FLOOR,
        "boundary_alpha_step": BOUNDARY_ALPHA_STEP,
        "boundary_sdf": "unnormalized (Kervadec original), frozen",
        "threshold_grid": "0.01:0.99:0.01",
        "threshold_comparator": "p >= t",
        "probmap_dtype": "float32",
    }


# =========================================================================
# 6. Full-image inference (native resolution) and validation metric
# =========================================================================
@torch.no_grad()
def infer_full(model, img_path: Path) -> np.ndarray:
    """Probability map at native resolution: reflect-pad to /32, forward,
    crop back. Returns float32 HxW in [0, 1]."""
    img = load_image(img_path)
    padded, (h, w) = pad_to_multiple(img)
    x = torch.from_numpy(normalize(padded)).permute(2, 0, 1)[None].float()
    x = x.to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
        prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    return prob[:h, :w]


@torch.no_grad()
def validate_f1_grid(model, pairs):
    """Dataset-global F1 maximized over the frozen threshold grid
    (0.01 to 0.99, step 0.01). Binarization comparator: p >= t.
    Returns (best_f1, best_threshold).
    Tie rule (frozen): among thresholds whose F1 ties to four decimal
    places, the one closest to 0.5 wins; if still tied, the lower
    threshold. Histogram implementation: the full 99-threshold sweep costs
    about the same as a single-threshold pass."""
    model.eval()
    edges = np.linspace(0.0, 1.0, 101)   # bin i covers [i/100, (i+1)/100)
    pos_hist = np.zeros(100)             # probs on ground-truth-positive px
    all_hist = np.zeros(100)             # probs on all px
    n_pos = 0.0
    for ip, mp in pairs:
        prob = infer_full(model, ip)
        gt = load_mask(mp) > 0.5
        pos_hist += np.histogram(prob[gt], bins=edges)[0]
        all_hist += np.histogram(prob, bins=edges)[0]
        n_pos += float(gt.sum())
    # count(p >= k/100) is the tail sum of bins k..99
    pos_tail = np.cumsum(pos_hist[::-1])[::-1]
    all_tail = np.cumsum(all_hist[::-1])[::-1]
    thresholds = np.arange(1, 100) / 100.0
    tp = pos_tail[1:100]
    predpos = all_tail[1:100]
    # F1 = 2TP / (2TP + FP + FN) = 2TP / (predpos + n_pos)
    f1 = (2.0 * tp) / (predpos + n_pos + EPS)
    f1r = np.round(f1, 4)
    cand = np.flatnonzero(f1r == f1r.max())
    order = np.lexsort((thresholds[cand],
                        np.abs(thresholds[cand] - 0.5)))
    best = cand[order[0]]
    return float(f1[best]), float(thresholds[best])


@torch.no_grad()
def save_probmaps(model, pairs, out_dir: Path) -> None:
    """Float32 .npy probability map per image, named by image stem.
    The directory is cleared first so no stale maps from a previous
    attempt can survive."""
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
    """A run counts as complete only if the manifest says so AND every
    artifact is present: best.pt plus exactly one probmap per val/test
    image."""
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
    """Regenerate runs_summary.csv from the manifests on disk. Idempotent:
    resuming never duplicates rows."""
    rows = []
    for loss_name in LOSS_ORDER:
        for seed in SEEDS:
            mp = OUTPUT_ROOT / f"{loss_name}_seed{seed}" / "manifest.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            rows.append([m.get("run_id"), m.get("loss"), m.get("seed"),
                         m.get("status"), m.get("best_epoch"),
                         m.get("best_val_f1")])
    with open(OUTPUT_ROOT / "runs_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "loss", "seed", "status", "best_epoch",
                    "best_val_f1"])
        w.writerows(rows)


# =========================================================================
# 8. One run = one loss + one seed
# =========================================================================
def train_one_run(loss_name: str, seed: int,
                  train_pairs, val_pairs, test_pairs) -> None:
    run_id = f"{loss_name}_seed{seed}"
    run_dir = OUTPUT_ROOT / run_id
    manifest_path = run_dir / "manifest.json"

    if run_is_complete(run_dir, len(val_pairs), len(test_pairs)):
        print(f"[skip] {run_id} verified complete")
        return
    if manifest_path.exists():
        print(f"[redo] {run_id} has a manifest but failed verification; "
              f"re-running")

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    # "started" manifest: config is on disk even if this attempt crashes
    manifest = {
        "run_id": run_id, "loss": loss_name, "seed": seed,
        "status": "started",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": frozen_config(),
        "env": env_info(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    loss_fn = LOSS_REGISTRY[loss_name]()
    dataset = Crack500TrainSet(train_pairs, needs_sdf=loss_fn.needs_sdf)
    # Explicit generator: shuffle order is pinned to the run seed itself,
    # not to whatever global RNG state remains after model init, and the
    # DataLoader derives its worker seeds (which drive the augmentation
    # streams) from this generator as well.
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=g, num_workers=NUM_WORKERS,
                        pin_memory=True, drop_last=True)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=1, activation=None).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    best_f1, best_epoch, best_thr, epochs_no_improve = -1.0, -1, 0.5, 0
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_f1", "val_thr",
                                "lr", "seconds"])

    print(f"[run ] {run_id} starting on {DEVICE}")
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        running = 0.0
        for x, y, d in loader:
            x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type,
                                enabled=AMP_ENABLED):
                loss = loss_fn(model(x), y, sdf=d, epoch=epoch)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item()
        sched.step()
        train_loss = running / max(1, len(loader))

        val_f1, val_thr = validate_f1_grid(model, val_pairs)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.5f}", f"{val_f1:.5f}",
                 f"{val_thr:.2f}",
                 f"{sched.get_last_lr()[0]:.2e}", f"{time.time()-t0:.1f}"])
        print(f"  {run_id} epoch {epoch:03d} "
              f"loss {train_loss:.4f} val_f1 {val_f1:.4f} thr {val_thr:.2f}")

        # Frozen tie rule: epochs are compared at FOUR decimal places; on
        # a tie the earliest epoch keeps the checkpoint. Raw values are
        # still stored and reported.
        if round(val_f1, 4) > round(best_f1, 4):
            best_f1, best_epoch, best_thr, epochs_no_improve = (
                val_f1, epoch, val_thr, 0)
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  {run_id} early stop at epoch {epoch}")
                break

    # reload best checkpoint, dump probability maps for val AND test
    model.load_state_dict(torch.load(run_dir / "best.pt",
                                     map_location=DEVICE))
    save_probmaps(model, val_pairs, run_dir / "probmaps_val")
    save_probmaps(model, test_pairs, run_dir / "probmaps_test")

    manifest.update({
        "status": "complete",
        "best_epoch": best_epoch,
        "best_val_f1": round(best_f1, 5),
        # Deployable operating point; canonically recomputed offline from
        # the released validation probability maps.
        "best_val_threshold": round(best_thr, 2),
        "epochs_trained": epoch + 1,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] {run_id} best_val_f1 {best_f1:.4f} (epoch {best_epoch})")


# =========================================================================
# 9. Main: all 18 runs back-to-back
# =========================================================================
def main():
    for name, d in [("TRAIN_IMG_DIR", TRAIN_IMG_DIR),
                    ("TRAIN_MASK_DIR", TRAIN_MASK_DIR),
                    ("VAL_IMG_DIR", VAL_IMG_DIR),
                    ("VAL_MASK_DIR", VAL_MASK_DIR),
                    ("TEST_IMG_DIR", TEST_IMG_DIR),
                    ("TEST_MASK_DIR", TEST_MASK_DIR)]:
        if "FILL_ME" in str(d):
            raise SystemExit(f"Please fill in {name} at the top of the file.")
    if "FILL_ME" in str(OUTPUT_ROOT):
        raise SystemExit("Please fill in OUTPUT_ROOT at the top of the file.")

    train_pairs = list_pairs(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_pairs = list_pairs(VAL_IMG_DIR, VAL_MASK_DIR)
    test_pairs = list_pairs(TEST_IMG_DIR, TEST_MASK_DIR)

    counts = {"train": len(train_pairs), "val": len(val_pairs),
              "test": len(test_pairs)}
    print(f"pairs found: {counts}  expected: {EXPECTED_COUNTS}")
    if STRICT_SPLIT_COUNTS and counts != EXPECTED_COUNTS:
        raise SystemExit(
            "Split counts do not match the published Crack500 partition "
            f"({EXPECTED_COUNTS}). Fix the directories. Overriding "
            "STRICT_SPLIT_COUNTS is a protocol deviation and must be "
            "documented as such."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_split_lists({"train": train_pairs, "val": val_pairs,
                       "test": test_pairs})

    for loss_name in LOSS_ORDER:
        for seed in SEEDS:
            train_one_run(loss_name, seed,
                          train_pairs, val_pairs, test_pairs)
            rebuild_summary()   # idempotent; safe on resume

    rebuild_summary()
    print("All 18 runs complete. Probability maps are ready for the "
          "offline scoring suite.")


if __name__ == "__main__":
    main()
