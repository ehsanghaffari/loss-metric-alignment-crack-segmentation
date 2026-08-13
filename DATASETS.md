# Dataset preparation

The final controlled demonstration uses four public crack-segmentation datasets. The repository does **not** redistribute the original image datasets; dataset files remain subject to the original providers' terms.

| Dataset | Train | Validation | Test | Split provenance |
|---|---:|---:|---:|---|
| Crack500 | 1,896 | 348 | 1,124 | Published patch-level train/validation/test split preserved |
| DeepCrack | 240 | 60 | 237 | Published 300/237 train/test division preserved; fixed 60-image validation subset drawn from the training pool with seed `20260705` |
| CFD | 82 | 12 | 24 | Fixed deterministic 70/10/20 partition with seed `20260705` |
| CrackTree260 | 182 | 26 | 52 | Fixed deterministic 70/10/20 partition with seed `20260705` |

## Split preparation

DeepCrack, CFD, and CrackTree260 split creation and integrity checks are implemented in `code/prepare_paper3_extension_splits_v2_1.py`. The script validates image/mask pairing and size agreement, checks mask encodings, writes `train.csv`, `val.csv`, and `test.csv`, and records per-file SHA-256 hashes in `split_manifest.json`.

Crack500 uses the published patch directories directly in the final 72-run training script.

## Training and evaluation preprocessing

- Random `448 × 448` training crops.
- Images smaller than the crop are padded before sampling.
- Horizontal and vertical flips each use probability 0.5.
- Rotation is sampled from 0°, 90°, 180°, and 270°.
- Brightness and contrast factors are sampled independently from `[0.8, 1.2]`.
- ImageNet mean/std normalization is used.
- No resizing is used for training or evaluation.
- Full-image inference uses reflect padding to a multiple of 32, followed by crop-back to the original dimensions.
- Tolerance radii are applied at the retained evaluation resolution.

See `supplementary/Table_S1_Dataset_Characteristics.md` for the manuscript-oriented dataset summary.
