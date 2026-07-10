# Dataset preparation

The repository does not redistribute Crack500 or DeepCrack. Dataset files remain subject to the original providers’ terms.

## Crack500

The training script expects the published patch-level split:

| Split | Images |
|---|---:|
| Train | 1,896 |
| Validation | 348 |
| Test | 1,124 |

Default folder assumptions are declared in the path block at the top of `code/train_crack500_18runs.py`. Image/mask pairing is by identical stem, with `.jpg` images and `.png` masks in the uploaded script configuration.

## DeepCrack

The training script preserves the published 300-image training pool and 237-image test set. A frozen 60-image validation subset is selected once from the training pool using `VAL_SPLIT_SEED = 20260705`, leaving 240 images for training.

Default folder assumptions are declared in the path block at the top of `code/train_deepcrack_6runs.py`. Common image and mask extensions are accepted and pairing is by identical stem.

## Integrity records

Both training scripts can write exact split lists with SHA-256 hashes for every image and mask. These files should be archived with checkpoints and probability maps for a fully auditable release.
