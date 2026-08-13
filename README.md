# What to Optimize and How to Measure It

## Loss–Metric Alignment for Deep Crack Segmentation

Companion repository for **What to Optimize and How to Measure It: Loss–Metric Alignment for Deep Crack Segmentation** by Ehsan Ghaffari, Kelvin C. P. Wang, Philip Barutha, and Neda Nazemi.

## Final controlled demonstration

The current manuscript uses **4 public datasets × 6 losses × 3 seeds = 72 training runs**.

| Dataset | Train | Validation | Test | Runs |
|---|---:|---:|---:|---:|
| Crack500 | 1,896 | 348 | 1,124 | 18 |
| DeepCrack | 240 | 60 | 237 | 18 |
| CFD | 82 | 12 | 24 | 18 |
| CrackTree260 | 182 | 26 | 52 | 18 |
| **Total** |  |  |  | **72** |

### Fixed training protocol

- U-Net with ImageNet-pretrained ResNet-34 encoder
- Losses: BCE, Focal, BCE + Dice, Focal Tversky, Dice + Boundary, Dice + clDice
- Seeds: `0`, `1`, `2`
- Random `448 × 448` training crops; no resizing
- AdamW, learning rate `1e-4`, weight decay `1e-4`
- Cosine learning-rate schedule
- Batch size `8`
- Exactly `100` epochs per run
- **Early stopping disabled**
- Validation dataset-global F1 used for checkpoint selection over thresholds `0.01`–`0.99`
- Native-resolution inference with reflect padding to a multiple of 32 and crop-back
- Float32 validation/test probability maps saved for offline scoring
- Seed-controlled runs; bitwise determinism is not claimed

### Loss hyperparameters

- Focal: `gamma = 2.0`
- Focal Tversky: `alpha = 0.3`, `beta = 0.7`, exponent `0.75`
- Dice + clDice: clDice weight `0.3`, 5 soft-skeleton iterations
- Dice + Boundary: Dice weight `max(0.01, 1 - 0.01 × epoch)` with complementary boundary weight

### Offline scoring

- Comparator: `p >= t`
- Threshold conventions: fixed `0.5`, validation-selected, ODS, OIS
- Relaxed-F1 radii: `r ∈ {0, 2, 3, 4, 5}` px
- Boundary F1 tolerance: `2` px
- clDice and 8-connected fragmentation/component-count error
- Dataset-level area and skeleton-length errors
- Per-image and dataset-global aggregation
- Explicit empty-mask rules and case counts
- Paired-image bootstrap for strict aggregation reversals

ODS and OIS are test-label-dependent **oracle upper bounds**, not deployable operating points.

## Selected final results

At the validation-selected operating point, the highest mean global F1 is obtained by Dice + Boundary on Crack500 (`0.7392 ± 0.0004`) and DeepCrack (`0.8428 ± 0.0081`), and by BCE + Dice on CFD (`0.6822 ± 0.0044`) and CrackTree260 (`0.6309 ± 0.0231`).

On Crack500, Dice + clDice has the highest clDice (`0.8021 ± 0.0036`) but the largest skeleton-length error (`63.3% ± 13.3%`), illustrating the study's central loss–metric alignment point.

For the strict-score leader on each dataset, relaxed F1 increases from `r=0` to `r=5` by `0.1516` (Crack500), `0.1246` (DeepCrack), `0.2667` (CFD), and `0.3462` (CrackTree260).

## Reproducibility files

The final v2 entry points are:

- `code/prepare_paper3_extension_splits_v2_1.py`
- `code/train_all_four_datasets_72runs_full100_no_early_stopping_v2.py`
- `code/score_paper_probmaps_v2_preflight.py`
- `protocol/paper3_protocol_v2_no_early_stopping.md`

The SHA-256 identities of the exact local execution files are recorded in `FINAL_SCRIPT_IDENTITIES.md`. The exact files matching those hashes should also be preserved in the reviewer/archive package. The historical v1 protocol is retained only for provenance; the manuscript results use the v2 full-100-epoch/no-early-stopping pipeline.

## Results and supplementary material

- `results/summary_by_dataset_loss_val_selected.csv` — final 72-run mean/SD summary
- `results/validation_thresholds.csv` — final per-run validation-selected thresholds
- `results/training_*.csv` — final training summaries for all four datasets
- `results/final_run_matrix.csv` — exact split counts and run counts
- `supplementary/Table_S1_Dataset_Characteristics.md` — dataset/preprocessing summary

Large per-image tables and native-resolution probability maps are intentionally not stored as ordinary Git files. Their archival identities and intended release status are documented in `results/README.md`.

## Data and artifacts

The original Crack500, DeepCrack, CFD, and CrackTree260 datasets are not redistributed and remain subject to their providers' licenses. Compact result files and reproducibility documentation are versioned here. Large artifacts should be supplied through the versioned reviewer/archive release.

## License

Code is released under the MIT License. Original dataset licenses remain with the dataset providers.
