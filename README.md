# What to Optimize and How to Measure It

## Loss–Metric Alignment for Deep Crack Segmentation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-segmentation-ee4c2c.svg)](https://pytorch.org/)
[![Protocol](https://img.shields.io/badge/protocol-frozen-success.svg)](protocol/paper3_protocol_freeze_v1.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Companion code, frozen protocol, and result archive for:

> **What to Optimize and How to Measure It: Loss–Metric Alignment for Deep Crack Segmentation**
>
> Ehsan Ghaffari, Kelvin C. P. Wang, and Philip Barutha  
> Department of Civil Engineering, Montana State University, Bozeman, Montana, USA

**Project page:** `https://ehsanghaffari.github.io/loss-metric-alignment-crack-segmentation/`

## Purpose

This repository supports a review and controlled demonstration built around one question:

> **Does the apparent best training loss change when the evaluation yardstick changes?**

The study is not another universal loss leaderboard. It treats the training objective and evaluation metric as a coupled **claim–verifier system**. Identical saved probability maps are rescored under different metric families and protocol conventions so that ranking changes can be attributed to the evaluation yardstick rather than retraining.

The analysis varies:

- metric family;
- tolerance radius;
- threshold convention;
- aggregation convention;
- dataset regime; and
- empty-mask handling.

## Main finding

There is no universally best loss independent of the evaluation protocol. A loss can lead under overlap while another leads under topology, boundary quality, fragmentation, or geometric quantification. On Crack500, Focal and Dice + Boundary are effectively tied under dataset-global F1, whereas Dice + clDice leads clDice but has the largest skeleton-length error. The result is an alignment problem, not a single-winner conclusion.

## Controlled demonstration

### Datasets and run matrix

| Dataset | Train | Validation | Test | Runs |
|---|---:|---:|---:|---:|
| Crack500 | 1,896 | 348 | 1,124 | 6 losses × 3 seeds = 18 |
| DeepCrack | 240 | 60 | 237 | 6 losses × 1 seed = 6 |
| **Total** |  |  |  | **24 runs** |

The datasets are not redistributed. Obtain Crack500 and DeepCrack from their original providers and edit only the path blocks at the top of the training and scoring scripts.

### Fixed model and optimization protocol

- U-Net with an ImageNet-pretrained ResNet-34 encoder
- Binary pixel-level segmentation
- Random `448 × 448` training crops; no resizing anywhere
- Full native-resolution evaluation with stride-compatible reflect padding and crop-back
- AdamW, learning rate `1e-4`, weight decay `1e-4`
- Cosine schedule, 100-epoch maximum, early-stopping patience 20
- Batch size 8
- Float32 validation and test probability maps saved for offline scoring
- Checkpoint selection by validation dataset-global F1 maximized over the frozen threshold grid

### Losses

1. Binary cross-entropy
2. Focal loss
3. BCE + Dice
4. Focal Tversky
5. Dice + Boundary
6. Dice + clDice

### Frozen scoring protocol

- Threshold grid: `0.01`–`0.99` in increments of `0.01`
- Comparator: `p >= t`
- Threshold conventions: fixed `0.5`, validation-selected, ODS, and OIS
- Relaxed F1 radii: `r ∈ {0, 2, 3, 4, 5}` px
- Boundary F1 tolerance: `2` px at native resolution
- Topology: clDice and 8-connected fragmentation error
- Quantification: dataset-level area and skeleton-length errors
- Aggregation: per-image mean and dataset-global pooling
- Explicit empty-mask rules and case counts

ODS and OIS are test-label-dependent **oracle upper bounds**, not deployable operating points. Fixed `0.5` and validation-selected thresholds are deployable conventions.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── code/
│   ├── train_crack500_18runs.py
│   ├── train_deepcrack_6runs.py
│   └── score_paper_probmaps.py
├── protocol/
│   └── paper3_protocol_freeze_v1.md
├── results/
│   ├── validation_thresholds.csv
│   ├── overlap_summary_all_thresholds.csv
│   ├── advanced_summary_val_selected.csv
│   ├── summary_by_dataset_loss_val_selected.csv
│   ├── scoring_manifest.json
│   ├── OFFLINE_SCORE_OUTPUT.zip
│   └── README.md
└── docs/
    ├── index.html
    ├── styles.css
    └── assets/
```

`OFFLINE_SCORE_OUTPUT.zip` contains the full per-image overlap and advanced-metric tables in addition to the summary outputs. The compact summary CSVs are also versioned separately for convenient inspection.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Reproduction workflow

### 1. Configure paths

Edit only the path blocks at the start of:

```text
code/train_crack500_18runs.py
code/train_deepcrack_6runs.py
code/score_paper_probmaps.py
```

### 2. Train Crack500

```bash
python code/train_crack500_18runs.py
```

This executes six losses for seeds `0`, `1`, and `2`, and saves the best checkpoint, epoch log, run manifest, and native-resolution validation/test probability maps.

### 3. Train DeepCrack

```bash
python code/train_deepcrack_6runs.py
```

The script preserves the published 300/237 train/test split and creates the frozen 240/60 training/validation split using the pre-specified validation seed.

### 4. Score the saved probability maps

```bash
python code/score_paper_probmaps.py
```

The scoring suite recomputes validation-selected thresholds and evaluates all four threshold conventions, tolerance radii, overlap, boundary, topology, fragmentation, quantification, aggregation, and empty-mask outputs.

## Headline Crack500 results

Validation-selected operating point; mean ± standard deviation across three seeds.

| Loss | Global F1 | Boundary F1 | clDice | Fragmentation ↓ | Area error % ↓ | Skeleton-length error % ↓ |
|---|---:|---:|---:|---:|---:|---:|
| BCE | 0.733 ± 0.003 | 0.382 ± 0.003 | 0.761 ± 0.004 | 2.20 ± 0.06 | 7.3 ± 1.7 | 6.3 ± 1.1 |
| **Focal** | **0.740 ± 0.001** | **0.389 ± 0.003** | 0.764 ± 0.001 | 2.40 ± 0.29 | 7.7 ± 2.1 | 3.3 ± 3.2 |
| BCE + Dice | 0.736 ± 0.001 | 0.386 ± 0.003 | 0.769 ± 0.004 | 2.00 ± 0.06 | 6.1 ± 1.1 | 9.0 ± 1.7 |
| Focal Tversky | 0.733 ± 0.002 | 0.370 ± 0.004 | 0.773 ± 0.004 | 2.07 ± 0.04 | 17.2 ± 4.3 | 7.1 ± 2.9 |
| **Dice + Boundary** | 0.739 ± 0.001 | 0.382 ± 0.004 | 0.773 ± 0.005 | **1.94 ± 0.00** | **2.4 ± 0.6** | 16.1 ± 2.2 |
| **Dice + clDice** | 0.725 ± 0.001 | 0.353 ± 0.011 | **0.801 ± 0.008** | 2.02 ± 0.01 | 19.1 ± 3.4 | 73.6 ± 14.4 |

The 0.0003 mean-F1 separation between Focal and Dice + Boundary is smaller than their seed variation. By contrast, the Dice + clDice topology advantage is substantially larger than seed spread. This is the central metric-family flip.

## Reproducibility notes

- Training runs are seed-controlled but are not claimed to be bitwise deterministic.
- Exact split lists and optional SHA-256 hashes are produced by the training scripts.
- Validation-selected thresholds are canonically recomputed from released validation maps.
- Loss hyperparameters are fixed in advance and never tuned per dataset or observed result.
- Pixel scale is preserved because tolerance radii are resolution-dependent.
- Test-set oracle thresholds are never presented as deployable performance.

## Data and artifact availability

Crack500 and DeepCrack are third-party datasets and are not included. Checkpoints and native-resolution probability maps can be large and should be deposited in a versioned archival release. The frozen protocol, code, scoring manifest, full per-image score archive, and summary outputs are provided here so the study is independently inspectable.

## Citation

```bibtex
@article{ghaffari2026lossmetric,
  title   = {What to Optimize and How to Measure It: Loss--Metric Alignment for Deep Crack Segmentation},
  author  = {Ghaffari, Ehsan and Wang, Kelvin C. P. and Barutha, Philip},
  year    = {2026},
  note    = {Manuscript in preparation},
  url     = {https://github.com/ehsanghaffari/loss-metric-alignment-crack-segmentation}
}
```

## License

Code is released under the [MIT License](LICENSE). Dataset licenses remain with the original providers.

## Contact

**Ehsan Ghaffari**  
Montana State University  
Email: ehsanghaffari@montana.edu
