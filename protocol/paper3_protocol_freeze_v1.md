# Paper 3 — Experimental Protocol Freeze (v1.0)

**Study:** What to Optimize and How to Measure It: Loss–Metric Alignment for Deep Crack Segmentation (controlled demonstration component)
**Status:** DRAFT FOR SIGN-OFF. Becomes FROZEN when both signatures below are present.
**Date prepared:** July 5, 2026

**Binding clause.** No item in this document may change after the first training run. If a change becomes unavoidable, it is recorded in the Deviation Log (Section 10) with date, reason, and impact, and disclosed in the manuscript as a deviation from the frozen protocol.

---

## 1. Purpose and terminology

**Purpose statement (verbatim in the manuscript):** the goal is not to find the best loss; it is to show, on identical predictions, that the apparent best loss changes with the metric family, the tolerance radius, the threshold rule, and the aggregation convention, and that some of these flips exceed seed noise.

**Terminology rule:** the study is a "controlled demonstration" or "protocol study" throughout. "Benchmark" is reserved for datasets and prior leaderboard-style studies.

---

## 2. Fixed training protocol

| Item | Frozen value |
|---|---|
| Architecture | U-Net, ResNet-34 encoder, ImageNet initialization; single backbone (controlled variable, not a subject) |
| Task / output | Binary segmentation; single-channel logits with sigmoid |
| Training input | Random 448×448 crops from native-resolution images; constant-zero padding when an image is smaller; no resizing anywhere in the pipeline |
| Augmentation | Horizontal flip p=0.5; vertical flip p=0.5; rotations of k·90°; brightness and contrast jitter, factors U(0.8, 1.2); ImageNet mean/std normalization |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-4 |
| Schedule | Cosine annealing, T_max = 100 epochs |
| Epoch budget | 100 maximum; early stopping with patience 20 |
| Batch size / workers | 8 / 4 |
| Model selection | Best epoch by validation-selected dataset-global F1: for each epoch, F1 is maximized over the frozen 0.01–0.99 threshold grid on the validation set (threshold-free selection; a fixed-0.5 rule would bias checkpoints toward losses calibrated near 0.5). The argmax threshold at the best epoch is recorded and is the run's deployable validation-selected threshold. Ties at four decimals: earliest epoch; threshold closest to 0.5, then the lower threshold. Identical rule for every run |
| Evaluation input | Full images at native resolution; reflect padding to the nearest multiple of 32; predictions cropped back to native size before any use; tolerance radii defined at native resolution |
| Reproducibility | Seeds pinned (below); per-run DataLoader generator; runs are seed-controlled, not bitwise deterministic (cuDNN benchmark mode); stated in manuscript and recorded per run |
| Saved artifacts per run | Best checkpoint; per-epoch log; manifest with full configuration and environment versions; float32 probability maps for every validation and test image |
| Seeds | Crack500: {0, 1, 2}. DeepCrack: {0} |

---

## 3. Losses and frozen hyperparameters

| # | Loss | Frozen hyperparameters | Provenance |
|---|---|---|---|
| 1 | BCE | none | standard baseline |
| 2 | Focal | γ = 2, unweighted | originating-paper default |
| 3 | BCE + Dice | Pixel-averaged BCE plus scalar soft-Dice loss, combined as Loss = BCE + Dice with no additional weighting | field-standard compound |
| 4 | Focal Tversky | α = 0.3 (FP), β = 0.7 (FN), exponent 0.75, i.e. (1 − TI)^(1/γ) with γ = 4/3 | originating-paper defaults |
| 5 | Dice + Boundary | Kervadec-style signed distance map, unnormalized (original convention); loss = a·Dice + (1 − a)·Boundary with a = max(0.01, 1 − 0.01·epoch); SDF set to zero for empty or full masks | originating-paper defaults; the known conservative tendency of the unnormalized SDF on thin structures is accepted and, if observed, reported rather than repaired |
| 6 | Dice + clDice | Loss = 0.7·Dice + 0.3·clDice; weight 0.3 is a PRE-SPECIFIED PROTOCOL VALUE (the originating paper studies α in [0.1, 0.5] with no single canonical default); soft skeletonization, 5 iterations, computed on probability maps (loss side) | mechanism from originating paper; weight pre-specified here |

No loss hyperparameter is tuned per dataset, per seed, or after any result is seen.

---

## 4. Datasets and splits

| Dataset | Train | Val | Test | Rules |
|---|---|---|---|---|
| Crack500 (primary) | 1,896 | 348 | 1,124 | Published patch-level partition; counts hard-enforced before training; SHA-256 file lists released |
| DeepCrack (secondary) | 240 | 60 | 237 | Published 300/237 train/test preserved; validation = fixed 20% subset (60 images) drawn once from the 300 training images with a fixed, reported seed; image list released; test set untouched |

No re-splitting under any circumstance. All seed-variance analysis on Crack500.

---

## 5. Scoring protocol (offline, on saved probability maps, identical for every run)

**5.1 Threshold conventions.** Search grid frozen at 0.01 to 0.99 in steps of 0.01. Binarization comparator frozen: a pixel is predicted positive iff p ≥ t.
(a) Fixed 0.5 [deployable]. (b) Validation-selected global threshold: argmax of dataset-global F1 on the run's own validation maps; one value per run; applied unchanged to test [deployable]. This threshold coincides with the model-selection argmax (Section 2) and is canonically recomputed offline from the released validation maps. (c) ODS: single test-optimal threshold [oracle upper bound]. (d) OIS: per-image optimal threshold [oracle upper bound]. ODS and OIS thresholds are selected to maximize F1/Dice only; any other metric reported at an oracle threshold uses that F1-optimal threshold and never receives its own separately optimized threshold. Tie rule everywhere: scores equal to four decimal places resolve to the threshold closest to 0.5, then the lower threshold. Every reported number is labeled deployable or oracle.

**5.2 Overlap.** IoU, F1/Dice, Precision, Recall, under each threshold convention.

**5.3 Relaxed (tolerance) F1.** Radii r ∈ {0, 2, 3, 4, 5} px at native resolution. Matching rule (frozen): bidirectional distance-transform relaxation. A predicted positive pixel counts as TP for relaxed precision if its Euclidean distance to the nearest ground-truth positive pixel is ≤ r; a ground-truth positive pixel counts as matched for relaxed recall if its distance to the nearest predicted positive pixel is ≤ r; relaxed F1 is their harmonic mean. r = 0 reduces to strict pixel F1.

**5.4 Boundary.** Boundary F1 (BF-score) on boundary pixels of binarized masks with match tolerance θ = 2 px, fixed in pixels at native resolution (the diagonal-relative default of the original BF formulation is deliberately not used, to preserve pixel scale across images and datasets; stated in the manuscript). HD95 in supplementary only.

**5.5 Topology.** clDice metric computed on binarized masks using scikit-image skeletonization (metric side; the soft skeleton is loss-side only, per Section 3). Optional companion: fragmentation error |#components_pred − #components_gt| with 8-connectivity.

**5.6 Aggregation.** Both reported for IoU and F1: (i) per-image mean of per-image scores, with Section 6 rules applied; (ii) dataset-global pooling of TP/FP/FN over all pixels of all images.

**5.7 Quantification.** Skeleton-length error and area error as dataset-level relative errors, |Σ_pred − Σ_gt| / Σ_gt × 100, with length = skeleton pixel count after skeletonization and area = positive pixel count; computed at the validation-selected global threshold (primary operating point, 5.8). Per-image distributions in supplementary. Width error only if time allows; otherwise treated conceptually in the metrics section.

**5.8 Primary operating point.** Relaxed F1, Boundary F1, clDice, fragmentation, and quantification metrics are reported primarily at the validation-selected global threshold [deployable]. Fixed-0.5 and oracle (ODS/OIS) variants of these metrics appear only as threshold-sensitivity analyses or in supplementary material. Overlap metrics (5.2) remain reported under all four conventions, since the threshold-convention comparison is itself a study axis.

---

## 6. Empty-mask rules (per metric; case counts always reported; nothing silently discarded)

Cases per test image at the operative threshold: **A** both masks empty; **B** ground truth empty, prediction non-empty; **C** ground truth non-empty, prediction empty; **D** both non-empty (normal scoring).

| Metric (per-image) | Case A | Case B | Case C |
|---|---|---|---|
| IoU, F1, Precision, Recall, relaxed F1, Boundary F1, clDice | 1.0 | 0.0 | 0.0 |
| HD95 (supplementary) | excluded, counted | excluded, counted | excluded, counted |
| Fragmentation error | defined naturally (value = 0 or k); never excluded | same | same |

Dataset-global pooled metrics are unaffected (pixel counts pool over all images). Dataset-level quantification errors are unaffected by construction; the supplementary per-image variants exclude ground-truth-empty images with counts. Note: Crack500 test patches were selected to contain cracks, so cases A and B are expected to be rare; empty predictions (case C) can occur and their counts are reported per loss and threshold convention.

---

## 7. Run matrix and pre-specified contingencies

Crack500: 6 losses × 3 seeds = 18 runs. DeepCrack: 6 losses × 1 seed = 6 runs. **Total: 24.**

Pre-specified upgrade: DeepCrack to 3 seeds (36 total) only if compute allows AND the decision is made before any test-set scoring has occurred. Pre-specified fallback: Crack500-only, with an explicit limitation sentence that cross-regime generalization of the flips is untested. Neither contingency may be invoked in response to results.

---

## 8. Honesty and release commitments

Hypotheses live in the framework section of the manuscript; the results section reports observed outcomes, including null or partial flips. Released with the paper: training code, configurations, split lists with SHA-256 hashes, all validation and test probability maps (float32), scoring scripts, and this protocol document.

---

## 9. Sign-off

| Role | Name | Signature | Date |
|---|---|---|---|
| Student | | | |
| Supervisor | | | |

Upon both signatures, this document is FROZEN as v1.0.

---

## 10. Deviation log (append-only; empty at freeze)

| Date | Item | Change | Reason | Manuscript disclosure |
|---|---|---|---|---|
| | | | | |
