#!/usr/bin/env python
"""Create reviewer-ready per-image geometric-error summaries for the submitted 72-run study.

This script uses the already-generated original offline scorer output:
    OFFLINE_SCORE_OUTPUT/per_image_advanced_val_selected.csv

It does not retrain, rethreshold, or rescore probability maps. It derives per-image
absolute relative area and skeleton-length errors from the saved per-image counts.

To avoid treating the same image under multiple seeds as independent observations,
reportable image-level distributions are formed after averaging each image's error
across the three matched seeds for a given dataset/loss. Seed-specific distributions
are also written for auditability.
"""
from __future__ import annotations

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

PAPER_ROOT = Path(os.environ.get("CRACKSEG_ROOT", Path.cwd())).expanduser().resolve()
OFFLINE_SCORE_ROOT = Path(os.environ.get("CRACKSEG_OFFLINE_SCORE_OUTPUT", PAPER_ROOT / "OFFLINE_SCORE_OUTPUT")).expanduser().resolve()
OUTPUT_BASE = Path(os.environ.get("CRACKSEG_OUTPUT", PAPER_ROOT / "Output")).expanduser().resolve()
SOURCE = OFFLINE_SCORE_ROOT / "per_image_advanced_val_selected.csv"
ORIGINAL_SUMMARY = OFFLINE_SCORE_ROOT / "summary_by_dataset_loss_val_selected.csv"
OUT = OUTPUT_BASE / "Extended_Analysis" / "Original_All_Dataset_Geometry"

EXPECTED_DATASETS = ["Crack500", "DeepCrack", "CFD", "CrackTree260"]
LOSS_ORDER = ["bce", "focal", "bce_dice", "focal_tversky", "dice_boundary", "dice_cldice"]
EXPECTED_SEEDS = [0, 1, 2]


def abs_rel_pct(pred: pd.Series, gt: pd.Series) -> pd.Series:
    pred = pd.to_numeric(pred, errors="coerce")
    gt = pd.to_numeric(gt, errors="coerce")
    out = (pred.sub(gt).abs().div(gt) * 100.0).where(gt > 0)
    return out


def distribution_stats(values: pd.Series, prefix: str) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return {f"{prefix}_{k}": np.nan for k in ["n", "mean", "std", "median", "q25", "q75", "q90", "q95"]}
    return {
        f"{prefix}_n": int(x.size),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_q25": float(np.quantile(x, 0.25)),
        f"{prefix}_q75": float(np.quantile(x, 0.75)),
        f"{prefix}_q90": float(np.quantile(x, 0.90)),
        f"{prefix}_q95": float(np.quantile(x, 0.95)),
    }


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing original per-image scorer output:\n{SOURCE}\n"
            "Run/copy the submitted v2 OFFLINE_SCORE_OUTPUT before this evaluation."
        )
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE)
    required = {
        "dataset", "loss", "seed", "image", "area_pred", "area_gt",
        "skeleton_length_pred", "skeleton_length_gt",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {SOURCE.name}: {missing}")

    datasets = sorted(df["dataset"].unique())
    if sorted(EXPECTED_DATASETS) != datasets:
        raise RuntimeError(f"Dataset mismatch. Expected {sorted(EXPECTED_DATASETS)}, found {datasets}")
    for ds in EXPECTED_DATASETS:
        for loss in LOSS_ORDER:
            seeds = sorted(df[(df.dataset == ds) & (df.loss == loss)].seed.unique().tolist())
            if seeds != EXPECTED_SEEDS:
                raise RuntimeError(f"{ds}/{loss}: expected seeds {EXPECTED_SEEDS}, found {seeds}")

    df["area_abs_error_pct"] = abs_rel_pct(df["area_pred"], df["area_gt"])
    df["skeleton_length_abs_error_pct"] = abs_rel_pct(
        df["skeleton_length_pred"], df["skeleton_length_gt"]
    )
    df.to_csv(OUT / "Per_Image_Geometric_Errors_All_72_Runs.csv", index=False)

    run_rows = []
    for (dataset, loss, seed), g in df.groupby(["dataset", "loss", "seed"], sort=False):
        row = {"dataset": dataset, "loss": loss, "seed": int(seed), "n_images": int(len(g))}
        row.update(distribution_stats(g["area_abs_error_pct"], "area_abs_error_pct"))
        row.update(distribution_stats(g["skeleton_length_abs_error_pct"], "skeleton_length_abs_error_pct"))
        run_rows.append(row)
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(OUT / "Run_Level_Per_Image_Geometry_Distributions.csv", index=False)

    image_mean = (
        df.groupby(["dataset", "loss", "image"], as_index=False)
        .agg(
            area_abs_error_pct_seedmean=("area_abs_error_pct", "mean"),
            skeleton_length_abs_error_pct_seedmean=("skeleton_length_abs_error_pct", "mean"),
            seeds_present=("seed", "nunique"),
        )
    )
    if not (image_mean["seeds_present"] == 3).all():
        bad = image_mean[image_mean["seeds_present"] != 3].head()
        raise RuntimeError(f"Some image/loss groups do not have all three seeds:\n{bad}")
    image_mean.to_csv(OUT / "Per_Image_SeedMean_Geometric_Errors.csv", index=False)

    report_rows = []
    for (dataset, loss), g in image_mean.groupby(["dataset", "loss"], sort=False):
        row = {"dataset": dataset, "loss": loss, "n_images": int(len(g)), "n_seeds": 3}
        row.update(distribution_stats(g["area_abs_error_pct_seedmean"], "area_abs_error_pct_per_image_seedmean"))
        row.update(distribution_stats(
            g["skeleton_length_abs_error_pct_seedmean"],
            "skeleton_length_abs_error_pct_per_image_seedmean",
        ))
        report_rows.append(row)
    report_df = pd.DataFrame(report_rows)

    if ORIGINAL_SUMMARY.exists():
        old = pd.read_csv(ORIGINAL_SUMMARY)
        keep = [
            "dataset", "loss", "area_error_pct_dataset_mean", "area_error_pct_dataset_std",
            "skeleton_length_error_pct_dataset_mean", "skeleton_length_error_pct_dataset_std",
        ]
        if all(c in old.columns for c in keep):
            report_df = report_df.merge(old[keep], on=["dataset", "loss"], how="left")

    report_df["dataset"] = pd.Categorical(report_df["dataset"], EXPECTED_DATASETS, ordered=True)
    report_df["loss"] = pd.Categorical(report_df["loss"], LOSS_ORDER, ordered=True)
    report_df = report_df.sort_values(["dataset", "loss"]).reset_index(drop=True)
    report_df.to_csv(OUT / "Supplementary_Table_All_Datasets_Per_Image_Geometric_Errors.csv", index=False)

    manifest = {
        "source": str(SOURCE),
        "source_is_submitted_study_output": True,
        "threshold_policy": "uses the original validation-selected-threshold per-image counts; no threshold reselection",
        "area_error_per_image": "100*abs(area_pred-area_gt)/area_gt",
        "skeleton_length_error_per_image": "100*abs(length_pred-length_gt)/length_gt",
        "reportable_distribution_unit": "image after averaging matched image error across three seeds",
        "gt_zero_handling": "undefined relative error stored as NaN and excluded from distribution summaries",
        "physical_units_claimed": False,
    }
    (OUT / "geometry_revision_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()
