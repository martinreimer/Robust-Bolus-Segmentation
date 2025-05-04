#!/usr/bin/env python3
import subprocess
import sys
import time
import re
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# Path to your original runs summary CSV
SUMMARY_CSV = Path(r"D:\Martin\thesis\training_runs\experiments_summary.csv")

# Directory where your UNet runs live
RUNS_DIR = Path(r"D:\Martin\thesis\training_runs\U-Net\runs")

# Where to drop all predict.py outputs
OUTPUT_BASE = Path(r"D:\Martin\thesis\test_runs")

# Use the same Python interpreter for subprocesses (your venv)
PREDICT_PYTHON = sys.executable
PREDICT_SCRIPT = Path(__file__).parent / "predict.py"

# Constant dataset locations & CSV‐mapping
VAL_DATA_DIR      = Path(r"D:/Martin/thesis/data/processed/dataset_labelbox_export_test_2504_test_final_roi_crop/val")
TEST_DATA_DIR     = Path(r"D:/Martin/thesis/data/processed/dataset_labelbox_export_test_2504_test_final_roi_crop/test")
DATA_OVERVIEW_CSV = Path(r"D:/Martin/thesis/data/processed/dataset_labelbox_export_test_2504_test_final_roi_crop/data_overview.csv")

# Inference parameters
FPS       = 10
THRESHOLD = 0.8
COMMON_FLAGS = [
    "-v",
    "-t", str(THRESHOLD),
    "--save-metrics-csv",
    "--save-video-mp4s",
    "--fps", str(FPS),
    "--plot-metrics"
]

# ─── UTILS ─────────────────────────────────────────────────────────────────────

def pick_best_epoch_rows(df, run_col="run_name", epoch_col="best_epoch"):
    """Keep only the row with the highest epoch_col per run_col."""
    idx = df.groupby(run_col)[epoch_col].idxmax()
    return df.loc[idx].reset_index(drop=True)

def find_checkpoint_for_epoch(run_name, target_epoch):
    """
    In RUNS_DIR / run_name / checkpoints, find all
    'checkpoint_epoch{N}.pth' and return the one
    whose N is closest to target_epoch.
    """
    ckpt_dir = RUNS_DIR / run_name / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"{ckpt_dir} not found")
    candidates = list(ckpt_dir.glob("checkpoint_epoch*.pth"))
    best = None
    best_diff = None
    pat = re.compile(r"checkpoint_epoch(\d+)\.pth$")
    for p in candidates:
        m = pat.search(p.name)
        if not m:
            continue
        ep = int(m.group(1))
        diff = abs(ep - target_epoch)
        if best is None or diff < best_diff:
            best, best_diff = p, diff
    if best is None:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return best

def run_predict(run_name, model_path, split):
    """
    Calls predict.py for one run and one split ("val" or "test").
    Returns the Path to the metrics CSV it generates.
    """
    data_dir = VAL_DATA_DIR if split == "val" else TEST_DATA_DIR
    cmd = [
        str(PREDICT_PYTHON),
        str(PREDICT_SCRIPT),
        "--test-name", run_name,
        "--model-path", str(model_path),
        "--output-dir", str(OUTPUT_BASE),
        "--data-dir", str(data_dir),
        "--csv-path", str(DATA_OVERVIEW_CSV),
        "--dataset-split", split,
        *COMMON_FLAGS
    ]
    print("Running inference:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # location of the metrics CSV
    metrics_csv = (
        OUTPUT_BASE /
        f"{run_name}" /
        split /
        f"{run_name}_metrics.csv"
    )

    # wait a bit to ensure file is written
    time.sleep(2)

    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Expected metrics not at {metrics_csv}")
    return metrics_csv

def load_and_annotate_metrics(csv_path, run_name, cli, epoch, model_path, split):
    """
    Load one metrics CSV, add metadata columns, and return a DataFrame.
    """
    df = pd.read_csv(csv_path)
    df["run_name"]   = run_name
    df["cli"]        = cli
    df["epoch"]      = epoch
    df["model_path"] = str(model_path)
    df["split"]      = split
    return df

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # Read and filter summary CSV
    df0 = pd.read_csv(SUMMARY_CSV)
    df0 = pick_best_epoch_rows(df0, run_col="run_name", epoch_col="best_epoch")

    # Will collect per-run summary for the global overview
    summary_data = {}

    # Loop over runs with progress bar
    records = df0.to_dict("records")
    for row in tqdm(records, desc="Evaluating runs", unit="run"):
        run_name = row["run_name"]
        epoch    = int(row["best_epoch"])
        cli      = row.get("cli", "")

        # 1) Find the best matching checkpoint
        ckpt_path = find_checkpoint_for_epoch(run_name, epoch)
        tqdm.write(f"{run_name}: using checkpoint {ckpt_path.name}")

        # 2) Run inference on val and test
        val_csv  = run_predict(run_name, ckpt_path, "val")
        test_csv = run_predict(run_name, ckpt_path, "test")

        # 3) Load and annotate
        df_val  = load_and_annotate_metrics(val_csv,  run_name, cli, epoch, ckpt_path, "val")
        df_test = load_and_annotate_metrics(test_csv, run_name, cli, epoch, ckpt_path, "test")

        # 4) Combine and write per-run detailed summary
        df_run = pd.concat([df_val, df_test], ignore_index=True)
        out_csv = OUTPUT_BASE / f"{run_name}_inference_summary.csv"
        df_run.to_csv(out_csv, index=False)
        tqdm.write(f"Wrote per-run summary to {out_csv}")

        # 5) Compute split-level means for summary_experiment_csv
        metric_cols = [c for c in df_run.columns
                       if c not in ("Video","run_name","cli","epoch","model_path","split")]
        means = {}
        for split_label in ("val", "test"):
            m = df_run[df_run["split"] == split_label][metric_cols].mean()
            means[f"{split_label} Dice Mean"]    = m.get("Dice Mean", np.nan)
            means[f"{split_label} specificity"]  = m.get("specificity", np.nan)
            means[f"{split_label} recall"]       = m.get("recall", np.nan)
            means[f"{split_label} precision"]    = m.get("precision", np.nan)
            means[f"{split_label} f1"]           = m.get("f1", np.nan)
            means[f"{split_label} iou"]          = m.get("iou", np.nan)

        # Store into summary_data
        summary_data[run_name] = {
            "cli": cli,
            **means
        }

    # After all runs, write the global summary_experiment_csv_{date}.csv
    summary_df = pd.DataFrame(summary_data)
    # Define row order
    row_order = ["cli"]
    for split_label in ("val", "test"):
        row_order += [
            f"{split_label} Dice Mean",
            f"{split_label} specificity",
            f"{split_label} recall",
            f"{split_label} precision",
            f"{split_label} f1",
            f"{split_label} iou",
        ]
    summary_df = summary_df.reindex(row_order)

    date_str = datetime.date.today().strftime("%Y%m%d")
    summary_out = OUTPUT_BASE / f"summary_experiment_csv_{date_str}.csv"
    summary_df.to_csv(summary_out)
    print(f"Wrote global summary to {summary_out}")

if __name__ == "__main__":
    main()
