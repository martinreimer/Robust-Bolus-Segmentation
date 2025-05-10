#!/usr/bin/env python3
"""
Replaces the original bash loop for running experiments.
Uses the same Python interpreter (venv) for subprocesses via sys.executable.
"""
import sys
import subprocess
import itertools
import time
from pathlib import Path
from datetime import datetime

# Use the same Python interpreter (virtualenv) for the training script
PREDICT_PYTHON = sys.executable
PREDICT_SCRIPT = Path(__file__).parent / "train.py"

# Directory to stash per-run logs
LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)

# Your dataset path
DATASET = Path("D:/Martin/thesis/data/processed/dataset_labelbox_export_test_2504_test_final_roi_crop")

# Base arguments common to all runs
COMMON_ARGS = [
    "-b", "8",
    "--optimizer", "adamax",
    "--scheduler", "plateau",
    "--mask-suffix", "_bolus",
    "-d", str(DATASET),
    "--bilinear",
    "--amp"
]

# Hyperparameter grids
batchnorms    = ["instance"]#"None", "Batch",
attentions = [False, True]
lrs       = ["1e-3", "1e-4"]

any_failed = False

for norm, attention, lr in itertools.product(batchnorms, attentions, lrs):
    # Build command
    if attention:
        cmd = [
            PREDICT_PYTHON, str(PREDICT_SCRIPT),
            "--epochs", "30",
            "-l", lr,
            "--loss", "dice",
            *COMMON_ARGS,
            "--model-source", "custom",

            "--custom-use-norm", norm,
            "--use-attention",
        ]
    else:
        cmd = [
            PREDICT_PYTHON, str(PREDICT_SCRIPT),
            "--epochs", "30",
            "-l", lr,
            "--loss", "dice",
            *COMMON_ARGS,
            "--model-source", "custom",

            "--custom-use-norm", norm,
            "--use-attention",
        ]
    print("RUNNING:", " ".join(cmd))
    print("------------------------------------------------------------------------------")

    # Run subprocess, redirect both stdout and stderr to the log file
    process = subprocess.Popen(cmd, stderr=subprocess.STDOUT)
    return_code = process.wait()
    time.sleep(2)

# Summary and exit code
if any_failed:
    print(f"Some runs failed; see {LOGDIR} for details.")
    sys.exit(1)
else:
    print(f"All experiments done. Logs are in {LOGDIR}/")
    sys.exit(0)
