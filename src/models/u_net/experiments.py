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
]

# Hyperparameter grids
models    = ["UNetPlusPlus"]#"Unet", "Segformer"
losses    = ["dice", "focal", "tversky"]
backbones= ["mobilenet_v2", "inceptionresnetv2"]
lrs       = ["1e-3", "1e-4", "1e-5"]
depths    = [5]

any_failed = False

for model, loss, backbone, lr, depth in itertools.product(models, losses, backbones, lrs, depths):
    # Build command
    cmd = [
        PREDICT_PYTHON, str(PREDICT_SCRIPT),
        "--epochs", "60",
        "-l", lr,
        "--loss", loss,
        *COMMON_ARGS,
        "--model-source", "smp",
        "--smp-model", model,
        "--encoder-name", backbone,
        "--encoder-weights", "imagenet",
        "--encoder-depth", str(depth),
        "--decoder-interpolation", "nearest",
        "--decoder-use-norm", "batchnorm",
    ]

    # Log filenames
    base = f"{loss}__{backbone}__lr{lr}__d{depth}"
    log = LOGDIR / f"{base}.log"
    err = LOGDIR / f"{base}.error"

    print("------------------------------------------------------------------------------")
    print("RUNNING:", " ".join(cmd))
    print(" → log →", log)
    print("------------------------------------------------------------------------------")

    # Run subprocess, redirect both stdout and stderr to the log file
    with log.open("w") as lf:
        process = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        return_code = process.wait()

    # Handle success/failure
    if return_code != 0:
        any_failed = True
        with err.open("w") as ef:
            ef.write(f"[{datetime.now().strftime('%F %T')}] EXIT {return_code}\n")
        print(f"!!! Failed — see {err}")
    else:
        if err.exists():
            err.unlink()

    time.sleep(2)

# Summary and exit code
if any_failed:
    print(f"Some runs failed; see {LOGDIR} for details.")
    sys.exit(1)
else:
    print(f"All experiments done. Logs are in {LOGDIR}/")
    sys.exit(0)
