#!/usr/bin/env python3
"""
Randomized sweep runner with valid encoder-depth ↔ decoder-channels mapping.
Prints errors to CLI for failed runs. Logs experiment stdout separately.
"""

import sys
import subprocess
import random
import time
from pathlib import Path
from datetime import datetime
import itertools

# Setup
SEED = 42
N_SAMPLES = 160

random.seed(SEED)

PREDICT_PYTHON = sys.executable
PREDICT_SCRIPT = Path(__file__).parent / "train.py"
LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)

DATASET = Path("D:/Martin/thesis/data/processed/dataset_normal_0514_final_roi_crop")

COMMON_ARGS = [
    "-b", "6", #"8",
    "--optimizer", "adamax",
    "--scheduler", "plateau",
    "--mask-suffix", "_bolus",
    "-d", str(DATASET),
]

# Hyperparameter space
models = ["Unet"]#, "UNetPlusPlus"]
losses = ["dice", "focal", "tversky"] #bce_dice
backbones = ["mobilenet_v2", "inceptionresnetv2"]
use_norm = ["batchnorm"]
lrs = ["1e-4", "5e-4", "1e-3"]#["1e-4", "5e-4", "1e-3"]
depths = [3, 4, 5]
base_channels = [16, 32, 64, 64, 128]
adam_weight_decay = [1e-4, 1e-5, 1e-6]

def make_decoder_channels(depth, base):
    return [base * (2 ** i) for i in reversed(range(depth))]

# Sample combinations
all_combos = list(itertools.product(models, losses, backbones, lrs, depths, base_channels, use_norm, adam_weight_decay))
sampled_combos = random.sample(all_combos, k=N_SAMPLES)

any_failed = False
counter = 0
for i, (model, loss, backbone, lr, depth, base, norm, adam_weight_decay) in enumerate(sampled_combos):
    counter += 1
    if counter < 23:
        continue
    decoder_channels = make_decoder_channels(depth, base)


    cmd = [
        PREDICT_PYTHON, str(PREDICT_SCRIPT),
        "--epochs", "40",
        "-l", lr,
        "--loss", loss,
        *COMMON_ARGS,
        "--model-source", "smp",
        "--smp-model", model,
        "--encoder-name", backbone,
        "--encoder-weights", "imagenet",
        "--encoder-depth", str(depth),
        "--decoder-interpolation", "nearest",
        "--decoder-use-norm", norm,
        "--decoder-channels", ",".join(map(str, decoder_channels)),
    ]
    # Conditionally add loss-specific hyperparameters
    if loss in ["focal", "bce_dice"]:
        focal_alpha = random.choice([0.15, 0.25, 0.3, 0.5, 0.75])
        focal_gamma = random.choice([2.0, 4.0, 8.0])
        cmd += ["--focal-alpha", str(focal_alpha), "--focal-gamma", str(focal_gamma)]

    if loss == "tversky":
        tversky_alpha = random.choice([0.4, 0.5, 0.6, 0.7, 0.75, 0.8])
        cmd += ["--tversky-alpha", str(tversky_alpha)]

    # Optimizer-specific hyperparams
    cmd += ["--adam-weight-decay", str(adam_weight_decay)]  # Always include if you're always using Adam/Adamax

    base_name = f"{i:02d}__{loss}__{backbone}__lr{lr}__d{depth}__ch{base}"

    print("------------------------------------------------------------------------------")
    print("RUNNING:", " ".join(cmd))
    print("------------------------------------------------------------------------------")

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return_code = process.wait()

    if return_code != 0:
        any_failed = True
        print(f"❌ FAILED: {base_name} — Exit code {return_code}")

    time.sleep(2)
