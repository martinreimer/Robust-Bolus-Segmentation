#!/usr/bin/env bash
set -uo pipefail   # leave off -e so we can catch per-run failures

# Where to stash per‐run logs
LOGDIR=logs
mkdir -p "${LOGDIR}"

# Your dataset path
DATASET="D:/Martin/thesis/data/processed/dataset_labelbox_export_test_2504_test_final_roi_crop"

# Base arguments common to all runs, as an ARRAY, not a quoted string
COMMON_ARGS=(
  -b 8
  --optimizer adamax
  --scheduler plateau
  --mask-suffix _bolus
  -d "${DATASET}"
)

# Hyperparameter grids
models=(Unet UNetPlusPlus Segformer)
losses=(dice focal tversky) #bce focal tversky
backbones=(mobilenet_v2 inceptionresnetv2) # inceptionresnetv2 inceptionv4 resnet34 mobilenet_v2
lrs=(1e-3 1e-4)# 1e-4)
depths=(5)
for model in "${models[@]}"; do
  for loss in "${losses[@]}"; do
    for backbone in "${backbones[@]}"; do
      for lr in "${lrs[@]}"; do
        for depth in "${depths[@]}"; do

          # start your cmd array
          cmd=(python train.py
               --epochs 40
               -l "${lr}"
               --loss "${loss}"
          )

          # splice in all the common args, correctly tokenized
          cmd+=( "${COMMON_ARGS[@]}" )

          cmd+=( --model-source smp
                 --smp-model "${model}"
                 --encoder-name "${backbone}"
                 --encoder-weights imagenet
                 --encoder-depth "${depth}"
                 --decoder-interpolation nearest
                 --decoder-use-norm batchnorm
               )
          # logfile names
          base="${loss}__${backbone}__lr${lr}__d${depth}"
          log="${LOGDIR}/${base}.log"
          err="${LOGDIR}/${base}.error"

          echo "------------------------------------------------------------------------------"
          echo "RUNNING: ${cmd[*]}"
          echo " → log → ${log}"
          echo "------------------------------------------------------------------------------"

          sleep 2

        done
      done
    done
  done
done

echo "All experiments done. Logs are in ${LOGDIR}/"
