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
losses=(dice) #bce focal tversky
backbones=(none inceptionresnetv2 inceptionv4 resnet34 mobilenet_v2)
lrs=(1e-3 1e-4)
depths=(3 4 5)

for loss in "${losses[@]}"; do
  for backbone in "${backbones[@]}"; do
    for lr in "${lrs[@]}"; do
      for depth in "${depths[@]}"; do

        # start your cmd array
        cmd=(python train.py
             --epochs 25
             -l "${lr}"
             --loss "${loss}"
        )

        # splice in all the common args, correctly tokenized
        cmd+=( "${COMMON_ARGS[@]}" )

        if [[ "${backbone}" != "none" ]]; then
          cmd+=( --model-source smp
                 --encoder-name "${backbone}"
                 --encoder-weights imagenet
                 --encoder-depth "${depth}"
                 --decoder-interpolation nearest
                 --decoder-use-norm batchnorm
               )
        else
          cmd+=( --model-source custom )
        fi

        # logfile names
        base="${loss}__${backbone}__lr${lr}__d${depth}"
        log="${LOGDIR}/${base}.log"
        err="${LOGDIR}/${base}.error"

        echo "------------------------------------------------------------------------------"
        echo "RUNNING: ${cmd[*]}"
        echo " → log → ${log}"
        echo "------------------------------------------------------------------------------"

        # disable exit-on-error so we can capture exit code
        set +e
        "${cmd[@]}" 2>&1 | tee "${log}"
        exit_code=${PIPESTATUS[0]}
        set -e

        if [ "${exit_code}" -ne 0 ]; then
          echo "[$(date '+%F %T')] EXIT ${exit_code}" > "${err}"
          echo "!!! Failed — see ${err}"
        else
          rm -f "${err}"
        fi

        sleep 2

      done
    done
  done
done

echo "All experiments done. Logs are in ${LOGDIR}/"
