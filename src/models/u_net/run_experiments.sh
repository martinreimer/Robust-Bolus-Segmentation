#!/bin/bash

# Create logs directory
mkdir -p logs

# Define an array of experiments
experiments=(
    "python train.py --epochs 60 -b 8 -l 1e-5 -d D:\Martin\thesis\data\processed\dataset_0328_final --filters 32,64,128,256,512 --loss combined  --optimizer rmsprop --scheduler plateau --mask-suffix _bolus"
    "python train.py --epochs 60 -b 8 -l 1e-5 -d D:\Martin\thesis\data\processed\dataset_0328_final --filters 32,64,128,256,512 --loss combined  --optimizer adam --scheduler plateau --mask-suffix _bolus

)

# Iterate through experiments
for i in "${!experiments[@]}"; do
    echo "------------------------------------------------------------"
    echo "Starting Experiment $((i+1)):"
    echo "${experiments[i]}"
    echo "------------------------------------------------------------"

    # Run the experiment and log output
    ${experiments[i]} | tee "logs/experiment_$((i+1)).log"

    echo "------------------------------------------------------------"
    echo "Finished Experiment $((i+1))"
    echo "------------------------------------------------------------"

    # Optional: wait between experiment
    sleep 5
done

echo "All experiments completed."
