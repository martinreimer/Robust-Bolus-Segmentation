#!/bin/bash

# Create logs directory
mkdir -p logs

# Define an array of experiments
experiments=(
    "python train.py -e 40 -b 8 -d D:/Martin/thesis/data/processed/dataset_0228_final --loss combined --optimizer adam --scheduler plateau -l 1e-4"
    "python train.py -e 40 -b 8 -d D:/Martin/thesis/data/processed/dataset_0228_final --loss combined --optimizer rmsprop --scheduler plateau -l 1e-4 --filters 32,64,128,256"
    "python train.py -e 40 -b 8 -d D:/Martin/thesis/data/processed/dataset_0228_final --loss combined --optimizer rmsprop --scheduler plateau -l 1e-4 --filters 32,64,128,256,512"
    "python train.py -e 40 -b 8 -d D:/Martin/thesis/data/processed/dataset_0228_final --loss combined --optimizer rmsprop --scheduler plateau -l 1e-4 --filters 16,32,64,128,256,512"


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

    # Optional: wait between experiments
    sleep 5
done

echo "All experiments completed."
