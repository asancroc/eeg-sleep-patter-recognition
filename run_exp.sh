#!/bin/bash
declare -a models=(
    experiment_1
    experiment_2
    experiment_3
    experiment_4
)

for model in "${models[@]}"
do
    echo "Experiment: $model"
    python train.py --experiment "$model"

done