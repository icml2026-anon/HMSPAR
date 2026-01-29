#!/bin/bash

DATASET=""
MODEL=""


while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            MODEL="$1"
            shift
            ;;
    esac
done

run_model_on_dataset() {
    local model=$1
    local dataset=$2
    local script="${model}_baseline.py"
    
    if [ "$dataset" == "merchant" ]; then
        for i in 0 1 2 3; do
            echo "--- $model on Industry-$i ---"
            python $script --dataset merchant --industry "Industry-$i"
        done
    else
        echo "--- $model on $dataset ---"
        python $script --dataset $dataset
    fi
}

run_model() {
    local model=$1
    local script="${model}_baseline.py"
    
    echo "========================================"
    echo "Running $model"
    echo "========================================"
    
    if [ -n "$DATASET" ]; then
        run_model_on_dataset $model $DATASET
    else
        for i in 0 1 2 3; do
            echo "--- $model on Industry-$i ---"
            python $script --dataset merchant --industry "Industry-$i"
        done
        
        for dataset in cdnow retail instacart; do
            echo "--- $model on $dataset ---"
            python $script --dataset $dataset
        done
    fi
}

cd "$(dirname "$0")"

if [ -z "$MODEL" ]; then
    for model in xgboost catboost lstm lightgbm randomforest timemoe units vqshape mptsnet; do
        run_model $model
    done
else
    run_model $MODEL
fi

echo ""
echo "========================================"
echo "All baseline experiments completed!"
echo "========================================"
