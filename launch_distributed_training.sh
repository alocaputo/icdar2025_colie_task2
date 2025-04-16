#!/bin/bash
# filepath: /home/alocaputo/alocaputo/icdar2025/launch_distributed_training.sh

# Number of folds to train
K=4

# Number of epochs
EPOCHS=10

# Data path
DATA_PATH="./data/ensemble_data.pkl"

# Define the specific GPUs to use
GPU_IDS=(2 3 4 5)

# Check if we have enough GPUs for all folds
if [ ${#GPU_IDS[@]} -lt $K ]; then
    echo "Warning: Number of folds ($K) exceeds available GPUs (${#GPU_IDS[@]})"
    echo "Some folds will have to wait for GPUs to become available"
fi

# Ensure the data directory exists
mkdir -p $(dirname $DATA_PATH)

# Prepare data
echo "Preparing data for ensemble training..."
python prepare_ensemble_data.py --output $DATA_PATH

# Launch training for each fold on a different GPU
echo "Launching distributed training across specified GPUs (${GPU_IDS[@]})..."
for ((fold=1; fold<=$K; fold++))
do
    # Use modulo to wrap around if more folds than GPUs
    idx=$(( (fold-1) % ${#GPU_IDS[@]} ))
    gpu_id=${GPU_IDS[$idx]}
    echo "Launching fold $fold on GPU $gpu_id"
    python train_fold_on_gpu.py --fold $fold --gpu $gpu_id --data $DATA_PATH --epochs $EPOCHS --k $K &
done

# Wait for all processes to complete
wait

echo "All training processes complete. Now evaluating the ensemble."

# Run evaluation of the ensemble
python evaluate_ensemble.py --data $DATA_PATH --model-dir "./models/task2x_ensemble"