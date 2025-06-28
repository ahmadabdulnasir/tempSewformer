#!/bin/bash
# Script to train the Sewformer model

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include necessary directories
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/Factory/packages

# Create all necessary directories
mkdir -p "$(pwd)/former/assets/data_configs"
mkdir -p "$(pwd)/data_configs"
mkdir -p "$(pwd)/outputs/checkpoints"

# Copy configuration files if they don't exist in the target location
if [ ! -f "$(pwd)/data_configs/panel_classes_condenced.json" ]; then
  echo "Copying panel_classes_condenced.json to data_configs directory"
  cp "$(pwd)/former/assets/data_configs/panel_classes_condenced.json" "$(pwd)/data_configs/" 2>/dev/null || echo "Warning: Could not copy panel_classes_condenced.json"
fi

if [ ! -f "$(pwd)/data_configs/param_filter.json" ]; then
  echo "Copying param_filter.json to data_configs directory"
  cp "$(pwd)/former/assets/data_configs/param_filter.json" "$(pwd)/data_configs/" 2>/dev/null || echo "Warning: Could not copy param_filter.json"
fi

# Check if the dataset directory exists, if not create a minimal structure for testing
DATASET_DIR="$(pwd)/Factory/sewformer_dataset"
if [ ! -d "$DATASET_DIR" ]; then
  echo "Dataset directory not found! Creating minimal structure for testing"
  mkdir -p "$DATASET_DIR"
fi

# Create system.json if it doesn't exist
if [ ! -f "$(pwd)/system.json" ]; then
  echo "Creating system.json configuration file"
  cat > "$(pwd)/system.json" << EOL
{
  "wandb_username": "your_wandb_username",
  "datasets_path": "$(pwd)/Factory/sewformer_dataset",
  "sim_root": "$(pwd)/Factory/sewformer_dataset"
}
EOL
  echo "Created system.json with default values. Please update with your actual wandb username if needed."
fi

# Disable wandb syncing - we're running in offline mode
export WANDB_MODE="dryrun"

# Detect if we're running on a server with multiple GPUs or locally
if [ -z "$LOCAL_TRAINING" ]; then
  # Check available GPUs
  if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  else
    GPU_COUNT=0
  fi

  # Determine if we're on a Mac
  if [[ "$(uname)" == "Darwin" ]]; then
    IS_MAC=1
  else
    IS_MAC=0
  fi

  # Set training mode based on environment
  if [ $GPU_COUNT -gt 1 ] && [ $IS_MAC -eq 0 ]; then
    # Server mode with multiple GPUs
    echo "Starting training with $GPU_COUNT GPUs..."
    torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_COUNT former/train.py -c former/configs/train.yaml
  else
    # Local mode (Mac or single/no GPU)
    echo "Starting training in local mode..."
    # For Mac/CPU environments, set worker count lower to avoid warnings
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    # Use Python directly without distributed training
    python former/train.py -c former/configs/train_local.yaml
  fi
else
  # Explicitly requested local training
  echo "Starting training in local mode (explicitly requested)..."
  # For Mac/CPU environments, set worker count lower to avoid warnings
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  python former/train.py -c former/configs/train_local.yaml
fi
