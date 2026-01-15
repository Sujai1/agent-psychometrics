#!/bin/bash
#SBATCH --job-name=sad_irt
#SBATCH --output=logs/sad_irt_%j.out
#SBATCH --error=logs/sad_irt_%j.err
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --nodes=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Load modules (adjust based on cluster setup)
module load miniforge 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Set up distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2

# Run training with accelerate for multi-GPU
# Using accelerate launch for distributed training across 2 GPUs
accelerate launch \
    --num_processes=2 \
    --multi_gpu \
    --mixed_precision=bf16 \
    -m experiment_sad_irt.train_evaluate \
    --mode full_auc \
    --model_name Qwen/Qwen3-0.6B \
    --max_length 8192 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs 3 \
    --output_dir chris_output/sad_irt

echo "End time: $(date)"
echo "Exit code: $?"
