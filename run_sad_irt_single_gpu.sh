#!/bin/bash
#SBATCH --job-name=sad_irt
#SBATCH --output=logs/sad_irt_%j.out
#SBATCH --error=logs/sad_irt_%j.err
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Load modules
module load miniforge 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Run training (single GPU)
python -m experiment_sad_irt.train_evaluate \
    --mode full_auc \
    --model_name Qwen/Qwen3-0.6B \
    --max_length 8192 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 3 \
    --output_dir chris_output/sad_irt

echo "End time: $(date)"
echo "Exit code: $?"
