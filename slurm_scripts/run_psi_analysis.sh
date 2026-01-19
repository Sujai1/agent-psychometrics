#!/bin/bash
#SBATCH --job-name=psi_analysis
#SBATCH --output=logs/psi_analysis_%j.out
#SBATCH --error=logs/psi_analysis_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1

# Default checkpoint path (can be overridden with --checkpoint=...)
CHECKPOINT="${CHECKPOINT:-chris_output/sad_irt_long/full_20260118_024625_psi_batchnorm_lora_r64/checkpoint_epoch_9_step4248_20260118_044922.pt}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --checkpoint=*)
            CHECKPOINT="${arg#*=}"
            ;;
        --batch_size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
    esac
done

# Create log directory
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "Batch size: $BATCH_SIZE"

# Load modules
module load miniforge 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Run analysis
python -m experiment_sad_irt.analyze_psi_distribution \
    --checkpoint "$CHECKPOINT" \
    --batch_size "$BATCH_SIZE"

echo "End time: $(date)"
echo "Exit code: $?"
