#!/bin/bash
#SBATCH --job-name=compare_methods
#SBATCH --output=logs/compare_methods_%j.out
#SBATCH --error=logs/compare_methods_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1

# =============================================================================
# Experiment B: Compare Methods for Frontier Task Difficulty Prediction
#
# This script runs compare_methods.py with GPU acceleration for the
# Ordered Logit IRT predictor which benefits from batched optimization.
# =============================================================================

# Create directories
mkdir -p logs
mkdir -p chris_output/experiment_b

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Load modules
module load miniforge 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Verify GPU is available
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run the experiment with GPU device
python -m experiment_b.compare_methods \
    --dataset swebench \
    --device cuda \
    --verbose \
    --output_csv chris_output/experiment_b/results_$(date +%Y%m%d_%H%M%S).csv

echo "End time: $(date)"
echo "Exit code: $?"
