#!/bin/bash
#SBATCH --job-name=two_stage
#SBATCH --output=two_stage_%j.out
#SBATCH --error=two_stage_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=mit_normal_gpu

# Two-Stage Training Ablation Study
# Tests whether initializing from IRT + joint fine-tuning can match frozen IRT performance.
# Run with: sbatch experiment_a/mlp_ablation/slurm_two_stage.sh

set -e

echo "=========================================="
echo "Two-Stage Training Ablation Study"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

# Setup environment
cd ~/model_irt
source .venv/bin/activate
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run two-stage ablation study
echo ""
echo "Running two-stage training ablation on LLM Judge..."
python -m experiment_a.mlp_ablation.test_two_stage

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
