#!/bin/bash
#SBATCH --job-name=balanced_arch
#SBATCH --output=logs/balanced_arch_%j.out
#SBATCH --error=logs/balanced_arch_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu

# Balanced Representation Architecture Sweep (Part 6)
# Tests: TaskBottleneck, CrossAttention, FeatureGated
#
# Run with: mkdir -p logs && sbatch experiment_a/mlp_ablation/slurm_interaction_sweep_v2.sh

set -e

echo "=========================================="
echo "Balanced Representation Architecture Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

cd /home/cge7/model_irt
source .venv/bin/activate

export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo ""
echo "Running balanced representation sweep..."
python -m experiment_a.mlp_ablation.interaction_sweep_v2

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
