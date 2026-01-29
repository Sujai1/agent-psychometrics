#!/bin/bash
#SBATCH --job-name=interaction_sweep
#SBATCH --output=logs/interaction_sweep_%j.out
#SBATCH --error=logs/interaction_sweep_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu

# Interaction Architecture Sweep (Part 5)
# Tests new ways to combine agent and task features:
# - Two-Tower (dot product)
# - Bilinear interaction
# - NCF (Neural Collaborative Filtering)
# - Multiplicative interaction
# - Agent Embedding (learned low-dim embeddings)
#
# Run with: mkdir -p logs && sbatch experiment_a/mlp_ablation/slurm_interaction_sweep.sh

set -e

echo "=========================================="
echo "Interaction Architecture Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

cd /home/cge7/model_irt
source .venv/bin/activate

export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo ""
echo "Running interaction architecture sweep..."
python -m experiment_a.mlp_ablation.interaction_sweep

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
