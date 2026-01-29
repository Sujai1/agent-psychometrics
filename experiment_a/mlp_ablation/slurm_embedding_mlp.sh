#!/bin/bash
#SBATCH --job-name=emb_mlp
#SBATCH --output=emb_mlp_%j_%a.out
#SBATCH --error=emb_mlp_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=1-2

# MLP on Embeddings Ablation Study (Parallel Execution)
# Uses SLURM array to run two parts in parallel on separate GPUs:
#   Part 1: Baselines + baseline MLP + frozen IRT configs
#   Part 2: Two-stage + dropout + PCA + early stopping configs
#
# Run with: sbatch experiment_a/mlp_ablation/slurm_embedding_mlp.sh
# Results saved to: chris_output/experiment_a/mlp_embedding/embedding_mlp_results_part{1,2}.json

set -e

echo "=========================================="
echo "MLP on Embeddings Ablation - Part ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
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

# Run the part assigned to this array task
echo ""
echo "Running MLP embedding ablation Part ${SLURM_ARRAY_TASK_ID}..."
python -m experiment_a.mlp_ablation.test_embedding_mlp --part ${SLURM_ARRAY_TASK_ID}

echo ""
echo "=========================================="
echo "Part ${SLURM_ARRAY_TASK_ID} finished at: $(date)"
echo "=========================================="
