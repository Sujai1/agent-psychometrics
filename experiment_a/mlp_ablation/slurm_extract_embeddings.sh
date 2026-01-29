#!/bin/bash
#SBATCH --job-name=extract_emb
#SBATCH --output=logs/extract_emb_%j.out
#SBATCH --error=logs/extract_emb_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu

# Extract agent embeddings from trained AgentEmb model
# Run with: mkdir -p logs && sbatch experiment_a/mlp_ablation/slurm_extract_embeddings.sh

set -e

echo "=========================================="
echo "Extract Agent Embeddings"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

cd /home/cge7/model_irt
source .venv/bin/activate

export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
export PYTHONUNBUFFERED=1

python -m experiment_a.mlp_ablation.extract_agent_embeddings

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "Output: chris_output/experiment_a/mlp_embedding/agent_embeddings.csv"
echo "=========================================="
