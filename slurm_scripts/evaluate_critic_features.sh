#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH --job-name=eval_critic
#SBATCH --output=logs/eval_critic_%j.out
#SBATCH --error=logs/eval_critic_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Evaluate critic model features for Experiment B
# Runs posterior difficulty prediction with critic features
#
# Run from: ~/model_irt
# Submit with: sbatch slurm_scripts/evaluate_critic_features.sh

set -euo pipefail

# Project in home directory
cd ~/model_irt

# Load modules and activate conda environment
module load miniforge
conda activate irt

# Create log directory if needed
mkdir -p logs

echo "=== Environment Info ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python -V
echo ""

echo "=== Running Experiment B with Critic Features ==="
python -m experiment_b.train_evaluate \
    --feature_source critic_model \
    --critic_features_dir chris_output/experiment_b/critic_rewards

echo ""
echo "=== Done ==="
echo "Date: $(date)"
echo "Results: chris_output/experiment_b/experiment_b_results.json"
