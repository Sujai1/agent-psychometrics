#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH --job-name=critic_rewards
#SBATCH --output=logs/critic_rewards_%j.out
#SBATCH --error=logs/critic_rewards_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16

# OpenHands Critic Model inference for Experiment B
# Extracts per-step reward predictions from agent trajectories
#
# Run from: ~/model_irt
# Submit with: sbatch slurm_scripts/critic_model_inference.sh

set -euo pipefail

# Project in home directory
cd ~/model_irt

# Load modules and activate conda environment
module load miniforge
conda activate irt

# Set HuggingFace cache (avoids filling home quota)
export HF_HOME="${PWD}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}"

# Create log directory if needed
mkdir -p logs

echo "=== Environment Info ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python -V
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Define paths
TRAJECTORIES_DIR="trajectory_data/unified_trajs"
OUTPUT_DIR="chris_output/experiment_b/critic_rewards"

echo "=== Configuration ==="
echo "Trajectories: ${TRAJECTORIES_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Using 2 GPUs with sharding"
echo ""

# Run with sharding across 2 GPUs
echo "=== Starting GPU 0 (shard 0/2) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiment_b.critic_model.compute_rewards \
    --trajectories_dir "${TRAJECTORIES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --shard_id 0 \
    --num_shards 2 \
    > logs/critic_shard0_${SLURM_JOB_ID}.log 2>&1 &
PID0=$!

echo "=== Starting GPU 1 (shard 1/2) ==="
CUDA_VISIBLE_DEVICES=1 python -m experiment_b.critic_model.compute_rewards \
    --trajectories_dir "${TRAJECTORIES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --shard_id 1 \
    --num_shards 2 \
    > logs/critic_shard1_${SLURM_JOB_ID}.log 2>&1 &
PID1=$!

echo "Waiting for both shards to complete..."
echo "Shard 0 PID: ${PID0}"
echo "Shard 1 PID: ${PID1}"

wait ${PID0}
STATUS0=$?
echo "Shard 0 completed with status: ${STATUS0}"

wait ${PID1}
STATUS1=$?
echo "Shard 1 completed with status: ${STATUS1}"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
echo "Output directory: ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}/" | head -20

# Report final status
if [ ${STATUS0} -eq 0 ] && [ ${STATUS1} -eq 0 ]; then
    echo "SUCCESS: Both shards completed successfully"
    exit 0
else
    echo "ERROR: One or more shards failed"
    echo "Check logs/critic_shard*_${SLURM_JOB_ID}.log for details"
    exit 1
fi
