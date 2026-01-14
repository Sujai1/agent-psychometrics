#!/bin/bash
#SBATCH --job-name=traj_embed
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7
#SBATCH --output=logs/traj_embed_%A_%a.out
#SBATCH --error=logs/traj_embed_%A_%a.err

# Array job for parallel embedding computation
# Submits 8 independent jobs, each processing 1/8 of agents
#
# Usage:
#   sbatch scripts/embedding/compute_embeddings_array.sh
#
# Benefits over multi-GPU single job:
# - Jobs can start as resources become available (no need to wait for 8 GPUs)
# - If one fails, others continue
# - Easier to monitor individual progress

set -e

# Load environment
module load miniforge
conda activate irt

# Enable fast HuggingFace downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Configuration
BACKBONE="${BACKBONE:-Qwen/Qwen3-VL-8B-Instruct}"
CONTENT_TYPE="${CONTENT_TYPE:-full}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-difficulty}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# Use SLURM array task ID for sharding
SHARD_ID=$SLURM_ARRAY_TASK_ID
NUM_SHARDS=8

# Paths
TRAJECTORIES_DIR="trajectory_data/unified_trajs"
OUTPUT_DIR="chris_output/experiment_b/trajectory_embeddings"

echo "=============================================="
echo "Array Job: Trajectory Embedding Computation"
echo "=============================================="
echo "Backbone: $BACKBONE"
echo "Content type: $CONTENT_TYPE"
echo "Instruction type: $INSTRUCTION_TYPE"
echo "Shard: $SHARD_ID / $NUM_SHARDS"
echo "=============================================="

python -m experiment_b.compute_trajectory_embeddings \
    --trajectories_dir "$TRAJECTORIES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --embedding_layer -1 \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --content_type "$CONTENT_TYPE" \
    --instruction_type "$INSTRUCTION_TYPE" \
    --shard_id "$SHARD_ID" \
    --num_shards "$NUM_SHARDS"

echo "Shard $SHARD_ID completed!"
