#!/bin/bash
#SBATCH --job-name=traj_embed_2gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/traj_embed_2gpu_%j.out
#SBATCH --error=logs/traj_embed_2gpu_%j.err

# 2-GPU trajectory embedding computation
# Runs 2 parallel processes, each on its own GPU
# Use this if you hit QOSMaxGRESPerUser limits with 8 GPUs
#
# Usage:
#   sbatch scripts/embedding/compute_embeddings_2gpu.sh
#
# Expected time: ~3 hours (vs ~45 min with 8 GPUs)

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
NUM_GPUS=2

# Paths
TRAJECTORIES_DIR="trajectory_data/unified_trajs"
OUTPUT_DIR="chris_output/experiment_b/trajectory_embeddings"

echo "=============================================="
echo "2-GPU Trajectory Embedding Computation"
echo "=============================================="
echo "Backbone: $BACKBONE"
echo "Content type: $CONTENT_TYPE"
echo "Instruction type: $INSTRUCTION_TYPE"
echo "Max length: $MAX_LENGTH"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

# Launch parallel processes, each assigned to a different GPU
pids=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching shard $i on GPU $i..."

    CUDA_VISIBLE_DEVICES=$i python -m experiment_b.compute_trajectory_embeddings \
        --trajectories_dir "$TRAJECTORIES_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --backbone "$BACKBONE" \
        --embedding_layer -1 \
        --max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --content_type "$CONTENT_TYPE" \
        --instruction_type "$INSTRUCTION_TYPE" \
        --device_map "none" \
        --shard_id "$i" \
        --num_shards "$NUM_GPUS" \
        > "logs/shard_${SLURM_JOB_ID}_gpu${i}.log" 2>&1 &

    pids+=($!)
done

echo "All $NUM_GPUS shards launched. Waiting for completion..."

# Wait for all processes and check exit codes
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "ERROR: Shard $i (PID ${pids[$i]}) failed!"
        failed=1
    else
        echo "Shard $i completed successfully"
    fi
done

if [ $failed -eq 1 ]; then
    echo "Some shards failed. Check logs/shard_${SLURM_JOB_ID}_gpu*.log"
    exit 1
fi

echo ""
echo "=============================================="
echo "All shards completed successfully!"
echo "=============================================="
echo "Output: $OUTPUT_DIR/${CONTENT_TYPE}_${INSTRUCTION_TYPE}/"

# Count total embeddings
total=$(find "$OUTPUT_DIR/${CONTENT_TYPE}_${INSTRUCTION_TYPE}" -name "*.npz" 2>/dev/null | wc -l)
echo "Total embeddings: $total"
