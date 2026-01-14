#!/bin/bash
#SBATCH --job-name=traj_embed_resume
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/embed_resume_%j.out
#SBATCH --error=logs/embed_resume_%j.err

# Resume trajectory embedding computation (skips existing .npz files)
# Single GPU, no sharding - simpler and fills in all gaps

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

# Paths
TRAJECTORIES_DIR="trajectory_data/unified_trajs"
OUTPUT_DIR="chris_output/experiment_b/trajectory_embeddings"

echo "=============================================="
echo "Resume Trajectory Embedding Computation"
echo "=============================================="
echo "Backbone: $BACKBONE"
echo "Content type: $CONTENT_TYPE"
echo "Instruction type: $INSTRUCTION_TYPE"
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
    --device_map "none"

echo ""
echo "=============================================="
echo "Completed!"
echo "=============================================="

# Count total embeddings
total=$(find "$OUTPUT_DIR/${CONTENT_TYPE}_${INSTRUCTION_TYPE}" -name "*.npz" 2>/dev/null | wc -l)
echo "Total embeddings: $total"
