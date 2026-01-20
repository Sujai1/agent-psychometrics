#!/bin/bash
#SBATCH --job-name=tb_embed_single
#SBATCH --output=logs/terminalbench_embeddings_single_%j.out
#SBATCH --error=logs/terminalbench_embeddings_single_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8

# TerminalBench Embeddings Generation - Single Task (headless-terminal)
# Regenerates ALL embeddings to include the newly added headless-terminal task.
#
# Usage:
#   sbatch slurm_scripts/terminalbench_embeddings_single.sh

set -e

BACKBONE="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

echo "=== TerminalBench Embeddings Generation (with headless-terminal) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Backbone: $BACKBONE"
echo ""

# Set up environment
cd ~/model_irt
source .venv/bin/activate

# Use scratch for HuggingFace cache (avoid home quota limits)
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
mkdir -p "$HF_HOME"

# Create output directory
OUTPUT_DIR="chris_output/experiment_a_terminalbench/embeddings"
mkdir -p "$OUTPUT_DIR"

# Create logs directory if needed
mkdir -p logs

echo "Environment:"
echo "  Python: $(which python)"
echo "  HF_HOME: $HF_HOME"
echo "  Output: $OUTPUT_DIR"
echo ""

# Verify headless-terminal task exists
if [ ! -f "terminal-bench/tasks/headless-terminal/task.yaml" ]; then
    echo "ERROR: terminal-bench/tasks/headless-terminal/task.yaml not found!"
    echo "Make sure you've committed and pushed the task directory."
    exit 1
fi

echo "Verified: headless-terminal task exists"
echo ""

# Run embedding generation for ALL tasks (will include headless-terminal now)
python -m experiment_a_terminalbench.generate_embeddings \
    --items_path chris_output/terminal_bench_2.0_binomial_1pl/1d/items.csv \
    --repo_path terminal-bench \
    --out_dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --max_length 8192 \
    --batch_size 1 \
    --device_map auto \
    --torch_dtype bfloat16 \
    --trust_remote_code

echo ""
echo "=== Embedding generation complete ==="
echo "End time: $(date)"

# Verify output
python3 -c "
import numpy as np
emb = np.load('$OUTPUT_DIR/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__maxlen8192.npz', allow_pickle=True)
task_ids = list(emb['task_ids'])
print(f'Total tasks embedded: {len(task_ids)}')
print(f'headless-terminal included: {\"headless-terminal\" in task_ids}')
"
