#!/bin/bash
#SBATCH --job-name=wd_sweep
#SBATCH --output=logs/wd_sweep_%j_%a.out
#SBATCH --error=logs/wd_sweep_%j_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=1-2

# Weight Decay Sweep for FullMLP (Parallel Execution)
# Uses SLURM array to run two parts in parallel on separate GPUs:
#   Part 1: Baselines + wd=[0.1, 0.2, 0.3, 0.5]
#   Part 2: wd=[0.7, 1.0, 2.0, 5.0, 10.0]
#
# Fixed: h=1024, early_stopping=True, init_from_irt=True
#
# Run with: mkdir -p logs && sbatch experiment_a/mlp_ablation/slurm_weight_decay_sweep.sh
# Results saved to: chris_output/experiment_a/mlp_embedding/weight_decay_sweep_part{1,2}.json

set -e

echo "=========================================="
echo "Weight Decay Sweep - Part ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

# Setup environment
cd ~/model_irt
mkdir -p logs
source .venv/bin/activate
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run the part assigned to this array task
echo ""
echo "Running Weight Decay Sweep Part ${SLURM_ARRAY_TASK_ID}..."
python -m experiment_a.mlp_ablation.weight_decay_sweep --part ${SLURM_ARRAY_TASK_ID}

echo ""
echo "=========================================="
echo "Part ${SLURM_ARRAY_TASK_ID} finished at: $(date)"
echo "=========================================="
