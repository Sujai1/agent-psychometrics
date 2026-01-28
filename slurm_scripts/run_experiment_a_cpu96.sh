#!/bin/bash
#SBATCH --job-name=exp_a_cpu96
#SBATCH --output=logs/exp_a_cpu96_%j.out
#SBATCH --error=logs/exp_a_cpu96_%j.err
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# Experiment A: Run on all datasets with full CPU parallelization
#
# Parallelization strategy (96 cores):
# - Datasets run sequentially (4 datasets)
# - Methods within each dataset run in parallel (n_jobs_methods=-1 = all cores)
# - Folds within each method run in parallel (n_jobs_folds=5 = all 5 folds)
#
# This maximizes core utilization while avoiding nested parallelism issues.
# Each dataset uses all 96 cores for its methods and folds.

set -e

echo "Starting Experiment A on $(hostname)"
echo "Date: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# Load conda environment
module load miniforge
conda activate irt

# Set HuggingFace cache to scratch (avoid home quota limits)
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

# Set OMP threads to avoid over-subscription with joblib
export OMP_NUM_THREADS=1

# Change to project directory
cd ~/model_irt

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="chris_output/experiment_a/results_${TIMESTAMP}.csv"
OUTPUT_DIR="/tmp/experiment_a_${SLURM_JOB_ID}"

echo "Output CSV: $OUTPUT_CSV"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Run all datasets with full parallelization
# - --sequential: run datasets one at a time (each uses all 96 cores)
# - --n_jobs_methods=-1: parallelize all methods within a dataset
# - --n_jobs_folds=5: parallelize all 5 CV folds within each method
python -m experiment_a.run_all_datasets \
    --output "$OUTPUT_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --sequential \
    --n_jobs_methods=-1 \
    --n_jobs_folds=5

echo ""
echo "Experiment A completed at $(date)"
echo "Results saved to: $OUTPUT_CSV"