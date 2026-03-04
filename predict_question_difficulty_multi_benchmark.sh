#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship
source .venv/bin/activate

export HF_HOME="/orcd/scratch/orcd/001/daria_k/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

python -u predict_question_difficulty_multi_benchmark.py \
  --trust_remote_code \
  --train_benchmarks terminalbench \
  --out_dir data/terminalbench \
  --method embedding \
  --split_by task \
  --include_zero_success \
  --overwrite