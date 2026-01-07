#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship

source /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/activate

which python
python -V
python -c "import sys; print(sys.executable)"

# Keep HF caches on scratch.
export HF_HOME="/orcd/scratch/orcd/001/daria_k/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

# 1) Build a single trajectories JSONL from verified traj files (stdlib-only).
python trajectory_data/make_trajectories_from_verified_trajs.py \
  --verified-root experiments/evaluation/verified \
  --output trajectory_data/verified_trajectories.jsonl \
  --text-sampling tail \
  --max-chars 12000 \
  --seed 0 \
  --log-invalid-task-examples 20

# 2) Run embedding + linear regression.
DIFFS="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/irt_verified_1pl_e500_seed0/question_difficulties.csv"
TRAJS="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/trajectory_data/verified_trajectories.jsonl"
OUT_DIR="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/verified_qwen25coder14b_lr_valid"
mkdir -p "${OUT_DIR}"

python /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/predict_question_difficulty.py \
  --trajectories "${TRAJS}" \
  --difficulties "${DIFFS}" \
  --backbone Qwen/Qwen2.5-Coder-7B \
  --max_length 1024 \
  --max_chars 12000 \
  --text_sampling tail \
  --batch_size 1 \
  --device_map auto \
  --torch_dtype bfloat16 \
  --attn_implementation auto \
  --seed 0 \
  --out_dir "${OUT_DIR}" \
  --overwrite


