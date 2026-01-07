#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal
#
# Train IRT (1PL) on SWE-bench Verified evaluation results aggregated in:
#   trajectory_data/irt_verified.jsonlines

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship

source /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/activate

which python
python -V
python -c "import sys; print(sys.executable)"

DATA="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/trajectory_data/irt_verified.jsonlines"
OUT_DIR="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/irt_verified_1pl_e500_seed0"
mkdir -p "${OUT_DIR}"

python -m py_irt.cli train 1pl "${DATA}" "${OUT_DIR}" \
  --device cpu \
  --epochs 2000 \
  --seed 0 \
  --log-every 50

# Optional: export per-question difficulties to CSV
python -m py_irt.cli export-question-difficulties \
  "${OUT_DIR}/best_parameters.json" \
  "${OUT_DIR}/question_difficulties.csv"




