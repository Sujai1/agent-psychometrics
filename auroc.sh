#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship

source /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/activate

which python
python -V
python -c "import sys; print(sys.executable)"

python auroc.py




