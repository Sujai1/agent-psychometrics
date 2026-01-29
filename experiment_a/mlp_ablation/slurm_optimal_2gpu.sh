#!/bin/bash
#SBATCH --job-name=mlp_opt_2gpu
#SBATCH --output=chris_output/experiment_a/mlp_ablation/optimal_2gpu_%j.out
#SBATCH --error=chris_output/experiment_a/mlp_ablation/optimal_2gpu_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Setup environment
cd /home/chMDge/model_irt
source .venv/bin/activate

# HuggingFace cache on scratch to avoid quota
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

echo "Starting 2-GPU optimal hyperparameter search at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# Run both GPUs in parallel
CUDA_VISIBLE_DEVICES=0 python -m experiment_a.mlp_ablation.test_optimal_hyperparams_fast --gpu 0 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -m experiment_a.mlp_ablation.test_optimal_hyperparams_fast --gpu 1 &
PID1=$!

# Wait for both to complete
echo "Waiting for GPU 0 (PID $PID0) and GPU 1 (PID $PID1)..."
wait $PID0
STATUS0=$?
wait $PID1
STATUS1=$?

echo "GPU 0 exit status: $STATUS0"
echo "GPU 1 exit status: $STATUS1"

# Combine results
echo ""
echo "Combining results..."
python -c "
import json
from pathlib import Path

output_dir = Path('chris_output/experiment_a/mlp_ablation')
combined = {}

for gpu in [0, 1]:
    path = output_dir / f'optimal_hyperparams_gpu{gpu}.json'
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            combined.update(data)
            print(f'Loaded {len(data)} results from GPU {gpu}')

# Sort and print top 15
sorted_results = sorted(combined.items(), key=lambda x: x[1]['mean_auc'], reverse=True)
print()
print('=' * 70)
print('TOP 15 RESULTS')
print('=' * 70)
for name, r in sorted_results[:15]:
    print(f\"{r['display_name'][:50]:<52} {r['mean_auc']:.4f}\")

# Save combined
with open(output_dir / 'optimal_hyperparams_combined.json', 'w') as f:
    json.dump(combined, f, indent=2)
print(f'\nCombined results saved to {output_dir}/optimal_hyperparams_combined.json')
"

echo ""
echo "Finished at $(date)"
