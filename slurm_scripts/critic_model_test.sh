#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH --job-name=critic_test
#SBATCH --output=logs/critic_test_%j.out
#SBATCH --error=logs/critic_test_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8

# Test script for OpenHands Critic Model
# Runs on a small subset to verify model loading and inference work
#
# Run from: ~/model_irt
# Submit with: sbatch slurm_scripts/critic_model_test.sh

set -euo pipefail

cd ~/model_irt

module load miniforge
conda activate irt

export HF_HOME="${PWD}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}"
mkdir -p logs

echo "=== Environment Info ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python -V
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Test 1: Verify model can be loaded
echo "=== Test 1: Model Loading ==="
python -c "
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    'all-hands/openhands-critic-32b-exp-20250417',
    trust_remote_code=True
)
print(f'Tokenizer loaded. Vocab size: {tokenizer.vocab_size}')

print('Loading model...')
model = AutoModelForTokenClassification.from_pretrained(
    'all-hands/openhands-critic-32b-exp-20250417',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)
print(f'Model loaded on {next(model.parameters()).device}')
print(f'Model config: {model.config.architectures}')
print(f'Num labels: {model.config.num_labels}')

# Quick inference test
text = 'User: Fix the bug\n\nAssistant: I will fix the bug by editing the file.\n\n'
inputs = tokenizer(text, return_tensors='pt').to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
print(f'Output logits shape: {outputs.logits.shape}')
print(f'Sample logits (last 5 tokens): {outputs.logits[0, -5:, 0].tolist()}')
print('Model loading test PASSED')
"

echo ""
echo "=== Test 2: Process 5 Trajectories ==="
python -m experiment_b.critic_model.compute_rewards \
    --trajectories_dir trajectory_data/unified_trajs \
    --output_dir chris_output/experiment_b/critic_rewards_test \
    --limit 5

echo ""
echo "=== Test 3: Verify Output Files ==="
echo "Output files:"
find chris_output/experiment_b/critic_rewards_test -name "*.npz" | head -10

echo ""
echo "Inspecting first output file:"
python -c "
import numpy as np
from pathlib import Path

output_dir = Path('chris_output/experiment_b/critic_rewards_test')
npz_files = list(output_dir.rglob('*.npz'))
if npz_files:
    f = npz_files[0]
    print(f'File: {f}')
    data = np.load(f)
    print(f'Keys: {list(data.keys())}')
    print(f'Rewards shape: {data[\"rewards\"].shape}')
    print(f'Rewards: {data[\"rewards\"]}')
    print(f'Task ID: {data[\"task_id\"]}')
    print(f'Agent ID: {data[\"agent_id\"]}')
    print(f'Resolved: {data[\"resolved\"]}')
else:
    print('No output files found!')
"

echo ""
echo "=== All Tests Passed ==="
echo "Date: $(date)"
