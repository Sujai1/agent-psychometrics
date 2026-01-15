# SAD-IRT: State-Aware Deep Item Response Theory

This experiment implements SAD-IRT for SWE-bench task difficulty estimation using agent trajectories.

## Model

**Core equation:**
```
P(y_ij = 1 | θ_j, β_i, ψ_ij) = σ(θ_j - (β_i + ψ_ij))
```

Where:
- `θ_j`: Agent ability (learnable embedding)
- `β_i`: Task difficulty (learnable embedding)
- `ψ_ij`: Trajectory-based interaction term (predicted by Qwen3 + LoRA + MLP)

The `ψ_ij` parameter is constrained to be zero-mean via BatchNorm for identifiability.

## Input Format

Each sample combines:
```
[PROBLEM]
{problem_statement from SWE-bench}

[SOLUTION]
{gold_patch from SWE-bench}

[TRAJECTORY]
{agent trajectory - conversation between system/user/assistant}
```

If the total length exceeds `max_length`, the trajectory is truncated from the beginning (keeping the suffix).

## Evaluation

### Part 2: Full AUC (PRIMARY)
- Train SAD-IRT on all agents, 80% of (agent, task) pairs
- Test on held-out 20% of pairs (both agent and task seen in training)
- Compare AUC-ROC to baseline 1PL IRT

### Part 1: Calibration (SECONDARY)
- Train on M1+M2 agents only
- Compare difficulty estimates on hard tasks to oracle (trained on M1+M2+M3)

## Usage

### Local dry run
```bash
source .venv/bin/activate
python -m experiment_sad_irt.train_evaluate --dry_run --max_samples 50
```

### Full training (single GPU)
```bash
python -m experiment_sad_irt.train_evaluate \
    --mode full_auc \
    --model_name Qwen/Qwen3-0.6B \
    --max_length 8192 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 3 \
    --output_dir chris_output/sad_irt
```

### Multi-GPU with accelerate
```bash
accelerate launch \
    --num_processes=2 \
    --multi_gpu \
    --mixed_precision=bf16 \
    -m experiment_sad_irt.train_evaluate \
    --mode full_auc \
    --model_name Qwen/Qwen3-0.6B \
    --max_length 8192 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs 3
```

### MIT Engaging cluster
```bash
# Single GPU
sbatch run_sad_irt_single_gpu.sh

# 2x H200 GPUs
sbatch run_sad_irt_h200.sh

# Monitor
squeue -u $USER
tail -f logs/sad_irt_*.out
```

## Installation

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate hf_transfer datasets pandas scikit-learn tqdm numpy jsonlines scipy
```

## Files

- `config.py`: Configuration dataclass
- `dataset.py`: Trajectory dataset loader (drops missing trajectories)
- `model.py`: SAD-IRT and baseline StandardIRT models
- `train.py`: Training loop with gradient accumulation
- `evaluate.py`: Metrics (AUC, Brier score, calibration)
- `train_evaluate.py`: Main entry point

## Expected Outputs

```
chris_output/sad_irt/
├── results.json           # Final metrics comparison
├── checkpoint_best.pt     # Best SAD-IRT checkpoint
├── checkpoint_epoch_*.pt  # Epoch checkpoints
```
