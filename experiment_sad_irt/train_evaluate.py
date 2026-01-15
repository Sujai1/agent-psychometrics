"""Main entry point for SAD-IRT training and evaluation."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

# Check if accelerate is available for multi-GPU
try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

from .config import SADIRTConfig
from .dataset import TrajectoryIRTDataset, create_train_test_split
from .evaluate import compute_metrics, log_parameter_stats
from .model import SADIRT, StandardIRT
from .train import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> SADIRTConfig:
    """Parse command line arguments into config."""
    parser = argparse.ArgumentParser(description="SAD-IRT Training and Evaluation")

    # Mode
    parser.add_argument("--mode", type=str, default="full_auc", choices=["full_auc", "calibration"])

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lora_r", type=int, default=16)

    # Data
    parser.add_argument("--response_matrix_path", type=str, default="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    parser.add_argument("--trajectory_dir", type=str, default="trajectory_data/unified_trajs")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--hard_threshold", type=float, default=0.2)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate_encoder", type=float, default=1e-4)
    parser.add_argument("--learning_rate_embeddings", type=float, default=1e-3)

    # Output
    parser.add_argument("--output_dir", type=str, default="chris_output/sad_irt")
    parser.add_argument("--seed", type=int, default=42)

    # Debug
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true", help="Quick test: load model, run 1 batch, exit")

    # Resumption
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Convert to config
    config = SADIRTConfig(
        mode=args.mode,
        model_name=args.model_name,
        lora_r=args.lora_r,
        response_matrix_path=args.response_matrix_path,
        trajectory_dir=args.trajectory_dir,
        max_length=args.max_length,
        test_fraction=args.test_fraction,
        hard_threshold=args.hard_threshold,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate_encoder=args.learning_rate_encoder,
        learning_rate_embeddings=args.learning_rate_embeddings,
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        smoke_test=args.smoke_test,
        resume_from=args.resume_from,
    )

    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_smoke_test(config: SADIRTConfig):
    """Quick smoke test: load everything, run 1 forward/backward pass, exit."""
    logger.info("=" * 60)
    logger.info("SMOKE TEST: Checking code paths")
    logger.info("=" * 60)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded OK")

    # Create minimal dataset (just 10 samples)
    logger.info("Loading minimal dataset...")
    full_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        swebench_dataset=config.swebench_dataset,
    )
    logger.info(f"Dataset: {len(full_dataset)} total samples, {full_dataset.num_agents} agents, {full_dataset.num_tasks} tasks")

    # Get just 4 samples for smoke test
    train_pairs, test_pairs = create_train_test_split(full_dataset, test_fraction=0.5, seed=config.seed)
    train_pairs = train_pairs[:4]

    mini_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        agent_ids=full_dataset.agent_ids,
        task_ids=full_dataset.task_ids,
        pairs=train_pairs,
        swebench_dataset=config.swebench_dataset,
    )
    logger.info(f"Mini dataset: {len(mini_dataset)} samples")

    # Load one batch
    loader = DataLoader(mini_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    logger.info(f"Batch loaded: input_ids shape = {batch['input_ids'].shape}")

    # Create model
    logger.info(f"Creating SAD-IRT model with {config.model_name}...")
    model = SADIRT(
        num_agents=full_dataset.num_agents,
        num_tasks=full_dataset.num_tasks,
        model_name=config.model_name,
        lora_r=config.lora_r,
    ).to(device)
    logger.info("Model created OK")

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward pass
    logger.info("Running forward pass...")
    model.train()
    logits = model(
        agent_idx=batch["agent_idx"],
        task_idx=batch["task_idx"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logger.info(f"Forward pass OK: logits shape = {logits.shape}, values = {logits.detach().cpu().numpy()}")

    # Backward pass
    logger.info("Running backward pass...")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["response"])
    loss.backward()
    logger.info(f"Backward pass OK: loss = {loss.item():.4f}")

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"Gradients computed for {grad_count}/{total_params} trainable parameters")

    logger.info("=" * 60)
    logger.info("SMOKE TEST PASSED")
    logger.info("=" * 60)

    return {"status": "passed"}


def run_full_auc_evaluation(config: SADIRTConfig):
    """Run Part 2: Full AUC evaluation comparing SAD-IRT to baseline IRT."""
    logger.info("=" * 60)
    logger.info("Part 2: Full AUC Evaluation")
    logger.info("=" * 60)

    # Set seed
    set_seed(config.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create full dataset (to get all agents/tasks)
    logger.info("Loading dataset...")
    full_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        swebench_dataset=config.swebench_dataset,
    )

    logger.info(f"Full dataset: {len(full_dataset)} samples")
    logger.info(f"Agents: {full_dataset.num_agents}, Tasks: {full_dataset.num_tasks}")

    # Create train/test split by (agent, task) pairs
    train_pairs, test_pairs = create_train_test_split(
        full_dataset, test_fraction=config.test_fraction, seed=config.seed
    )

    # Limit samples if dry run
    if config.dry_run or config.max_samples:
        max_samples = config.max_samples or 100
        train_pairs = train_pairs[:max_samples]
        test_pairs = test_pairs[:min(max_samples // 4, len(test_pairs))]
        logger.info(f"DRY RUN: Limited to {len(train_pairs)} train, {len(test_pairs)} test pairs")

    # Create train/test datasets with the specific pairs
    train_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        agent_ids=full_dataset.agent_ids,
        task_ids=full_dataset.task_ids,
        pairs=train_pairs,
        swebench_dataset=config.swebench_dataset,
    )

    test_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        agent_ids=full_dataset.agent_ids,
        task_ids=full_dataset.task_ids,
        pairs=test_pairs,
        swebench_dataset=config.swebench_dataset,
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ===== Train Baseline IRT =====
    logger.info("\n" + "=" * 40)
    logger.info("Training Baseline IRT (no trajectories)")
    logger.info("=" * 40)

    baseline_model = StandardIRT(
        num_agents=full_dataset.num_agents,
        num_tasks=full_dataset.num_tasks,
    ).to(device)

    baseline_trainer = Trainer(
        model=baseline_model,
        train_loader=train_loader,
        eval_loader=test_loader,
        config=config,
        device=device,
        is_sad_irt=False,
    )

    baseline_metrics = baseline_trainer.train()
    logger.info(f"Baseline IRT final metrics: {baseline_metrics}")
    log_parameter_stats(baseline_model, prefix="Baseline ")

    # ===== Train SAD-IRT =====
    logger.info("\n" + "=" * 40)
    logger.info("Training SAD-IRT (with trajectory encoder)")
    logger.info("=" * 40)

    sad_irt_model = SADIRT(
        num_agents=full_dataset.num_agents,
        num_tasks=full_dataset.num_tasks,
        model_name=config.model_name,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    ).to(device)

    sad_irt_trainer = Trainer(
        model=sad_irt_model,
        train_loader=train_loader,
        eval_loader=test_loader,
        config=config,
        device=device,
        is_sad_irt=True,
    )

    # Resume from checkpoint if specified
    if config.resume_from:
        sad_irt_trainer.load_checkpoint(config.resume_from)

    sad_irt_metrics = sad_irt_trainer.train()
    logger.info(f"SAD-IRT final metrics: {sad_irt_metrics}")
    log_parameter_stats(sad_irt_model, prefix="SAD-IRT ")

    # ===== Compare Results =====
    logger.info("\n" + "=" * 40)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 40)

    baseline_auc = baseline_metrics.get("auc", 0)
    sad_irt_auc = sad_irt_metrics.get("auc", 0)
    improvement = sad_irt_auc - baseline_auc

    logger.info(f"Baseline IRT AUC: {baseline_auc:.4f}")
    logger.info(f"SAD-IRT AUC:      {sad_irt_auc:.4f}")
    logger.info(f"Improvement:      {improvement:+.4f} ({improvement / baseline_auc * 100:+.2f}%)")

    # Save results
    results = {
        "config": vars(config),
        "baseline_metrics": baseline_metrics,
        "sad_irt_metrics": sad_irt_metrics,
        "improvement": improvement,
        "num_train_samples": len(train_dataset),
        "num_test_samples": len(test_dataset),
        "num_agents": full_dataset.num_agents,
        "num_tasks": full_dataset.num_tasks,
    }

    output_path = Path(config.output_dir) / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    return results


def run_calibration_evaluation(config: SADIRTConfig):
    """Run Part 1: Calibration evaluation on hard tasks."""
    logger.info("=" * 60)
    logger.info("Part 1: Calibration Evaluation (Hard Tasks)")
    logger.info("=" * 60)

    # TODO: Implement calibration evaluation
    # This requires:
    # 1. Train SAD-IRT on M1+M2 agents only
    # 2. Train oracle IRT on M1+M2+M3 agents
    # 3. Compare β estimates on hard tasks

    raise NotImplementedError(
        "Calibration evaluation not yet implemented. "
        "Use --mode full_auc for Part 2 evaluation first."
    )


def main():
    """Main entry point."""
    config = parse_args()

    logger.info("Configuration:")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")

    # Smoke test mode - just check code paths
    if config.smoke_test:
        run_smoke_test(config)
        return

    if config.mode == "full_auc":
        run_full_auc_evaluation(config)
    elif config.mode == "calibration":
        run_calibration_evaluation(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    main()
