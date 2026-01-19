"""Utilities for loading SAD-IRT checkpoints for inference."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data_splits import get_pre_frontier_agents
from .dataset import TrajectoryIRTDataset
from .model import SADIRT

logger = logging.getLogger(__name__)


@dataclass
class SADIRTCheckpoint:
    """Container for a loaded SAD-IRT checkpoint with model and data."""

    model: SADIRT
    tokenizer: AutoTokenizer
    dataset: TrajectoryIRTDataset
    agent_ids: List[str]
    task_ids: List[str]
    config: Dict
    device: torch.device

    def create_dataloader(
        self,
        batch_size: int = 16,
        shuffle: bool = False,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create a DataLoader for inference."""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )


def load_checkpoint_for_inference(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> SADIRTCheckpoint:
    """Load a SAD-IRT checkpoint and create an inference-ready model.

    Uses get_pre_frontier_agents() to ensure consistent agent ordering
    between training and inference.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model on (default: auto-detect GPU/CPU)

    Returns:
        SADIRTCheckpoint with model, tokenizer, dataset, and metadata
    """
    checkpoint_path = Path(checkpoint_path)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Get model dimensions from checkpoint
    num_agents = checkpoint["model_state_dict"]["theta.weight"].shape[0]
    num_tasks = checkpoint["model_state_dict"]["beta.weight"].shape[0]
    logger.info(f"Checkpoint has {num_agents} agents, {num_tasks} tasks")

    # Get config with defaults
    model_name = config.get("model_name", "Qwen/Qwen3-0.6B")
    lora_r = config.get("lora_r", 64)
    lora_alpha = config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.1)
    psi_normalization = config.get("psi_normalization", "batchnorm")
    max_length = config.get("max_length", 1024)
    cutoff_date = config.get("frontier_cutoff_date", "20250807")
    response_matrix_path = Path(
        config.get(
            "response_matrix_path",
            "clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
        )
    )
    trajectory_dir = Path(
        config.get("trajectory_dir", "chris_output/trajectory_summaries_api")
    )

    logger.info(f"Model: {model_name}, LoRA r={lora_r}, psi_norm={psi_normalization}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get agent list using canonical function
    pre_frontier_agents, _ = get_pre_frontier_agents(
        response_matrix_path, trajectory_dir, cutoff_date
    )
    logger.info(f"Found {len(pre_frontier_agents)} pre-frontier agents")

    if len(pre_frontier_agents) != num_agents:
        logger.warning(
            f"Agent count mismatch: {len(pre_frontier_agents)} vs {num_agents} in checkpoint"
        )
        pre_frontier_agents = pre_frontier_agents[:num_agents]

    # Create dataset
    logger.info("Creating dataset...")
    dataset = TrajectoryIRTDataset(
        response_matrix_path=str(response_matrix_path),
        trajectory_dir=str(trajectory_dir),
        tokenizer=tokenizer,
        max_length=max_length,
        agent_ids=pre_frontier_agents,
        use_summaries=True,
    )
    logger.info(f"Dataset has {len(dataset)} samples")

    # Create model
    logger.info("Creating model...")
    model = SADIRT(
        num_agents=num_agents,
        num_tasks=num_tasks,
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        psi_normalization=psi_normalization,
    )

    # Load weights
    logger.info("Loading checkpoint weights...")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    return SADIRTCheckpoint(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        agent_ids=pre_frontier_agents,
        task_ids=dataset.task_ids,
        config=config,
        device=device,
    )