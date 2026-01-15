"""SAD-IRT model with Qwen3 trajectory encoder."""

import logging
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


class SADIRT(nn.Module):
    """State-Aware Deep Item Response Theory model.

    P(y_ij = 1 | θ_j, β_i, ψ_ij) = σ(θ_j - (β_i + ψ_ij))

    Where:
    - θ_j: Agent ability (learnable embedding)
    - β_i: Task difficulty (learnable embedding)
    - ψ_ij: Trajectory-based interaction term (predicted by encoder)

    ψ_ij is constrained to be zero-mean via BatchNorm for identifiability.
    """

    def __init__(
        self,
        num_agents: int,
        num_tasks: int,
        model_name: str = "Qwen/Qwen3-0.6B",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[list] = None,
        use_gradient_checkpointing: bool = True,
    ):
        """Initialize SAD-IRT model.

        Args:
            num_agents: Number of unique agents
            num_tasks: Number of unique tasks
            model_name: HuggingFace model name for trajectory encoder
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
            lora_target_modules: Modules to apply LoRA to
            use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        """
        super().__init__()

        self.num_agents = num_agents
        self.num_tasks = num_tasks

        # IRT parameters (learnable embeddings)
        self.theta = nn.Embedding(num_agents, 1)  # Agent abilities
        self.beta = nn.Embedding(num_tasks, 1)  # Task difficulties

        # Initialize with small values
        nn.init.normal_(self.theta.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.beta.weight, mean=0.0, std=0.1)

        # Load base model
        logger.info(f"Loading base model: {model_name}")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Apply LoRA
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.encoder = get_peft_model(self.encoder, peft_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            f"LoRA applied: {trainable_params:,} trainable params "
            f"({trainable_params / total_params * 100:.2f}% of {total_params:,} total)"
        )

        # Get hidden size from config
        encoder_dim = config.hidden_size

        # MLP head for ψ prediction
        self.psi_head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        # Initialize final layer to output near-zero values
        # This makes ψ ≈ 0 at start, so model begins as standard IRT
        # and gradually learns trajectory-based corrections
        nn.init.zeros_(self.psi_head[-1].weight)
        nn.init.zeros_(self.psi_head[-1].bias)

        # BatchNorm for zero-mean constraint (affine=False to not learn shift/scale)
        self.psi_bn = nn.BatchNorm1d(1, affine=False, momentum=0.1)

    def forward(
        self,
        agent_idx: torch.Tensor,
        task_idx: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            agent_idx: Agent indices (batch_size,)
            task_idx: Task indices (batch_size,)
            input_ids: Tokenized input (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            logits: Unnormalized log-odds of success (batch_size,)
        """
        # Get IRT parameters
        theta = self.theta(agent_idx)  # (batch_size, 1)
        beta = self.beta(task_idx)  # (batch_size, 1)

        # Encode trajectory
        # Use last token representation (common for causal LMs)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Get representation from last non-padding token
        # Find the position of the last non-padding token for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_hidden = hidden_states[batch_indices, seq_lengths]  # (batch_size, hidden_dim)

        # Predict ψ
        psi_raw = self.psi_head(last_token_hidden.float())  # (batch_size, 1)

        # Apply BatchNorm for zero-mean constraint
        # Only apply during training with batch size > 1
        if self.training and psi_raw.size(0) > 1:
            psi = self.psi_bn(psi_raw)
        else:
            # During eval or batch_size=1, use running stats
            self.psi_bn.eval()
            psi = self.psi_bn(psi_raw)
            if self.training:
                self.psi_bn.train()

        # IRT formula: logit = θ - (β + ψ)
        logits = theta - (beta + psi)

        return logits.squeeze(-1)

    def get_abilities(self) -> torch.Tensor:
        """Get agent ability parameters."""
        return self.theta.weight.detach().squeeze(-1)

    def get_difficulties(self) -> torch.Tensor:
        """Get task difficulty parameters."""
        return self.beta.weight.detach().squeeze(-1)

    def get_psi_stats(self) -> dict:
        """Get statistics about ψ BatchNorm."""
        return {
            "running_mean": self.psi_bn.running_mean.item() if self.psi_bn.running_mean is not None else None,
            "running_var": self.psi_bn.running_var.item() if self.psi_bn.running_var is not None else None,
        }


class StandardIRT(nn.Module):
    """Standard 1PL IRT model without trajectory features (baseline).

    P(y_ij = 1 | θ_j, β_i) = σ(θ_j - β_i)
    """

    def __init__(self, num_agents: int, num_tasks: int):
        super().__init__()

        self.num_agents = num_agents
        self.num_tasks = num_tasks

        # IRT parameters
        self.theta = nn.Embedding(num_agents, 1)
        self.beta = nn.Embedding(num_tasks, 1)

        # Initialize
        nn.init.normal_(self.theta.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.beta.weight, mean=0.0, std=0.1)

    def forward(
        self,
        agent_idx: torch.Tensor,
        task_idx: torch.Tensor,
        **kwargs,  # Ignore other inputs
    ) -> torch.Tensor:
        """Forward pass."""
        theta = self.theta(agent_idx)
        beta = self.beta(task_idx)
        logits = theta - beta
        return logits.squeeze(-1)

    def get_abilities(self) -> torch.Tensor:
        return self.theta.weight.detach().squeeze(-1)

    def get_difficulties(self) -> torch.Tensor:
        return self.beta.weight.detach().squeeze(-1)
