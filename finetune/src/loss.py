# coding=utf-8
"""Loss functions for WeDLM training."""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def compute_mlm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masked_indices: torch.Tensor,
    p_mask: torch.Tensor,
    weighting_scheme: str = "weighted",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute MLM loss on masked positions.
    
    Args:
        logits: Model output logits
        targets: Original token ids (before masking)
        masked_indices: Bool tensor indicating masked positions
        p_mask: Per-token masking ratio
        weighting_scheme: "weighted" (1/γ weighting) or "uniform"
        eps: Small value to prevent division by zero
    
    Returns:
        loss: Scalar loss tensor
        logs: Dictionary of logging metrics
    """
    device = logits.device
    num_masked = masked_indices.sum()
    
    if num_masked == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"mlm/loss": zero.detach(), "mlm/num_tokens": torch.tensor(0, device=device)}
    
    masked_logits = logits[masked_indices]
    masked_targets = targets[masked_indices]
    per_token_loss = F.cross_entropy(masked_logits, masked_targets, reduction="none")
    
    if weighting_scheme == "weighted":
        weights = 1.0 / (p_mask[masked_indices] + eps)
        weights = weights / weights.sum()
        loss = (per_token_loss * weights).sum()
    else:
        loss = per_token_loss.mean()
    
    return loss, {
        "mlm/loss": loss.detach(),
        "mlm/num_tokens": num_masked.detach(),
    }


def compute_ar_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute standard autoregressive loss.
    
    Args:
        logits: Model output logits
        labels: Target labels (-100 for ignored positions)
    
    Returns:
        loss: Scalar loss tensor
        logs: Dictionary of logging metrics
    """
    device = logits.device
    
    shift_logits = logits[:-1]
    shift_labels = labels[1:]
    valid_mask = shift_labels != -100
    num_valid = valid_mask.sum()
    
    if num_valid == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"ar/loss": zero.detach(), "ar/num_tokens": torch.tensor(0, device=device)}
    
    per_token_loss = F.cross_entropy(shift_logits, shift_labels, reduction="none", ignore_index=-100)
    loss = per_token_loss.sum() / num_valid
    
    return loss, {
        "ar/loss": loss.detach(),
        "ar/num_tokens": num_valid.detach(),
    }

