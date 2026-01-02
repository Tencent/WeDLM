# coding=utf-8
"""WeDLM - A Weighted Diffusion Language Model for SFT training."""

from src.config import WeDLMTrainingConfig
from src.data import (
    WeDLMPackedDataset,
    WeDLMShuffledPackedDataset,
    packed_collate_fn,
    get_im_end_token_id,
)
from src.batch import WeDLMBatch, build_wedlm_batch
from src.model import wedlm_forward, wedlm_attention_forward
from src.loss import compute_mlm_loss, compute_ar_loss
from src.attention import (
    check_backend_available,
    get_available_backend,
    get_attention_wrapper,
)
from src.trainer import WeDLMTrainer

__all__ = [
    # Config
    "WeDLMTrainingConfig",
    # Data
    "WeDLMPackedDataset",
    "WeDLMShuffledPackedDataset",
    "packed_collate_fn",
    "get_im_end_token_id",
    # Batch
    "WeDLMBatch",
    "build_wedlm_batch",
    # Model
    "wedlm_forward",
    "wedlm_attention_forward",
    # Loss
    "compute_mlm_loss",
    "compute_ar_loss",
    # Attention
    "check_backend_available",
    "get_available_backend",
    "get_attention_wrapper",
    # Trainer
    "WeDLMTrainer",
]

