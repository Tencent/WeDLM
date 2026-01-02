# coding=utf-8
"""Attention backend utilities for WeDLM training."""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import magi_attention (optional)
_MAGI_AVAILABLE = False
try:
    from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
    _MAGI_AVAILABLE = True
except ImportError:
    flex_flash_attn_func = None


def check_backend_available(backend: str) -> bool:
    """Check if the specified attention backend is available."""
    if backend == "magi":
        return _MAGI_AVAILABLE
    elif backend == "dense":
        return True
    return False


def get_available_backend() -> str:
    """Get the best available backend."""
    if _MAGI_AVAILABLE:
        return "magi"
    return "dense"


class MagiAttentionWrapper(nn.Module):
    """Wrapper for Magi Flex Flash Attention."""
    
    def __init__(
        self, 
        head_dim: int, 
        softmax_scale: Optional[float] = None,
        deterministic: bool = False,  # 训练时使用False更快
    ):
        super().__init__()
        if not _MAGI_AVAILABLE:
            raise ImportError("magi_attention is not installed. Install with: pip install magi-attention")
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)
        self.deterministic = deterministic
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        magi_plan: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run Magi flex flash attention.
        
        Args:
            q: Query tensor [T, H, D]
            k: Key tensor [T, Hkv, D]
            v: Value tensor [T, Hkv, D]
            magi_plan: Dict containing q_ranges, k_ranges, attn_type_map, max_seqlen_q, max_seqlen_k
        """
        out, _ = flex_flash_attn_func(
            q if q.is_contiguous() else q.contiguous(),
            k if k.is_contiguous() else k.contiguous(),
            v if v.is_contiguous() else v.contiguous(),
            q_ranges=magi_plan["q_ranges"],
            k_ranges=magi_plan["k_ranges"],
            max_seqlen_q=magi_plan["max_seqlen_q"],
            max_seqlen_k=magi_plan["max_seqlen_k"],
            attn_type_map=magi_plan["attn_type_map"],
            softmax_scale=self.softmax_scale,
            softcap=0.0,
            deterministic=self.deterministic,
        )
        return out


class DenseAttentionWrapper(nn.Module):
    """Dense attention using PyTorch SDPA with 2D mask."""
    
    def __init__(self, head_dim: int, softmax_scale: Optional[float] = None):
        super().__init__()
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Run dense attention with 2D mask.
        
        Args:
            q: Query tensor [T, H, D]
            k: Key tensor [T, Hkv, D]
            v: Value tensor [T, Hkv, D]
            attn_mask_2d: Attention mask [T, T] bool, True=ALLOW
        """
        T, H_q, D = q.shape
        H_kv = k.shape[1]
        
        # GQA expansion
        if H_q != H_kv:
            expand_ratio = H_q // H_kv
            k = k.repeat_interleave(expand_ratio, dim=1)
            v = v.repeat_interleave(expand_ratio, dim=1)
        
        # Convert mask: True=ALLOW -> 0.0, False -> -inf
        sdpa_mask = torch.where(attn_mask_2d, 0.0, float("-inf")).to(dtype=q.dtype, device=q.device)
        sdpa_mask = sdpa_mask.unsqueeze(0).unsqueeze(0)
        
        # Reshape: [T, H, D] -> [1, H, T, D]
        q_b = q.permute(1, 0, 2).unsqueeze(0)
        k_b = k.permute(1, 0, 2).unsqueeze(0)
        v_b = v.permute(1, 0, 2).unsqueeze(0)
        
        out = F.scaled_dot_product_attention(
            q_b, k_b, v_b,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.softmax_scale,
        )
        
        return out.squeeze(0).permute(1, 0, 2).contiguous()


def get_attention_wrapper(
    backend: str, 
    head_dim: int, 
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
):
    """Get the appropriate attention wrapper.
    
    Args:
        backend: "magi" or "dense"
        head_dim: Dimension per attention head
        softmax_scale: Optional custom softmax scale
        deterministic: Whether to use deterministic operations (magi only)
    """
    if backend == "magi":
        return MagiAttentionWrapper(head_dim, softmax_scale, deterministic=deterministic)
    elif backend == "dense":
        return DenseAttentionWrapper(head_dim, softmax_scale)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: magi, dense")

