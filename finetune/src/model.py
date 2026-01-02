# coding=utf-8
"""Model forward pass with WeDLM attention."""

import torch
import torch.nn as nn
from src.batch import WeDLMBatch


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings."""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def wedlm_attention_forward(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch: WeDLMBatch,
    attn_wrapper: nn.Module,
    backend: str,
) -> torch.Tensor:
    """Forward through attention with WeDLM mask."""
    q_len = hidden_states.size(0)
    head_dim = attn_module.head_dim
    num_heads = attn_module.config.num_attention_heads
    num_kv_heads = attn_module.config.num_key_value_heads
    
    q = attn_module.q_proj(hidden_states).view(q_len, num_heads, head_dim)
    k = attn_module.k_proj(hidden_states).view(q_len, num_kv_heads, head_dim)
    v = attn_module.v_proj(hidden_states).view(q_len, num_kv_heads, head_dim)
    
    if attn_module.qk_norm:
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)
    
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    
    if backend == "magi" and batch.magi_plan is not None:
        out = attn_wrapper(q, k, v, batch.magi_plan)
    elif batch.attn_mask_2d is not None:
        out = attn_wrapper(q, k, v, batch.attn_mask_2d)
    else:
        raise ValueError("No valid attention mask available")
    
    return attn_module.o_proj(out.reshape(q_len, -1))


def wedlm_forward(
    model: nn.Module,
    batch: WeDLMBatch,
    attn_wrapper: nn.Module,
    backend: str,
) -> torch.Tensor:
    """Full forward pass with WeDLM attention pattern."""
    base_model = model.model if hasattr(model, "model") else model
    lm_head = model.lm_head
    
    hidden_states = base_model.embed_tokens(batch.packed_input_ids)
    
    position_ids = batch.logical_positions.unsqueeze(0)
    cos, sin = base_model.rotary_emb(hidden_states.unsqueeze(0), position_ids)
    cos, sin = cos.squeeze(0), sin.squeeze(0)
    
    for layer in base_model.layers:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states = wedlm_attention_forward(
            layer.self_attn, hidden_states, cos, sin, batch, attn_wrapper, backend
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
    
    hidden_states = base_model.norm(hidden_states)
    return lm_head(hidden_states)

