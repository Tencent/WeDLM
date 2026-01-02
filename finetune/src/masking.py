# coding=utf-8
"""Masking logic and attention mask building for WeDLM."""

from typing import Dict, Tuple
import torch


@torch.no_grad()
def sample_block_mask_ratios(
    num_blocks: int,
    mask_per_block: bool,
    device: torch.device,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sample masking ratios for each block."""
    if mask_per_block:
        return torch.empty(num_blocks, device=device).uniform_(0.0, 1.0).clamp_min(eps)
    else:
        p_val = torch.empty(1, device=device).uniform_(0.0, 1.0).clamp_min(eps).item()
        return torch.full((num_blocks,), p_val, device=device)


@torch.no_grad()
def sample_mask_indices(
    maskable: torch.Tensor,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample which positions to mask within maskable candidates."""
    mask_indices = torch.zeros_like(maskable, dtype=torch.bool)
    candidate_indices = torch.where(maskable)[0]
    num_candidates = candidate_indices.numel()
    
    if num_candidates > 0:
        num_to_mask = int(round(num_candidates * mask_ratio))
        if num_to_mask > 0:
            perm = torch.randperm(num_candidates, device=device)
            mask_indices[candidate_indices[perm[:num_to_mask]]] = True
    
    return mask_indices


@torch.no_grad()
def reorder_block(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_ratio: float,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder block: observed tokens first, then masked tokens."""
    device = tokens.device
    
    xt = torch.where(mask_indices, mask_token_id, tokens)
    unmasked = ~mask_indices
    
    xt_reordered = torch.cat([xt[unmasked], xt[mask_indices]])
    orig_reordered = torch.cat([tokens[unmasked], tokens[mask_indices]])
    pos_reordered = torch.cat([positions[unmasked], positions[mask_indices]])
    
    p_values = torch.cat([
        torch.zeros(unmasked.sum(), device=device),
        torch.full((mask_indices.sum(),), mask_ratio, device=device),
    ])
    
    return xt_reordered, orig_reordered, pos_reordered, p_values


@torch.no_grad()
def build_2d_attention_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Build WeDLM 2D attention mask for a single sequence."""
    if seq_len <= 0:
        return torch.ones(0, 0, dtype=torch.bool, device=device)
    
    L, B = seq_len, block_size
    total = 2 * L
    
    q_idx = torch.arange(total, device=device)[:, None]
    k_idx = torch.arange(total, device=device)[None, :]
    
    x0_q, x0_k = q_idx < L, k_idx < L
    xt_q, xt_k = ~x0_q, ~x0_k
    
    blk_q = torch.where(xt_q, (q_idx - L) // B, q_idx // B)
    blk_k = torch.where(xt_k, (k_idx - L) // B, k_idx // B)
    
    intra_block_causal = (blk_q == blk_k) & (xt_q == xt_k) & (k_idx <= q_idx)
    xt_to_x0_prev = (blk_k < blk_q) & xt_q & x0_k
    x0_cross_block = (blk_k < blk_q) & x0_q & x0_k
    
    return intra_block_causal | xt_to_x0_prev | x0_cross_block


@torch.no_grad()
def build_magi_plan(
    base_cum_seqlens: torch.Tensor,
    packed_cum_seqlens: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build Magi attention plan with precomputed max_seqlen."""
    bs = base_cum_seqlens.numel() - 1
    B = block_size
    
    q_ranges, k_ranges, attn_types = [], [], []
    max_seqlen_q, max_seqlen_k = 0, 0
    
    for si in range(bs):
        L = base_cum_seqlens[si + 1].item() - base_cum_seqlens[si].item()
        if L <= 0:
            continue
        
        pst = packed_cum_seqlens[si].item()
        x0_base, xt_base = pst, pst + L
        nblk = (L + B - 1) // B
        
        # xt blocks
        xt_blocks = [(xt_base + b * B, xt_base + min((b + 1) * B, L)) for b in range(nblk)]
        
        # xt intra-block causal
        for xt_s, xt_e in xt_blocks:
            q_len = xt_e - xt_s
            if q_len > 0:
                q_ranges.append([xt_s, xt_e])
                k_ranges.append([xt_s, xt_e])
                attn_types.append(1)
                max_seqlen_q = max(max_seqlen_q, q_len)
                max_seqlen_k = max(max_seqlen_k, q_len)
        
        # xt to x0 (previous blocks)
        for b, (xt_s, xt_e) in enumerate(xt_blocks):
            x0_k_e = x0_base + min(b * B, L)
            q_len = xt_e - xt_s
            k_len = x0_k_e - x0_base
            if q_len > 0 and k_len > 0:
                q_ranges.append([xt_s, xt_e])
                k_ranges.append([x0_base, x0_k_e])
                attn_types.append(0)
                max_seqlen_q = max(max_seqlen_q, q_len)
                max_seqlen_k = max(max_seqlen_k, k_len)
        
        # x0 causal
        x0_e = x0_base + L
        if L > 0:
            q_ranges.append([x0_base, x0_e])
            k_ranges.append([x0_base, x0_e])
            attn_types.append(1)
            max_seqlen_q = max(max_seqlen_q, L)
            max_seqlen_k = max(max_seqlen_k, L)
    
    if not q_ranges:
        return {
            "q_ranges": torch.zeros((0, 2), dtype=torch.int32, device=device),
            "k_ranges": torch.zeros((0, 2), dtype=torch.int32, device=device),
            "attn_type_map": torch.zeros((0,), dtype=torch.int32, device=device),
            "max_seqlen_q": 0,
            "max_seqlen_k": 0,
        }
    
    return {
        "q_ranges": torch.tensor(q_ranges, dtype=torch.int32, device=device).contiguous(),
        "k_ranges": torch.tensor(k_ranges, dtype=torch.int32, device=device).contiguous(),
        "attn_type_map": torch.tensor(attn_types, dtype=torch.int32, device=device).contiguous(),
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }

