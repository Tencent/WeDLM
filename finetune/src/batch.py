# coding=utf-8
"""WeDLM batch construction."""

from typing import Dict, Optional
from dataclasses import dataclass
import torch

from src.masking import (
    sample_block_mask_ratios,
    sample_mask_indices,
    reorder_block,
    build_2d_attention_mask,
    build_magi_plan,
)


@dataclass
class WeDLMBatch:
    """Packed batch for WeDLM training."""
    packed_input_ids: torch.Tensor
    original_ids: torch.Tensor
    logical_positions: torch.Tensor
    masked_indices: torch.Tensor
    p_mask: torch.Tensor
    cum_seqlens: torch.Tensor
    base_cum_seqlens: torch.Tensor
    max_seqlen: int
    attn_mask_2d: Optional[torch.Tensor] = None
    magi_plan: Optional[Dict[str, torch.Tensor]] = None


@torch.no_grad()
def build_wedlm_batch(
    packed_input_ids: torch.Tensor,
    packed_labels: torch.Tensor,
    cum_seqlens: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    mask_per_block: bool = True,
    backend: str = "magi",
    eps: float = 1e-8,
) -> WeDLMBatch:
    """Build WeDLM batch with dual-stream masking."""
    device = packed_input_ids.device
    bs = cum_seqlens.numel() - 1
    B = block_size
    
    packed_parts, orig_parts, pos_parts = [], [], []
    mask_parts, p_parts, masks_2d = [], [], []
    new_cum, base_cum = [0], [0]
    
    for si in range(bs):
        seq_start, seq_end = cum_seqlens[si].item(), cum_seqlens[si + 1].item()
        L = seq_end - seq_start
        
        if L <= 0:
            new_cum.append(new_cum[-1])
            base_cum.append(base_cum[-1])
            continue
        
        seq = packed_input_ids[seq_start:seq_end]
        labels = packed_labels[seq_start:seq_end]
        prompt_mask = (labels == -100)
        positions = torch.arange(L, device=device, dtype=torch.long)
        nblk = (L + B - 1) // B
        
        p_blocks = sample_block_mask_ratios(nblk, mask_per_block, device, eps)
        
        xt_tokens, xt_orig, xt_pos, xt_mask, xt_p = [], [], [], [], []
        
        for b in range(nblk):
            blk_st, blk_ed = b * B, min(L, (b + 1) * B)
            tokens_blk = seq[blk_st:blk_ed]
            pos_blk = positions[blk_st:blk_ed]
            maskable = ~prompt_mask[blk_st:blk_ed]
            p_val = float(p_blocks[b].item())
            
            mask_indices = sample_mask_indices(maskable, p_val, device)
            xt_tok, orig_tok, pos_r, p_line = reorder_block(
                tokens_blk, pos_blk, mask_indices, p_val, mask_token_id
            )
            
            xt_tokens.append(xt_tok)
            xt_orig.append(orig_tok)
            xt_pos.append(pos_r)
            xt_mask.append(torch.cat([
                torch.zeros(((~mask_indices).sum(),), dtype=torch.bool, device=device),
                torch.ones((mask_indices.sum(),), dtype=torch.bool, device=device),
            ]))
            xt_p.append(p_line)
        
        xt_seq = torch.cat(xt_tokens)
        xt_orig_seq = torch.cat(xt_orig)
        xt_pos_seq = torch.cat(xt_pos)
        xt_mask_seq = torch.cat(xt_mask)
        xt_p_seq = torch.cat(xt_p)
        
        packed_parts.append(torch.cat([seq, xt_seq]))
        orig_parts.append(torch.cat([seq, xt_orig_seq]))
        pos_parts.append(torch.cat([positions, xt_pos_seq]))
        mask_parts.append(torch.cat([torch.zeros(L, dtype=torch.bool, device=device), xt_mask_seq]))
        p_parts.append(torch.cat([torch.zeros(L, device=device), xt_p_seq]))
        
        if backend == "dense":
            masks_2d.append(build_2d_attention_mask(L, B, device))
        
        new_cum.append(new_cum[-1] + 2 * L)
        base_cum.append(base_cum[-1] + L)
    
    def safe_cat(parts, dtype):
        return torch.cat(parts) if parts else torch.empty(0, dtype=dtype, device=device)
    
    packed_cum = torch.tensor(new_cum, device=device, dtype=torch.long)
    base_cum_t = torch.tensor(base_cum, device=device, dtype=torch.long)
    max_seqlen = (packed_cum[1:] - packed_cum[:-1]).max().item() if len(new_cum) > 1 else 0
    
    attn_mask_2d = torch.block_diag(*masks_2d).contiguous() if masks_2d else None
    magi_plan = build_magi_plan(base_cum_t, packed_cum, block_size, device) if backend == "magi" else None
    
    return WeDLMBatch(
        packed_input_ids=safe_cat(packed_parts, packed_input_ids.dtype),
        original_ids=safe_cat(orig_parts, packed_input_ids.dtype),
        logical_positions=safe_cat(pos_parts, torch.long),
        masked_indices=safe_cat(mask_parts, torch.bool),
        p_mask=safe_cat(p_parts, torch.float32),
        cum_seqlens=packed_cum,
        base_cum_seqlens=base_cum_t,
        max_seqlen=max_seqlen,
        attn_mask_2d=attn_mask_2d,
        magi_plan=magi_plan,
    )

