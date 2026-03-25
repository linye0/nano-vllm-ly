import math
import torch
import warnings
from typing import Optional, Union, Any

import custom_attention_ext

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    device = q.device
    
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise RuntimeError(f"Custom Kernel ONLY supports BF16! Expected torch.bfloat16, but got q={q.dtype}, k={k.dtype}.")

    q = q.contiguous()

    cu_seqlens_q = cu_seqlens_q.to(device)
    cu_seqlens_k = cu_seqlens_k.to(device)

    if dropout_p > 0.0:
        warnings.warn("Custom Flash Attention currently does not support dropout. Ignoring dropout_p.")
    if window_size != (-1, -1):
        warnings.warn("Custom Flash Attention currently does not support sliding window. Ignoring window_size.")
    if softcap > 0.0 or alibi_slopes is not None:
        raise NotImplementedError("softcap and alibi_slopes are not supported in the custom kernel.")
    if not causal:
        warnings.warn("Custom Flash Attention implicitly uses Causal Masking right now.")
    
    is_paged = block_table is not None
    if is_paged:
        num_blocks, block_size, num_kv_heads, _ = k.shape
        bt = block_table.to(device)
        max_blocks_per_seq = block_table.shape[1]
    else:
        total_k, num_kv_heads, _ = k.shape
        block_size = 256  
        max_blocks_per_seq = 0
        bt = torch.empty((0,), dtype=torch.int32, device=q.device) 
        k = k.contiguous()
        v = v.contiguous()

    total_q, num_heads, head_dim = q.shape

    if head_dim not in [64, 128]:
        raise ValueError(f"Custom Kernel only supports head_dim 64 or 128, but got {head_dim}")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # 预分配原生 bfloat16
    out = torch.empty_like(q)

    custom_attention_ext.run_custom_flash_attn_prefill(
         q, k, v, out, 
         cu_seqlens_q, cu_seqlens_k, bt, is_paged,
         softmax_scale, max_seqlen_q, max_seqlen_k,
         num_heads, num_kv_heads, block_size, max_blocks_per_seq
    )

    if return_attn_probs:
        return out, None, None
    
    return out

def flash_attn_with_kvcache(
    q, # (batch_size, seqlen, nheads, headdim), 在decode阶段，seqlen == 1
    k_cache, # (num_blocks, page_block_size, nheads_k, headdim)
    v_cache, # (num_blocks, page_block_size, nheads_k, headdim)
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False
):
    device = q.device

    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise RuntimeError(f"Custom Kernel ONLY supports BF16! Expected torch.bfloat16, but got q={q.dtype}, k_cache={k_cache.dtype}, v_cache={v_cache.dtype}.")
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if not causal:
        warnings.warn("Custom Flash Attention implicitly uses Causal Masking right now.")

    q = maybe_contiguous(q)
    
    total_q, _, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    max_blocks_per_seq = block_table.shape[1]

    if head_dim not in [64, 128]:
        raise ValueError(f"Custom Kernel only supports head_dim 64 or 128, but got {head_dim}.")

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if cache_seqlens is not None:
        if isinstance(cache_seqlens, int):
            cache_seqlens = torch.full((k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device)
        else:
            cache_seqlens = cache_seqlens.to(torch.int32)
        cache_seqlens = maybe_contiguous(cache_seqlens)
    else:
        raise ValueError("cache_seqlens cannot be None.")

    # 强制 block_table 为 int32
    if block_table is not None:
        block_table = block_table.to(torch.int32)
        block_table = maybe_contiguous(block_table)

    out = torch.empty_like(q)
    
    custom_attention_ext.run_custom_flash_attn_decode(
        q, k_cache, v_cache, out,
        cache_seqlens, block_table,
        softmax_scale,
        num_heads, num_kv_heads, block_size, max_blocks_per_seq
    )

    return out