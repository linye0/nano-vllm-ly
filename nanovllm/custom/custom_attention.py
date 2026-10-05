import math
import torch
from typing import Optional, Union

try:
    import custom_attention_ext
except ImportError as exc:
    custom_attention_ext = None
    _EXTENSION_IMPORT_ERROR = exc
else:
    _EXTENSION_IMPORT_ERROR = None


def _require_extension():
    if custom_attention_ext is None:
        raise RuntimeError(
            "custom attention extension is not installed; run "
            "`python -m pip install -e nanovllm/custom`"
        ) from _EXTENSION_IMPORT_ERROR

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
    _require_extension()
    device = q.device
    
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise RuntimeError(f"Custom kernel requires BF16 Q/K/V, got {q.dtype}/{k.dtype}/{v.dtype}.")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("Custom kernel requires CUDA Q/K/V tensors.")
    if not (q.device == k.device == v.device):
        raise RuntimeError("Q/K/V must be on the same CUDA device.")
    if q.ndim != 3 or k.ndim not in (3, 4) or v.shape != k.shape:
        raise ValueError("prefill expects Q=[tokens, heads, dim] and matching 3D or 4D K/V")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K/V head dimensions must match")

    q = q.contiguous()

    cu_seqlens_q = cu_seqlens_q.to(device=device, dtype=torch.int32).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(device=device, dtype=torch.int32).contiguous()

    if dropout_p > 0.0:
        raise NotImplementedError("dropout is not supported by the custom kernel")
    if window_size != (-1, -1):
        raise NotImplementedError("sliding-window attention is not supported by the custom kernel")
    if softcap > 0.0 or alibi_slopes is not None:
        raise NotImplementedError("softcap and alibi_slopes are not supported in the custom kernel.")
    if not causal:
        raise NotImplementedError("the custom kernel currently supports causal attention only")
    
    is_paged = block_table is not None
    if is_paged:
        num_blocks, block_size, num_kv_heads, _ = k.shape
        bt = block_table.to(device=device, dtype=torch.int32).contiguous()
        max_blocks_per_seq = block_table.shape[1]
    else:
        total_k, num_kv_heads, _ = k.shape
        block_size = 256  
        max_blocks_per_seq = 0
        bt = torch.empty((0,), dtype=torch.int32, device=q.device) 
        k = k.contiguous()
        v = v.contiguous()

    total_q, num_heads, head_dim = q.shape
    if num_heads % num_kv_heads:
        raise ValueError("the number of query heads must be divisible by KV heads")

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
        raise NotImplementedError("return_attn_probs is not supported by the custom kernel")
    
    return out

def flash_attn_with_kvcache(
    q, # (batch_size, seqlen, nheads, headdim), 在decode阶段，seqlen == 1
    k_cache, # (num_blocks, page_block_size, nheads_k, headdim)
    v_cache, # (num_blocks, page_block_size, nheads_k, headdim)
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False
):
    _require_extension()
    device = q.device

    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise RuntimeError(f"Custom Kernel ONLY supports BF16! Expected torch.bfloat16, but got q={q.dtype}, k_cache={k_cache.dtype}, v_cache={v_cache.dtype}.")
    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise RuntimeError("Custom decode requires CUDA Q/K/V tensors.")
    if not (q.device == k_cache.device == v_cache.device):
        raise RuntimeError("Q/K/V must be on the same CUDA device.")
    if q.ndim != 4 or q.shape[1] != 1 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("decode expects Q=[batch, 1, heads, dim] and matching paged 4D K/V")
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("Q and K/V head dimensions must match")
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise RuntimeError("K/V cache must have a contiguous last dimension")
    if not causal:
        raise NotImplementedError("the custom decode kernel currently supports causal attention only")
    if block_table is None:
        raise ValueError("block_table is required for paged decode")

    q = maybe_contiguous(q)
    
    total_q, _, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    max_blocks_per_seq = block_table.shape[1]
    if num_heads % num_kv_heads:
        raise ValueError("the number of query heads must be divisible by KV heads")

    if head_dim not in [64, 128]:
        raise ValueError(f"Custom Kernel only supports head_dim 64 or 128, but got {head_dim}.")

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if cache_seqlens is not None:
        if isinstance(cache_seqlens, int):
            cache_seqlens = torch.full((q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device)
        else:
            cache_seqlens = cache_seqlens.to(device=device, dtype=torch.int32)
        cache_seqlens = maybe_contiguous(cache_seqlens)
    else:
        raise ValueError("cache_seqlens cannot be None.")

    # 强制 block_table 为 int32
    if block_table is not None:
        block_table = block_table.to(device=device, dtype=torch.int32)
        block_table = maybe_contiguous(block_table)

    out = torch.empty_like(q)
    
    custom_attention_ext.run_custom_flash_attn_decode(
        q, k_cache, v_cache, out,
        cache_seqlens, block_table,
        softmax_scale,
        num_heads, num_kv_heads, block_size, max_blocks_per_seq
    )

    return out
