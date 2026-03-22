import torch
from torch import nn
import triton
import triton.language as tl
import os

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.custom.custom_attention import flash_attn_varlen_func as custom_flash_attn_varlen_func

from nanovllm.utils.context import get_context

USE_CUSTOM_ATTN = os.environ.get("USE_CUSTOM_PREFILL", "0") == "1"

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0) # 一个idx对应一个token
    slot = tl.load(slot_mapping_ptr + idx) # slot是当前token对应的具体索引(精度不是以block为单位，而是以token为单位)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape # N是待处理的tokens数量，num_heads是注意力头的数量，head_dim是每个头的维度
    D = num_heads * head_dim 
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # k_cache和v_cache的shape应该是[num_blocks, block_size, num_kv_heads, head_dim]
    assert k_cache.stride(1) == D and v_cache.stride(1) == D 
    assert slot_mapping.numel() == N
    # 开启N个线程，每个线程对应一个token
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D) 

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache

            if USE_CUSTOM_ATTN:
                #print("[DEBUG] Routing to CUSTOM Flash Attention Kernel...")
                o = custom_flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale, causal=True, block_table=context.block_tables
                )
            else:
                # warmup或非paged阶段
                o = flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
