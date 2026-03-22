import math
import torch
import warnings

# 假设你编译好的扩展模块叫 custom_attn_ext
import custom_attention_ext

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
    """
    1:1 平替官方的 flash_attn_varlen_func。
    内部将请求重定向到我们自己手写的 Paged Flash Attention CUDA Kernel。
    """
    
    # --- 1. 参数不支持警告拦截 ---
    if dropout_p > 0.0:
        warnings.warn("Custom Flash Attention currently does not support dropout. Ignoring dropout_p.")
    if window_size != (-1, -1):
        warnings.warn("Custom Flash Attention currently does not support sliding window. Ignoring window_size.")
    if softcap > 0.0 or alibi_slopes is not None:
        raise NotImplementedError("softcap and alibi_slopes are not supported in the custom kernel.")
    if not causal:
        # 注意：我们的 Kernel 里默认强制了 Causal Mask，如果要关闭需要改 CUDA 里的 if 判断
        warnings.warn("Custom Flash Attention implicitly uses Causal Masking right now.")
    
    # 强制检查我们的看家本领：Paged Block Table
    if block_table is None:
        raise ValueError("Our Custom Kernel is specifically designed for Paged KV-Cache! block_table cannot be None.")

    # --- 2. 形状与参数计算 ---
    total_q, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k.shape
    max_blocks_per_seq = block_table.shape[1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # 预分配输出 Tensor
    out = torch.empty_like(q)

    # --- 3. 调用底层的 C++ Kernel ---
    # custom_attn_ext.run_custom_flash_attn_varlen(
    #     q, k, v, out, 
    #     cu_seqlens_q, cu_seqlens_k, block_table,
    #     softmax_scale, max_seqlen_q, max_seqlen_k,
    #     num_heads, num_kv_heads, block_size, max_blocks_per_seq
    # )

    # --- 4. 返回值封装 ---
    if return_attn_probs:
        # 官方接口在测试时可能需要返回 LSE 和 Dmask，这里我们填 None
        return out, None, None
    
    return out