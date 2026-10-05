import argparse
import json
from pathlib import Path

import torch
from flash_attn import flash_attn_varlen_func as reference_varlen
from flash_attn import flash_attn_with_kvcache as reference_decode

from nanovllm.custom.custom_attention import flash_attn_varlen_func as custom_varlen
from nanovllm.custom.custom_attention import flash_attn_with_kvcache as custom_decode


def parse_args():
    parser = argparse.ArgumentParser(description="Compare the custom BF16 attention kernels with FlashAttention.")
    parser.add_argument("--prefill-length", type=int, default=1024)
    parser.add_argument("--decode-context", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def measure(callable_, warmup, repeats):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def error_metrics(actual, expected):
    error = (actual.float() - expected.float()).abs()
    return {"max_abs": error.max().item(), "mean_abs": error.mean().item()}


def main():
    args = parse_args()
    if args.num_heads % args.num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if args.decode_context > args.block_size * 16:
        raise ValueError("decode_context exceeds the benchmark's 16-page block table")
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(args.prefill_length, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.prefill_length, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    cu = torch.tensor([0, args.prefill_length], device=device, dtype=torch.int32)
    custom_prefill = lambda: custom_varlen(
        q, k, v, cu, cu, args.prefill_length, args.prefill_length, causal=True
    )
    reference_prefill = lambda: reference_varlen(
        q, k, v, cu, cu, args.prefill_length, args.prefill_length, causal=True
    )
    prefill_actual = custom_prefill()
    prefill_expected = reference_prefill()

    pages = (args.decode_context + args.block_size - 1) // args.block_size
    q_decode = torch.randn(1, 1, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(pages, args.block_size, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    cache_lengths = torch.tensor([args.decode_context], device=device, dtype=torch.int32)
    block_table = torch.arange(pages - 1, -1, -1, device=device, dtype=torch.int32).unsqueeze(0)
    custom_decode_call = lambda: custom_decode(
        q_decode, k_cache, v_cache, cache_lengths, block_table, causal=True
    )
    reference_decode_call = lambda: reference_decode(
        q_decode,
        k_cache,
        v_cache,
        cache_seqlens=cache_lengths,
        block_table=block_table,
        causal=True,
    )
    decode_actual = custom_decode_call()
    decode_expected = reference_decode_call()

    result = {
        "device": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "prefill": {
            "shape": [args.prefill_length, args.num_heads, args.num_kv_heads, args.head_dim],
            "custom_ms": measure(custom_prefill, args.warmup, args.repeats),
            "flash_attn_ms": measure(reference_prefill, args.warmup, args.repeats),
            "error": error_metrics(prefill_actual, prefill_expected),
        },
        "decode": {
            "shape": [args.decode_context, args.num_heads, args.num_kv_heads, args.head_dim],
            "custom_ms": measure(custom_decode_call, args.warmup, args.repeats),
            "flash_attn_ms": measure(reference_decode_call, args.warmup, args.repeats),
            "error": error_metrics(decode_actual, decode_expected),
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
