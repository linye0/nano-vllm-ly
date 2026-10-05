import unittest

import torch
try:
    from flash_attn import flash_attn_varlen_func as reference_varlen
    from flash_attn import flash_attn_with_kvcache as reference_decode
except ImportError:
    reference_varlen = reference_decode = None

from nanovllm.custom.custom_attention import (
    flash_attn_varlen_func as custom_varlen,
    flash_attn_with_kvcache as custom_decode,
)


@unittest.skipUnless(
    torch.cuda.is_available() and reference_varlen is not None,
    "CUDA and FlashAttention are required",
)
class CustomAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def assert_attention_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    def test_unpaged_varlen_gqa(self):
        lengths = [37, 127]
        total = sum(lengths)
        q = torch.randn(total, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(total, 2, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cu = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
        actual = custom_varlen(q, k, v, cu, cu, max(lengths), max(lengths), causal=True)
        expected = reference_varlen(q, k, v, cu, cu, max(lengths), max(lengths), causal=True)
        self.assert_attention_close(actual, expected)

    def test_unpaged_head_dim_128(self):
        length = 65
        q = torch.randn(length, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(length, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        actual = custom_varlen(q, k, v, cu, cu, length, length, causal=True)
        expected = reference_varlen(q, k, v, cu, cu, length, length, causal=True)
        self.assert_attention_close(actual, expected)

    def test_chunked_prefill_with_noncontiguous_pages(self):
        context_length, query_length = 301, 45
        num_heads, num_kv_heads, head_dim, block_size = 4, 2, 64, 256
        q = torch.randn(query_length, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        dense_k = torch.randn(context_length, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        dense_v = torch.randn_like(dense_k)
        k_cache = torch.zeros(4, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.zeros_like(k_cache)
        block_table = torch.tensor([[2, 0]], device="cuda", dtype=torch.int32)
        k_cache[2] = dense_k[:block_size]
        v_cache[2] = dense_v[:block_size]
        k_cache[0, : context_length - block_size] = dense_k[block_size:]
        v_cache[0, : context_length - block_size] = dense_v[block_size:]
        cu_q = torch.tensor([0, query_length], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, context_length], device="cuda", dtype=torch.int32)

        actual = custom_varlen(
            q, k_cache, v_cache, cu_q, cu_k, query_length, context_length,
            causal=True, block_table=block_table,
        )
        expected = reference_varlen(
            q, dense_k, dense_v, cu_q, cu_k, query_length, context_length, causal=True,
        )
        self.assert_attention_close(actual, expected)

    def test_paged_decode_varlen_gqa(self):
        lengths = [301, 127]
        batch, num_heads, num_kv_heads, head_dim, block_size = 2, 4, 2, 64, 256
        q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        k_cache = torch.zeros(6, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.zeros_like(k_cache)
        block_table = torch.tensor([[3, 1], [4, -1]], device="cuda", dtype=torch.int32)
        first_k = torch.randn(lengths[0], num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        first_v = torch.randn_like(first_k)
        second_k = torch.randn(lengths[1], num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        second_v = torch.randn_like(second_k)
        k_cache[3], v_cache[3] = first_k[:block_size], first_v[:block_size]
        k_cache[1, : lengths[0] - block_size] = first_k[block_size:]
        v_cache[1, : lengths[0] - block_size] = first_v[block_size:]
        k_cache[4, : lengths[1]], v_cache[4, : lengths[1]] = second_k, second_v
        cache_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)

        actual = custom_decode(q, k_cache, v_cache, cache_lengths, block_table, causal=True)
        expected = reference_decode(
            q, k_cache, v_cache, cache_seqlens=cache_lengths,
            block_table=block_table, causal=True,
        )
        self.assert_attention_close(actual, expected)

    def test_prefill_uses_current_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            q = torch.randn(64, 4, 64, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(64, 2, 64, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            cu = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
            actual = custom_varlen(q, k, v, cu, cu, 64, 64, causal=True)
            expected = reference_varlen(q, k, v, cu, cu, 64, 64, causal=True)
        stream.synchronize()
        self.assert_attention_close(actual, expected)

    def test_unsupported_mode_fails_explicitly(self):
        q = torch.randn(8, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
        with self.assertRaises(NotImplementedError):
            custom_varlen(q, k, v, cu, cu, 8, 8, causal=False)


if __name__ == "__main__":
    unittest.main()
