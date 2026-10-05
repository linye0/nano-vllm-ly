# Validated RTX 3060 snapshot

This page records a local validation snapshot, not a universal performance claim. Results were
collected on 2026-10-05 from an NVIDIA GeForce RTX 3060 Laptop GPU with Python 3.10.19, PyTorch
2.5.1+cu121, FlashAttention 2.8.3, Qwen3-0.6B, and the repository working tree containing the fixes
documented in this project. Each table is one run after explicit path warmup; use multiple trials
before quoting numbers externally.

## Kernel microbenchmark

Command:

```bash
python -m benchmarks.benchmark_attention \
  --prefill-length 512 --decode-context 512 \
  --num-heads 16 --num-kv-heads 8 --head-dim 64 \
  --warmup 10 --repeats 100
```

| Phase | Custom CUDA | FlashAttention | Custom numerical error |
| --- | ---: | ---: | ---: |
| Prefill | 0.1757 ms | 0.1843 ms | max abs 0.00390625 |
| Decode | 0.1084 ms | 0.0421 ms | max abs 0.0009765625 |

The custom kernel is competitive on this prefill shape but is substantially slower for one-token
decode. This identifies partition/reduction overhead in the custom decode path as optimization work;
it is not presented as a blanket speedup.

## End-to-end backend comparison

Both runs used eager mode, 8 requests, 128–256 input tokens, 16–32 output tokens, seed 0, and the
same 1,473 input / 203 generated tokens.

| Backend | Output throughput | Total-token throughput | Elapsed |
| --- | ---: | ---: | ---: |
| FlashAttention | 117.10 tok/s | 966.82 tok/s | 1.7335 s |
| Custom CUDA | 114.82 tok/s | 947.99 tok/s | 1.7680 s |

For this small mixed workload the custom backend is about 1.9% slower in output-token throughput.
The earlier exploratory “86.4% faster” claim was removed because it was not supported by the
reproducible harness and current differential measurements.

## Chunked-prefill interference

The scheduler comparison used the official FlashAttention backend on both sides, eager mode, the
same token budget/model length, and a short decode request with a long request injected at step 3.

| Long prompt | Policy | Chunk | Injection step | Maximum step | Steps |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | Legacy | n/a | 38.83 ms | 40.27 ms | 9 |
| 512 | Chunked | 64 | 45.35 ms | 64.13 ms | 10 |
| 3,584 | Legacy | n/a | 261.36 ms | 261.36 ms | 9 |
| 3,584 | Chunked | 256 | 44.27 ms | 61.39 ms | 16 |

Interpretation:

- At 512 tokens, chunking adds overhead and does not improve the latency-sensitive request.
- At 3,584 tokens, chunking reduces the injection-step stall by approximately 83.1% and the maximum
  measured step by approximately 76.5%.
- The long request spans more engine steps, so its own time-to-first-token can increase. Chunk size
  is an SLA/throughput trade-off, not a free speedup.

These results are deliberately reported with the unfavorable case: it defines when the scheduling
policy is useful and keeps resume claims technically defensible.
