# Benchmark protocol

## General rules

Performance numbers are useful only when the workload and environment are reproducible.

1. Run each alternative in a fresh process so CUDA Graphs, allocator state, and KV cache contents do
   not leak between configurations.
2. Pin model, dtype, GPU power mode, input/output length distribution, request count, random seed,
   eager/graph mode, token budget, and KV-cache utilization.
3. Warm up before timing and synchronize CUDA around host-side timers.
4. Run at least five trials for resume/public claims. Report median and p95 (or the full JSON), not
   only the best trial.
5. Separate kernel microbenchmarks from end-to-end engine throughput. A kernel speedup is not an
   end-to-end speedup unless the whole-model benchmark demonstrates it.

Record `nvidia-smi`, GPU name, driver, CUDA toolkit, PyTorch, FlashAttention, model revision, and Git
commit with published results.

## Attention microbenchmark

```bash
python -m benchmarks.benchmark_attention \
  --prefill-length 1024 --decode-context 1024 \
  --num-heads 16 --num-kv-heads 8 --head-dim 64 \
  --warmup 10 --repeats 100
```

This reports average CUDA-event latency and numerical error for custom versus official
FlashAttention. Sweep representative sequence lengths rather than quoting one favorable shape.

## End-to-end throughput

```bash
python -m benchmarks.benchmark_throughput \
  --model ~/huggingface/Qwen3-0.6B --num-seqs 32 \
  --min-input-len 128 --max-input-len 512 \
  --min-output-len 32 --max-output-len 128 \
  --output benchmarks/results/baseline.json

python -m benchmarks.benchmark_throughput \
  --model ~/huggingface/Qwen3-0.6B --num-seqs 32 \
  --min-input-len 128 --max-input-len 512 \
  --min-output-len 32 --max-output-len 128 \
  --custom-kernel --output benchmarks/results/custom.json
```

The script reports input tokens, actual generated tokens, elapsed time, output-token throughput, and
total-token throughput. Keep CUDA Graph mode identical on both sides.

## Chunked-prefill interference

The latency benchmark begins decoding a short request and injects a long prompt at a fixed step.
Run legacy and chunked policies separately:

```bash
python -m benchmarks.benchmark_chunked_prefill \
  --long-prompt-tokens 2048 --inject-step 4 \
  --output benchmarks/results/legacy.csv

python -m benchmarks.benchmark_chunked_prefill \
  --long-prompt-tokens 2048 --inject-step 4 --chunked-prefill \
  --prefill-chunk-size 256 --output benchmarks/results/chunked.csv
```

The CSV retains every step, including scheduled-token count and long-request progress. The adjacent
summary JSON reports mean, p50, p95, p99, and maximum step latency. The key tradeoff is expected:
smaller chunks reduce individual interference spikes but may add scheduling/kernel overhead and
increase the long request's time-to-first-token.

## Memory

```bash
python -m benchmarks.benchmark_memory --prompt-tokens 2048
python -m benchmarks.benchmark_memory --prompt-tokens 2048 --chunked-prefill
```

This uses PyTorch's peak allocated-memory counter. It is not total process VRAM; pair it with
`nvidia-smi` if reporting allocator reservations or complete device occupancy.
