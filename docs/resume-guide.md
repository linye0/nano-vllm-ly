# Resume and interview guide

## Defensible resume bullets

Use measured numbers from your own final benchmark JSON in the brackets. Do not reuse an exploratory
number unless the exact command, commit, and environment are preserved.

Chinese:

- 基于 nano-vLLM 实现 token-budgeted Chunked Prefill 调度器：decode 优先、prefill 分片、混合批处理与 KV-cache
  抢占恢复，使用单元测试覆盖分片边界、预算约束、完整前缀命中和抢占状态机。
- 使用 CUDA C++/WMMA 编写 BF16 causal attention，支持变长 Prefill、Paged KV Chunk continuation、GQA 与
  分区 Decode reduction；在 RTX 3060 Laptop 上与 FlashAttention 做 differential test，测试形状最大绝对误差
  不超过 0.00390625。
- 将自定义算子接入 Qwen3 推理全链路并消除模块级全局配置，使单卡和 spawn tensor-parallel worker 共享显式
  backend/scheduler 配置；补齐 current-stream、device guard、launch check 和 unsupported-mode fail-fast。
- 建立可复现实验工具，分别度量 kernel latency、端到端 tokens/s、长 prompt 注入下的 p50/p95/p99 step latency
  与 peak memory；在固定工作负载下获得 `[填入最终结果]`。

English:

- Implemented token-budgeted chunked prefill in a compact vLLM-style engine, including decode-first
  mixed batching, KV-cache preemption/recomputation, and regression tests for scheduling and prefix-
  cache edge cases.
- Wrote BF16 CUDA/WMMA causal-attention kernels for variable-length prefill, paged chunk continuation,
  GQA, and partitioned decode; differential-tested against FlashAttention with <=0.00390625 maximum
  absolute error across the documented RTX 3060 test matrix.
- Integrated the backend through Qwen3 model execution and tensor-parallel spawn workers; added
  PyTorch current-stream semantics, device guards, launch checks, and explicit feature validation.
- Built reproducible micro/end-to-end benchmarks for kernel latency, generation throughput, long-
  prompt interference percentiles, and peak memory, measuring `[insert final result]` on `[hardware]`.

## Interview questions you should be ready to answer

### Why does chunked prefill help inter-token latency?

A monolithic long prefill occupies the GPU for one large engine step, delaying decode work already in
the system. Chunking caps prompt work per step and reserves scheduling priority for one-token decode.
It trades additional launches and potentially worse long-request TTFT for a smaller interference
window and more predictable decode step latency.

### Why is the final cached prompt block recomputed?

The prefix cache stores K/V, not the final hidden state or logits. If every prompt token is skipped,
the model has no query token from which to produce next-token logits. Recomputing the last block
retains most prefix reuse while guaranteeing a valid sampling position.

### What changes for causal masking in a later prompt chunk?

Q covers only the new chunk, while K covers cached prefix plus the chunk. Causality is bottom-right
aligned: query row `i` can see keys through `cached_prefix + i`. A square causal mask applied only to
Q length would be wrong.

### Why partition decode?

One decode query may attend to a long paged context. Independent blocks compute stable partial
softmax statistics and weighted values for context partitions. A second kernel combines them using
log-sum-exp, exposing parallelism without materializing the score matrix.

### Why can BF16 results differ from FlashAttention?

The two kernels can use different tiling, reduction trees, accumulation order, and intermediate
rounding. Differential testing therefore uses absolute/relative tolerances and also reports raw max
and mean error. Correctness claims must remain limited to tested dtypes, masks, shapes, and layouts.

### What would you add for production serving?

An async API and request cancellation, stronger admission control, scheduler fairness/aging,
continuous metrics and tracing, deterministic/greedy decoding options, quantization, broader model
coverage, multi-node collectives, failure recovery, fuzz/property tests, and GPU CI across compute
capabilities.
