# Correctness strategy

The project validates policy/state logic separately from numerical CUDA behavior. This keeps
scheduler regressions cheap to diagnose and makes kernel comparisons independent of model sampling.

## Scheduler and cache tests

Run:

```bash
python -m unittest tests.test_scheduler -v
```

The suite covers:

- a 300-token prompt advancing as `128 + 128 + 44` and sampling only on the final chunk;
- decode-first ordering and enforcement of the per-step token budget;
- a mixed batch containing decode and prefill work;
- preempted decode recomputing original prompt plus generated context;
- accounting of remaining generation tokens after preemption;
- a full prompt prefix-cache hit recomputing its final block to produce logits;
- `Sequence` chunk state surviving multiprocessing serialization.

## CUDA differential tests

Run after building `nanovllm/custom`:

```bash
python -m unittest tests.test_custom_attention -v
```

Official FlashAttention is the numerical reference. Inputs are generated with a fixed seed and the
same packed offsets, paged block tables, cache lengths, scale, and causal mode are passed to both
implementations.

| Case | Q heads / KV heads | Head dim | Lengths | What it checks |
| --- | ---: | ---: | --- | --- |
| Varlen prefill | 4 / 2 | 64 | 37, 127 | packing, causal mask, GQA |
| Paged chunk prefill | 4 / 2 | 64 | Q=45, K=301 | causal offset, two non-contiguous pages |
| Paged decode | 4 / 2 | 64 | 301, 127 | variable cache lengths and partition reduction |
| Non-default stream | 4 / 2 | 64 | 64 | PyTorch current-stream semantics |

The comparison uses `torch.testing.assert_close(..., atol=0.05, rtol=0.05)`. On the RTX 3060 Laptop
development machine, observed maximum absolute errors were:

- unpaged varlen prefill: `0.00390625`;
- paged chunk prefill: `0.001953125`;
- paged decode: `0.0009765625`.

These are empirical BF16 results, not a guarantee for untested shapes. New head dimensions, masks,
dtypes, or cache layouts must add a differential case before being advertised.

## End-to-end smoke test

The model path checks integration that isolated kernel tests cannot: rotary positions, KV slot
mapping, model-layer routing, sampling, and scheduler transitions.

```bash
python example.py --model ~/huggingface/Qwen3-0.6B --enforce-eager
python example.py --model ~/huggingface/Qwen3-0.6B \
  --enforce-eager --chunked-prefill --custom-kernel --max-tokens 8
```

For deterministic regression comparisons, run both configurations in fresh processes with the same
`torch.manual_seed`, prompt token IDs, and eager mode. Compare generated token IDs as well as output
text; sampling text alone is not a kernel correctness test.
