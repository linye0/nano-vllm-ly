# Architecture and invariants

## Request lifecycle

A request becomes a `Sequence` containing prompt tokens, sampling parameters, logical-to-physical
KV block mappings, and two independent progress counters:

- `num_prompt_tokens` is the amount of context that must be recomputed before decode can resume.
- `num_computed_tokens` is the amount of that context already represented by valid model state.
- `orig_prompt_len` never changes and is used only to separate user prompt tokens from generated
  output.

The separation matters during preemption. A decode sequence can lose its KV pages; its prompt for
recomputation then becomes the original prompt plus every token generated so far. Resetting only the
cache state while retaining `orig_prompt_len` preserves both correct recomputation and correct output
slicing.

```text
WAITING --allocate/chunk--> RUNNING --sample--> RUNNING
   ^                         |                    |
   |                         | preempt            | EOS / token limit
   +-------------------------+                    v
                                                FINISHED
```

## Chunked-prefill scheduler

Each scheduler step obeys two limits: `max_num_seqs` and `max_num_batched_tokens`.

1. Pop active decode sequences first. Each consumes one token from the step budget.
2. If a decode needs a new KV page, preempt a lower-priority running sequence until allocation can
   succeed.
3. Use the remaining token budget for waiting prompt work. One sequence receives at most
   `prefill_chunk_size` tokens in a step.
4. Requeue an incomplete prompt at the front of `waiting`; move a completed prompt to `running`.
5. Accept a sampled token only when the final prompt chunk has completed. Intermediate-chunk logits
   are not semantically next-token logits and are discarded.

The scheduler supports mixed batches: a packed prefill launch may contain one-token decode queries
and prompt chunks. For every sequence, `seqlen_k >= seqlen_q`; the difference is the cached prefix.
The attention kernel applies bottom-right-aligned causal masking so a chunk attends to all prior
context and only earlier positions inside the current chunk.

## Paged KV cache and prefix reuse

`BlockManager` owns fixed-size physical pages and reference counts. A sequence stores only its page
IDs. Full prompt blocks are hashed with the previous block hash, so a cache hit is valid only when
the entire prefix chain matches.

Important invariants:

- A page with `ref_count == 0` is in `free_block_ids`; a referenced page is in `used_block_ids`.
- Incremental chunk allocation creates only the pages needed by the new computed-token boundary.
- A reused page increments its reference count before it is shared.
- The final prompt block is never accepted as a complete prefix-cache hit. KV reuse alone cannot
  provide the final hidden state/logits required to sample the first completion token.
- Preemption releases all page references and resets computed/cache progress before recomputation.

## Attention backend

`Attention` first writes the current K/V tensors into paged cache slots, then dispatches one of two
backends:

| Phase | Official backend | Custom backend |
| --- | --- | --- |
| Initial or chunked prefill | `flash_attn_varlen_func` | WMMA variable-length prefill kernel |
| Decode | `flash_attn_with_kvcache` | partitioned paged-decode kernel + LSE reduction |

For a later prefill chunk, Q is packed from only the current chunk while K/V are read through the
sequence block table. Decode partitions long contexts across 256-token regions, computes partial
outputs and log-sum-exp statistics, then reduces partitions into the final BF16 output.

The PyTorch extension launches on `at::cuda::getCurrentCUDAStream`, guards the active device, checks
launch errors, and validates device/dtype/layout metadata at the C++ boundary. The Python adapter
documents and enforces the narrower feature contract before launch.

## Tensor parallelism

The rank-0 engine creates a serializable `Config` and passes it to spawned workers. Backend and
chunking choices are instance fields; no module-global feature flag is required. Per-step `Sequence`
serialization includes chunk progress so all ranks build identical packed Q/K layouts and block
tables.
