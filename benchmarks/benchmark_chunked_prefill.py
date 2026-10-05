import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


def parse_args():
    parser = argparse.ArgumentParser(description="Measure decode interference from an arriving long prompt.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--custom-kernel", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--short-output-tokens", type=int, default=24)
    parser.add_argument("--long-prompt-tokens", type=int, default=2048)
    parser.add_argument("--inject-step", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main():
    args = parse_args()
    if args.long_prompt_tokens > args.max_model_len:
        raise ValueError("long_prompt_tokens must not exceed max_model_len")
    if not 1 < args.inject_step <= args.short_output_tokens:
        raise ValueError("inject_step must be after step 1 and no later than short_output_tokens")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    short_tokens = tokenizer.encode("Explain paged KV cache in one paragraph.")
    seed_tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    repeats = (args.long_prompt_tokens + len(seed_tokens) - 1) // len(seed_tokens)
    long_tokens = (seed_tokens * repeats)[: args.long_prompt_tokens]

    rows = []
    with LLM(
        args.model,
        enforce_eager=True,
        custom_kernel=args.custom_kernel,
        chunked_prefill=args.chunked_prefill,
        prefill_chunk_size=args.prefill_chunk_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
    ) as llm:
        # Exclude first-use library initialization and allocator effects from
        # the measured request timeline.
        warmup_length = min(args.max_model_len, args.prefill_chunk_size + 1)
        vocab_size = llm.config.hf_config.vocab_size
        warmup_short = Sequence(
            [1, 2, 3],
            SamplingParams(max_tokens=4, ignore_eos=True),
            block_size=llm.config.kvcache_block_size,
        )
        warmup_long = Sequence(
            [token % vocab_size for token in range(1, warmup_length + 1)],
            SamplingParams(max_tokens=1, ignore_eos=True),
            block_size=llm.config.kvcache_block_size,
        )
        llm.scheduler.add(warmup_short)
        llm.step()
        llm.step()
        llm.scheduler.add(warmup_long)
        while not llm.scheduler.is_finished():
            llm.step()
        torch.cuda.synchronize()
        torch.manual_seed(args.seed)
        short = Sequence(
            short_tokens,
            SamplingParams(max_tokens=args.short_output_tokens, ignore_eos=True),
            block_size=llm.config.kvcache_block_size,
        )
        long = Sequence(
            long_tokens,
            SamplingParams(max_tokens=1, ignore_eos=True),
            block_size=llm.config.kvcache_block_size,
        )
        llm.scheduler.add(short)
        step = 1
        while not llm.scheduler.is_finished():
            if step == args.inject_step:
                llm.scheduler.add(long)
            torch.cuda.synchronize()
            started = time.perf_counter()
            _, scheduled_tokens = llm.step()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "step": step,
                    "latency_ms": elapsed_ms,
                    "scheduled_tokens": scheduled_tokens,
                    "short_generated_tokens": len(short.completion_token_ids),
                    "long_computed_tokens": long.num_computed_tokens if step >= args.inject_step else 0,
                }
            )
            step += 1
            if step > args.short_output_tokens + args.inject_step + args.long_prompt_tokens + 16:
                raise RuntimeError("benchmark exceeded its safety step limit")

    latencies = [row["latency_ms"] for row in rows]
    summary = {
        "mode": "chunked" if args.chunked_prefill else "legacy",
        "custom_kernel": args.custom_kernel,
        "long_prompt_tokens": len(long_tokens),
        "prefill_chunk_size": args.prefill_chunk_size,
        "steps": len(rows),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies),
            "injection_step": rows[args.inject_step - 1]["latency_ms"],
        },
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
