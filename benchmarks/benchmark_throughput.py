import argparse
import json
import platform
import random
import time
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Measure end-to-end generation throughput.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--min-input-len", type=int, default=128)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--min-output-len", type=int, default=32)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--custom-kernel", action="store_true")
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_input_len + args.max_output_len > args.max_model_len:
        raise ValueError("max input + output length must not exceed max_model_len")
    if min(args.num_seqs, args.min_input_len, args.min_output_len) <= 0:
        raise ValueError("sequence count and token lengths must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with LLM(
        args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        custom_kernel=args.custom_kernel,
        chunked_prefill=args.chunked_prefill,
        prefill_chunk_size=args.prefill_chunk_size,
        enforce_eager=args.enforce_eager,
    ) as llm:
        vocab_size = llm.config.hf_config.vocab_size
        prompts = [
            [random.randrange(vocab_size) for _ in range(random.randint(args.min_input_len, args.max_input_len))]
            for _ in range(args.num_seqs)
        ]
        sampling_params = [
            SamplingParams(
                temperature=0.6,
                ignore_eos=True,
                max_tokens=random.randint(args.min_output_len, args.max_output_len),
            )
            for _ in range(args.num_seqs)
        ]

        llm.generate([[1, 2, 3]], SamplingParams(max_tokens=2, ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    input_tokens = sum(map(len, prompts))
    output_tokens = sum(len(output["token_ids"]) for output in outputs)
    result = {
        "benchmark": "end_to_end_throughput",
        "model": str(args.model),
        "device": torch.cuda.get_device_name(),
        "software": {"python": platform.python_version(), "torch": torch.__version__},
        "config": {
            "num_seqs": args.num_seqs,
            "custom_kernel": args.custom_kernel,
            "chunked_prefill": args.chunked_prefill,
            "prefill_chunk_size": args.prefill_chunk_size,
            "seed": args.seed,
        },
        "metrics": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_s": elapsed,
            "output_throughput_tokens_per_s": output_tokens / elapsed,
            "total_throughput_tokens_per_s": (input_tokens + output_tokens) / elapsed,
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
