import argparse
import json

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Measure peak allocated GPU memory for one request.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--custom-kernel", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prompt_tokens > args.max_model_len:
        raise ValueError("prompt_tokens must not exceed max_model_len")
    prompt = list(range(args.prompt_tokens))
    with LLM(
        args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        chunked_prefill=args.chunked_prefill,
        custom_kernel=args.custom_kernel,
        prefill_chunk_size=args.prefill_chunk_size,
    ) as llm:
        vocab_size = llm.config.hf_config.vocab_size
        prompt = [token % vocab_size for token in prompt]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        llm.generate([prompt], SamplingParams(max_tokens=1, ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
    print(json.dumps({
        "mode": "chunked" if args.chunked_prefill else "legacy",
        "custom_kernel": args.custom_kernel,
        "prompt_tokens": args.prompt_tokens,
        "peak_allocated_gib": peak / (1024 ** 3),
    }, indent=2))


if __name__ == "__main__":
    main()
