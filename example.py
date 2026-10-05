import argparse

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small nano-vLLM generation example.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--custom-kernel", action="store_true")
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = ["Introduce yourself.", "List all prime numbers below 100."]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    with LLM(
        args.model,
        custom_kernel=args.custom_kernel,
        chunked_prefill=args.chunked_prefill,
        prefill_chunk_size=args.prefill_chunk_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    ) as llm:
        outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=args.max_tokens))

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
