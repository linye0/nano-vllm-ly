import os
import argparse
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from nanovllm import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--custom_kernel", action="store_true", help="Use custom prefill kernel")
    parser.add_argument("--enforce_eager", action="store_true", default=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--chunked_prefill", action="store_true", help="Use custom chunked prefill")
    
    args = parser.parse_args()
    cfg = config.init_cfg(args)

    if cfg.custom_kernel:
        print("[INFO] Use CUSTOM kernel.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    # 初始化 LLM 引擎
    llm = LLM(cfg.model, enforce_eager=cfg.enforce_eager, tensor_parallel_size=cfg.tensor_parallel_size)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=10)
    
    base_text = "The quick brown fox jumps over the lazy dog. " * 2500
    
    prompts = [
        f"Please summarize the following text: {base_text}",
        f"What is the main idea of this story? {base_text}",
        f"Extract all the animals mentioned here: {base_text}",
        f"Rewrite this in a formal tone: {base_text}",
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    print(f"\n[INFO] Scale Target: ~27,500 Tokens per request.")
    print(f"[INFO] Total Batch: ~110,000 Tokens. (Fits in 6GB KV Cache, but kills Legacy Activations)\n")
    
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"Request {i} Completion: {output['text']!r}")

if __name__ == "__main__":
    main()