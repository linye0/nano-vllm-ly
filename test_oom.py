import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from nanovllm import config
import argparse

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
        print("[INFO] use CUSTOM kernel.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    # 初始化 LLM 引擎，此时系统会根据 6GB 显存分配固定的 KV Cache 物理块
    llm = LLM(cfg.model, enforce_eager=cfg.enforce_eager, tensor_parallel_size=cfg.tensor_parallel_size)

    # 我们不需要它生成很长，我们要的是它在 Prefill 阶段就立刻暴毙
    sampling_params = SamplingParams(temperature=0.6, max_tokens=10)
    
    # 制造“挤爆”显存的极端输入：单条请求约 10,000 个单词
    # Qwen-0.6B 的隐藏层较小，但 4 个 10k 长度的请求并发，足以击穿 6GB 显存
    base_text = "The quick brown fox jumps over the lazy dog. " * 15000 
    
    # 构造并发 Batch
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
    
    print(f"\n[WARNING] Ready to send {len(prompts)} long requests...")
    print(f"[WARNING] Each has 15,000 Tokens.")
    
    # 这里将触发灾难
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Completion: {output['text']!r}")

if __name__ == "__main__":
    main()