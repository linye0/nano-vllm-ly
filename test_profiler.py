import torch
import argparse
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from nanovllm import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--chunked_prefill", action="store_true")
    args = parser.parse_args()
    cfg = config.init_cfg(args)

    llm = LLM(cfg.model, enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    
    # 构造一个 10,000 Token 的单条请求
    # 这个量级足够产生明显的激活值，但又不会撑爆 6GB 的 KV Cache 存储
    prompt = "The quick brown fox jumps over the lazy dog. " * 1000
    sampling_params = SamplingParams(max_tokens=1)

    # 重置显存峰值统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"\n[PROFILER] Running {'CHUNKED' if cfg.chunked_prefill else 'LEGACY'} mode...")
    
    llm.generate([prompt], sampling_params)
    
    # 获取本次运行的显存峰值
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[PROFILER] Peak VRAM Usage: {peak_memory:.3f} GB")

if __name__ == "__main__":
    main()