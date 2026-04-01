import time
import argparse
import csv  # 新增
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--chunked_prefill", action="store_true")
    # ... 其他参数保持不变 ...
    args = parser.parse_args()
    cfg = config.init_cfg(args)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    llm = LLM(cfg.model, enforce_eager=True)
    
    short_prompt = "Hello, please count from 1 to 100: 1, 2, 3,"
    short_tokens = tokenizer.encode(short_prompt)
    seq_short = Sequence(short_tokens, SamplingParams(max_tokens=50), block_size=cfg.kvcache_block_size)
    
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 1400 
    long_tokens = tokenizer.encode(long_prompt)
    seq_long = Sequence(long_tokens, SamplingParams(max_tokens=1), block_size=cfg.kvcache_block_size)
    
    # 数据存储结构
    results = []
    
    llm.scheduler.add(seq_short)
    step_num = 1
    while not llm.scheduler.is_finished():
        t0 = time.perf_counter() # 使用高精度时钟
        outputs, _ = llm.step()
        t1 = time.perf_counter()
        
        step_ms = (t1 - t0) * 1000
        
        short_status = "Prefill" if not seq_short.is_prefill_finished else "Decode"
        long_status = "Inactive" if step_num < 4 else ("Chunking" if not seq_long.is_prefill_finished else "Finished")
        
        # 记录数据
        results.append({
            "step": step_num,
            "latency_ms": step_ms,
            "short_status": short_status,
            "long_status": long_status
        })
        
        print(f"[Step {step_num:02d}] Latency: {step_ms:.2f}ms")

        if step_num == 3:
            llm.scheduler.add(seq_long)
            
        step_num += 1
        if step_num > 40: break

    # 保存至 CSV
    csv_file = "latency_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "latency_ms", "short_status", "long_status"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[数据已存入 {csv_file}]")

if __name__ == "__main__":
    main()