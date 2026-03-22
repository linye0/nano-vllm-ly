import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
import argparse
from nanovllm import config
# from vllm import LLM, SamplingParams


def main():
    # 1. 设置参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--custom_prefill", action="store_true")
    # 注意：bench.py 原本 enforce_eager 为 False
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--num_seqs", type=int, default=256)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    args = parser.parse_args()

    # 2. 初始化全局配置
    cfg = config.init_cfg(args)
    if cfg.custom_prefill:
        print("[INFO] Benchmarking with CUSTOM prefill kernel")

    seed(0)
    num_seqs = args.num_seqs
    max_input_len = 1024
    max_ouput_len = 1024

    # 3. 使用 cfg 里的参数
    llm = LLM(cfg.model, enforce_eager=cfg.enforce_eager, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
