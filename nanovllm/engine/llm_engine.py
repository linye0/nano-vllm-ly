import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        unknown = set(kwargs) - config_fields
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown LLM configuration option(s): {names}")
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.scheduler = Scheduler(config)
        self._closed = False
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if len(prompt) > self.config.max_model_len:
            raise ValueError(
                f"prompt has {len(prompt)} tokens, exceeding max_model_len={self.config.max_model_len}"
            )
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError(
                "prompt plus requested output exceeds "
                f"max_model_len={self.config.max_model_len}"
            )
        seq = Sequence(prompt, sampling_params, block_size = self.config.kvcache_block_size)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()

        if not seqs:
            # 如果队列里还有东西，说明是死锁了，需要体面报错
            if self.scheduler.waiting or self.scheduler.running:
                print("\n[VRAM ALERT] Memory is too tight to schedule even one block. Wait or preemption may occur.")
                return [], 0
            return [], 0 # 正常结束

        if is_prefill:
            if self.config.chunked_prefill:
                num_tokens = sum(seq.cur_chunk_size for seq in seqs)
            else:
                num_tokens = sum(len(seq) - seq.num_cached_tokens for seq in seqs)
        else:
            num_tokens = -len(seqs)

        token_ids = self.model_runner.call("run", seqs, is_prefill)
        if self.config.chunked_prefill:
            self.scheduler.postprocess_chunked(seqs, token_ids) 
        else:
            self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def is_deadlock(self):
        return self.scheduler.is_deadlock()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if not prompts:
            return []
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        elif len(sampling_params) != len(prompts):
            raise ValueError("sampling_params must have the same length as prompts")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        if self.is_deadlock():
            raise RuntimeError("scheduler deadlock: insufficient KV-cache capacity")
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
