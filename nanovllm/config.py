import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    custom_kernel: bool = False
    chunked_prefill: bool = False
    prefill_chunk_size: int = 256

    def __post_init__(self):
        self.model = os.path.expanduser(self.model)
        if not os.path.isdir(self.model):
            raise FileNotFoundError(f"Model path not found: {self.model}")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be in [1, 8]")
        if self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if self.prefill_chunk_size > self.max_num_batched_tokens:
            raise ValueError("prefill_chunk_size cannot exceed max_num_batched_tokens")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if not self.chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("legacy prefill requires max_num_batched_tokens >= max_model_len")
        if self.custom_kernel:
            head_dim = getattr(self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads)
            if head_dim not in (64, 128):
                raise ValueError("custom attention supports head_dim 64 or 128 only")
