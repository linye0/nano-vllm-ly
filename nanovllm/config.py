import os
from dataclasses import dataclass
from transformers import AutoConfig
from typing import Optional
import sys

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

    def __post_init__(self):
        self.model = os.path.expanduser(self.model)
        assert os.path.isdir(self.model), f"Model path not found: {self.model}"
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

cfg: Optional[Config] = None

def init_cfg(args) -> Config:
    global cfg
    cli_val = getattr(args, "custom_kernel", False)
    
    cfg = Config(
        model=args.model,
        # 使用 getattr(对象, 属性名, 默认值) 替代直接点号访问
        enforce_eager=getattr(args, "enforce_eager", False),
        tensor_parallel_size=getattr(args, "tensor_parallel_size", 1), # 安全读取
        custom_kernel=cli_val
    )
    return cfg