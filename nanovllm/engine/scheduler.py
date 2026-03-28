from collections import deque
from nanovllm.config import Config, is_chunked_prefill
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if is_chunked_prefill():
            return self._schedule_chunked()
        return self._schedule_legacy()
    
    def _schedule_chunked(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                num_batched_tokens += 1
                seq.cur_chunk_size = 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            remaining_tokens = self.max_num_batched_tokens - num_batched_tokens
            if remaining_tokens <= 0:
                break
                
            chunk_size = min(remaining_tokens, seq.num_pending_prefill_tokens, 256)
            if not self.block_manager.can_allocate(seq, chunk_size=chunk_size):
                break

            # 核心修复：接收 allocate 返回的缓存命中跳跃量，并实时同步逻辑进度
            jump_offset = self.block_manager.allocate(seq, chunk_size=chunk_size)
            seq.num_computed_tokens += jump_offset

            # 基于同步后的最新进度计算本轮真实的计算量
            actual_chunk_size = min(chunk_size, seq.num_prompt_tokens - seq.num_computed_tokens)

            num_seqs += 1
            num_batched_tokens += actual_chunk_size
            seq.cur_chunk_size = actual_chunk_size
            seq.status = SequenceStatus.RUNNING  

            self.waiting.popleft()
            scheduled_seqs.append(seq)

        for seq in reversed(scheduled_seqs):
            expected_computed_tokens = seq.num_computed_tokens + seq.cur_chunk_size
            if expected_computed_tokens >= seq.num_prompt_tokens:
                self.running.appendleft(seq)
            else:
                self.waiting.appendleft(seq)
        
        has_prefill = any(not s.is_prefill_finished for s in scheduled_seqs) 
        return scheduled_seqs, has_prefill 

    def _schedule_legacy(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = [] 
        num_seqs = 0 
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs: return [], False # 健壮性增强
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)

    def postprocess_chunked(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.num_computed_tokens += seq.cur_chunk_size
            if seq.num_computed_tokens >= seq.num_prompt_tokens:
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running: self.running.remove(seq)
                    elif seq in self.waiting: self.waiting.remove(seq)