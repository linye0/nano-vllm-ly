from collections import deque
from nanovllm.config import Config
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
        self.dead_lock = False
        self.chunked_prefill = config.chunked_prefill
        self.prefill_chunk_size = config.prefill_chunk_size

    def is_finished(self):
        return (not self.waiting and not self.running) or self.dead_lock

    def is_deadlock(self):
        return self.dead_lock

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.chunked_prefill:
            return self._schedule_chunked()
        return self._schedule_legacy()

    def _schedule_chunked(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        scheduled_decode = []
        scheduled_prefill = []
        num_seqs = 0
        num_batched_tokens = 0
        # Decode-first scheduling protects inter-token latency.
        while self.running and num_seqs < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
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
                scheduled_decode.append(seq)

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # Allocate prefill work from the remaining per-step token budget.
            remaining_tokens = self.max_num_batched_tokens - num_batched_tokens

            if remaining_tokens <= 0:
                break

            chunk_size = min(remaining_tokens, seq.num_pending_prefill_tokens, self.prefill_chunk_size)

            can_alloc = self.block_manager.can_allocate(seq, chunk_size=chunk_size)
            while not can_alloc:
                if not self.running:
                    found_victim = False
                    for i in range(len(self.waiting) - 1, 0, -1):
                        if len(self.waiting[i].block_table) > 0:
                            victim = self.waiting[i]
                            self.block_manager.deallocate(victim)
                            found_victim = True
                            break
                    if found_victim:
                        can_alloc = self.block_manager.can_allocate(seq, chunk_size=chunk_size)
                    else:
                        break

            if not can_alloc:
                break

            jump_offset = self.block_manager.allocate(seq, chunk_size=chunk_size)
            seq.num_computed_tokens += jump_offset

            actual_chunk_size = min(chunk_size, seq.num_prompt_tokens - seq.num_computed_tokens)

            num_seqs += 1
            num_batched_tokens += actual_chunk_size
            seq.cur_chunk_size = actual_chunk_size
            seq.status = SequenceStatus.RUNNING

            self.waiting.popleft()
            scheduled_seqs.append(seq)
            scheduled_prefill.append(seq)

        # Round-robin active decode requests: work served in this step moves
        # behind requests that could not fit in the token/sequence budget.
        self.running.extend(scheduled_decode)

        incomplete_prefills = []
        for seq in scheduled_prefill:
            expected_computed_tokens = seq.num_computed_tokens + seq.cur_chunk_size
            if expected_computed_tokens >= seq.num_prompt_tokens:
                self.running.append(seq)
            else:
                incomplete_prefills.append(seq)
        for seq in reversed(incomplete_prefills):
            self.waiting.appendleft(seq)

        has_prefill = bool(scheduled_prefill)
        if not scheduled_seqs and (self.running or self.waiting):
            self.dead_lock = True
        return scheduled_seqs, has_prefill

    def _schedule_legacy(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        preempted = False
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
                    preempted = True
                else:
                    self.preempt(seq)
                    preempted = True
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            if not preempted and (self.running or self.waiting):
                self.dead_lock = True
            return [], False
        self.running.extend(scheduled_seqs)
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        if len(seqs) != len(token_ids):
            raise RuntimeError("sampler returned a different number of tokens than scheduled sequences")
        for seq, token_id in zip(seqs, token_ids):
            seq.num_computed_tokens = seq.num_prompt_tokens
            seq.append_token(token_id)

            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)

    def postprocess_chunked(self, seqs: list[Sequence], token_ids: list[int]):
        if len(seqs) != len(token_ids):
            raise RuntimeError("sampler returned a different number of tokens than scheduled sequences")
        for seq, token_id in zip(seqs, token_ids):
            seq.num_computed_tokens += seq.cur_chunk_size

            if seq.is_prefill_finished:
                # Intermediate chunks do not produce valid next-token logits.
                seq.append_token(token_id)

                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running:
                        self.running.remove(seq)
                    elif seq in self.waiting:
                        self.waiting.remove(seq)
