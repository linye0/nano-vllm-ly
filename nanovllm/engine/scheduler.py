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

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = [] 
        num_seqs = 0 # 每次schedule会为num_seqs数量的seqs预定资源
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            # 如果scheduler的waiting队列未空并且本次调度的seqs数量没到上限
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            # 把当前的序列从waiting里面弹出来，加入running，并且改变序列的信息
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            # 如果seqs是从waiting里面取出来的，说明一定是prefill阶段的序列
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            # running是一个双端队列，越靠左边的越老，越右边的越新
            seq = self.running.popleft()
            # decode阶段每次只会产生一个token，如果这个token导致多了一个需要分配的block，而blockmanager还拿不出这个block
            # 那么就要从running里面释放某个seqs的资源
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 优先踢掉新的请求
                    # preempt所做的事：1. 把seq的状态置为waiting 2.将其从running队列踢出 3.blockmanager回收其所占用的资源
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        # decode阶段，所以返回的is_prefill=False
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
