from collections import deque
import xxhash
import numpy as np
from nanovllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence, chunk_size: int = None) -> bool:
        if chunk_size is None:
            return len(self.free_block_ids) >= seq.num_blocks
        
        new_total_tokens = seq.num_computed_tokens + chunk_size
        needed_blocks = (new_total_tokens + self.block_size - 1) // self.block_size
        additional_needed = max(0, needed_blocks - len(seq.block_table))
        return len(self.free_block_ids) >= additional_needed

    def allocate(self, seq: Sequence, chunk_size: int = None) -> int:
        if chunk_size is None:
            self._allocate_init(seq)
            return 0
        
        if seq.num_computed_tokens == 0:
            return self._allocate_init(seq, chunk_size)
        else:
            self._allocate_inc(seq, chunk_size)
            return 0

    def _allocate_init(self, seq: Sequence, chunk_size: int = None) -> int:
        assert not seq.block_table
        matched_tokens = 0

        if chunk_size is not None:
            matched_blocks = 0
            h = -1
            for i in range(seq.num_blocks):
                token_ids = seq.block(i)
                if len(token_ids) != self.block_size:
                    break
                h = self.compute_hash(token_ids, h)
                block_id = self.hash_to_block_id.get(h, -1)
                # 核心修复：只匹配已经在使用的活跃块，防止幽灵匹配透支 free_block_ids
                if block_id == -1 or self.blocks[block_id].token_ids != token_ids or block_id not in self.used_block_ids:
                    break
                matched_blocks += 1
            
            matched_tokens = matched_blocks * self.block_size
            needed_total_tokens = matched_tokens + chunk_size
            num_blocks_to_process = (needed_total_tokens + self.block_size - 1) // self.block_size
        else:
            num_blocks_to_process = seq.num_blocks

        h = -1
        cache_miss = False
        for i in range(num_blocks_to_process):
            token_ids = seq.block(i)
            is_full_block = len(token_ids) == self.block_size
            if is_full_block:
                h = self.compute_hash(token_ids, h)
                block_id = self.hash_to_block_id.get(h, -1)
            else:
                h = -1
                block_id = -1

            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 核心修复：不再这里直接修改 seq.num_computed_tokens，统一由 Scheduler 返回后修改
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id] 
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)

            if is_full_block and cache_miss:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
        
        return matched_tokens

    def _allocate_inc(self, seq: Sequence, chunk_size: int = 0):
        start_block_idx = seq.num_computed_tokens // self.block_size
        h = self.blocks[seq.block_table[start_block_idx - 1]].hash if start_block_idx > 0 else -1

        new_total_tokens = seq.num_computed_tokens + chunk_size
        needed_blocks = (new_total_tokens + self.block_size - 1) // self.block_size
        
        additional_needed = max(0, needed_blocks - len(seq.block_table))
        for _ in range(additional_needed):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)

        end_block_idx = new_total_tokens // self.block_size
        for i in range(start_block_idx, end_block_idx):
            current_block_id = seq.block_table[i]
            block = self.blocks[current_block_id]
            if block.hash != -1:
                h = block.hash
                continue
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            cached_block_id = self.hash_to_block_id.get(h, -1)
            
            if cached_block_id != -1 and self.blocks[cached_block_id].token_ids == token_ids and cached_block_id != current_block_id:
                if cached_block_id in self.used_block_ids:
                    self._deallocate_block(current_block_id)
                    seq.block_table[i] = cached_block_id
                    self.blocks[cached_block_id].ref_count += 1
                else:
                    block.update(h, token_ids)
                    self.hash_to_block_id[h] = current_block_id
            else:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = current_block_id

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        if hasattr(seq, "num_computed_tokens"):
            seq.num_computed_tokens = 0
            if not seq.is_finished:
                seq.max_tokens -= seq.num_completion_tokens
                seq.num_prompt_tokens = seq.num_tokens

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1