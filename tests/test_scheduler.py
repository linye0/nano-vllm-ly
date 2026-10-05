import unittest
import pickle
from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def scheduler_config(**overrides):
    values = dict(
        max_num_seqs=8,
        max_num_batched_tokens=256,
        eos=99,
        num_kvcache_blocks=64,
        kvcache_block_size=256,
        chunked_prefill=True,
        prefill_chunk_size=128,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def sequence(length, max_tokens=4, block_size=256):
    return Sequence(list(range(length)), SamplingParams(max_tokens=max_tokens, ignore_eos=True), block_size)


class ChunkedSchedulerTests(unittest.TestCase):
    def test_prompt_advances_in_bounded_chunks(self):
        scheduler = Scheduler(scheduler_config())
        seq = sequence(300)
        scheduler.add(seq)

        observed = []
        for _ in range(3):
            scheduled, is_prefill = scheduler.schedule()
            self.assertTrue(is_prefill)
            self.assertEqual(scheduled, [seq])
            observed.append(seq.cur_chunk_size)
            scheduler.postprocess_chunked(scheduled, [7])

        self.assertEqual(observed, [128, 128, 44])
        self.assertEqual(seq.num_computed_tokens, 300)
        self.assertEqual(seq.completion_token_ids, [7])
        self.assertIn(seq, scheduler.running)

    def test_decode_is_prioritized_and_respects_token_budget(self):
        scheduler = Scheduler(scheduler_config(max_num_batched_tokens=2, prefill_chunk_size=2))
        seqs = [sequence(8) for _ in range(3)]
        for seq in seqs:
            scheduler.block_manager.allocate(seq)
            seq.num_computed_tokens = seq.num_prompt_tokens
            seq.status = SequenceStatus.RUNNING
            scheduler.running.append(seq)

        scheduled, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, seqs[:2])
        self.assertEqual([seq.cur_chunk_size for seq in scheduled], [1, 1])
        self.assertEqual(list(scheduler.running), [seqs[2], seqs[0], seqs[1]])

    def test_mixed_batch_places_decode_before_prefill(self):
        scheduler = Scheduler(scheduler_config(max_num_batched_tokens=129))
        decoding = sequence(8)
        scheduler.block_manager.allocate(decoding)
        decoding.num_computed_tokens = decoding.num_prompt_tokens
        decoding.status = SequenceStatus.RUNNING
        scheduler.running.append(decoding)
        prefill = sequence(400)
        scheduler.add(prefill)

        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [decoding, prefill])
        self.assertEqual([seq.cur_chunk_size for seq in scheduled], [1, 128])

    def test_preempted_decode_recomputes_all_existing_tokens(self):
        scheduler = Scheduler(scheduler_config())
        seq = sequence(10, max_tokens=5)
        scheduler.block_manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        seq.num_computed_tokens = 10
        seq.append_token(20)
        seq.append_token(21)

        scheduler.preempt(seq)
        self.assertEqual(seq.num_prompt_tokens, 12)
        self.assertEqual(seq.orig_prompt_len, 10)
        self.assertEqual(seq.num_pending_prefill_tokens, 12)
        self.assertFalse(seq.is_prefill_finished)
        self.assertEqual(seq.max_tokens, 3)

    def test_chunk_progress_survives_worker_serialization(self):
        seq = sequence(300)
        seq.num_computed_tokens = 128
        seq.cur_chunk_size = 64
        restored = pickle.loads(pickle.dumps(seq))
        self.assertEqual(restored.num_computed_tokens, 128)
        self.assertEqual(restored.cur_chunk_size, 64)
        self.assertEqual(restored.token_ids, seq.token_ids)


class PrefixCacheTests(unittest.TestCase):
    def test_full_prompt_cache_hit_still_recomputes_last_block(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        owner = sequence(8, block_size=4)
        manager.allocate(owner)
        requester = sequence(8, block_size=4)

        matched = manager.allocate(requester, chunk_size=4)
        self.assertEqual(matched, 4)
        self.assertEqual(requester.num_cached_tokens, 4)
        self.assertEqual(requester.block_table[0], owner.block_table[0])
        self.assertNotEqual(requester.block_table[1], owner.block_table[1])


class LegacySchedulerTests(unittest.TestCase):
    def test_unschedulable_request_reports_deadlock(self):
        scheduler = Scheduler(scheduler_config(chunked_prefill=False, num_kvcache_blocks=0))
        scheduler.add(sequence(8))
        scheduled, is_prefill = scheduler.schedule()
        self.assertEqual(scheduled, [])
        self.assertFalse(is_prefill)
        self.assertTrue(scheduler.is_deadlock())

    def test_decode_round_robin_does_not_starve_tail(self):
        scheduler = Scheduler(scheduler_config(chunked_prefill=False, max_num_seqs=2))
        seqs = [sequence(8) for _ in range(3)]
        for seq in seqs:
            scheduler.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            scheduler.running.append(seq)

        scheduled, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, seqs[:2])
        self.assertEqual(list(scheduler.running), [seqs[2], seqs[0], seqs[1]])


if __name__ == "__main__":
    unittest.main()
