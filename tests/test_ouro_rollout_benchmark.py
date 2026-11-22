"""Pure CPU request-identity checks for the isolated rollout benchmark."""
import unittest

from scripts.benchmark_ouro_rollout import BASE,CANDIDATES,WORKLOADS,confirmation_variants,partition


class OuroRolloutBenchmarkTests(unittest.TestCase):
    def requests(self):
        return [dict(id=f'q{q}',question_index=q,sample=s,seed=1000+q*8+s)
                for q in range(32) for s in range(8)]

    def test_question_partition_matches_existing_rank_ownership(self):
        shards=partition(self.requests(),4,'question')
        self.assertEqual([len(shard) for shard in shards],[64]*4)
        for rank,shard in enumerate(shards):
            self.assertTrue(all(row['question_index']%4==rank for row in shard))
            for q in range(rank,32,4):
                self.assertEqual([row['sample'] for row in shard if row['question_index']==q],list(range(8)))

    def test_spreading_samples_preserves_all_ids_and_seeds(self):
        requests=self.requests();shards=partition(requests,4,'sample')
        self.assertEqual([len(shard) for shard in shards],[64]*4)
        self.assertEqual(sorted((r['id'],r['sample'],r['seed']) for shard in shards for r in shard),
                         sorted((r['id'],r['sample'],r['seed']) for r in requests))
        for shard in shards:
            for q in range(32):self.assertEqual(sum(r['question_index']==q for r in shard),2)

    def test_screen_does_not_reduce_context_or_forced_output_lengths(self):
        for _,count,prefix,length in WORKLOADS:
            self.assertGreater(count,0);self.assertLessEqual(prefix+length,17408)
        self.assertIn(16384,[prefix for _,_,prefix,_ in WORKLOADS])
        for _,changes in CANDIDATES:
            options=BASE|changes
            self.assertGreater(options['memory_fraction'],0)
            self.assertLess(options['memory_fraction'],1)
            self.assertGreaterEqual(options['max_num_batched_tokens'],options['max_num_seqs'] or 1)

    def test_unknown_or_zero_partition_is_rejected(self):
        for count,strategy in ((0,'sample'),(4,'unknown')):
            with self.assertRaises(ValueError):partition(self.requests(),count,strategy)

    def test_small_screen_changes_do_not_replace_the_real_baseline(self):
        best=dict(long_context_geomean_speedup=1.01,speedup_by_workload=dict(pressure8k=1.,pressure16k=1.,decode4k=1.03),
            options=BASE|dict(memory_fraction=.9))
        variants=confirmation_variants(best)
        self.assertEqual([row[0] for row in variants],['real_baseline','real_spread'])
        self.assertEqual(variants[-1][2],BASE)

    def test_meaningful_screen_gain_is_confirmed_separately_from_partition(self):
        best=dict(long_context_geomean_speedup=1.10,speedup_by_workload=dict(pressure8k=1.08,pressure16k=1.1,decode4k=1.12),
            options=BASE|dict(memory_fraction=.9))
        variants=confirmation_variants(best)
        self.assertEqual([row[0] for row in variants],['real_baseline','real_tuned','real_spread'])
        self.assertEqual(variants[1][1],'question');self.assertEqual(variants[2][1],'sample')


if __name__=='__main__':unittest.main()
