"""Protect paired accuracy claims from identity and repeated-sample mistakes."""
import unittest

from scripts.benchmark_ouro_accuracy import paired_accuracy, validate_requests


class AccuracyBenchmarkTests(unittest.TestCase):
    def rows(self, values):
        return [dict(benchmark='test', id=f'q{i//2}', sample=i%2, seed=42+i, correct=value)
                for i,value in enumerate(values)]

    def test_opposite_sample_changes_stay_together_within_each_question(self):
        reference = self.rows([True,False]*8)
        candidate = self.rows([False,True]*8)
        result = paired_accuracy(reference,candidate,1000)
        self.assertEqual(result['delta_avg'],0.)
        self.assertEqual(result['ci95_question_bootstrap'],[0.,0.])
        self.assertEqual(result['lost_correct'],8)
        self.assertEqual(result['gained_correct'],8)

    def test_uniform_accuracy_loss_cannot_be_reported_as_equivalence(self):
        result = paired_accuracy(self.rows([True]*16),self.rows([False]*16),1000)
        self.assertEqual(result['delta_avg'],-1.)
        self.assertEqual(result['ci95_question_bootstrap'],[-1.,-1.])

    def test_missing_duplicate_and_seed_changed_responses_are_rejected(self):
        rows = self.rows([True,False]*4)
        changed = [dict(row) for row in rows]
        changed[0]['seed'] += 1
        for candidate in (rows[:-1],rows+[rows[0]],changed):
            with self.assertRaises(ValueError):
                paired_accuracy(rows,candidate,100)

    def test_input_changes_and_duplicates_are_rejected_before_sharding(self):
        rows = [dict(row,input_ids=[40,41],max_new_tokens=16384) for row in self.rows([True,False]*4)]
        validate_requests(rows,list(reversed(rows)))
        changed = [dict(row) for row in rows]
        changed[0]['input_ids'] = [40,42]
        for candidate in (changed,rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):
                validate_requests(rows,candidate)


if __name__=='__main__':
    unittest.main()
