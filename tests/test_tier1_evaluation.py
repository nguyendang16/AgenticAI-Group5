from __future__ import annotations

import json
import unittest
from pathlib import Path

from deepreview.evaluation.tier1 import (
    aggregate_verdict,
    check_annotation_grounding,
    evaluate_and_save_job,
    evaluate_job_dir,
    sample_grounding_checks,
    write_evaluation_report,
)


class Tier1EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.jobs_root = cls.repo_root / 'data' / 'jobs'

    def test_check_annotation_grounding_pass(self) -> None:
        page_index = {
            2: [
                'Header',
                'Vocabulary acquisition is the foundation of language learning.',
            ]
        }
        result = check_annotation_grounding(
            {
                'id': 'a1',
                'page': 2,
                'start_line': 2,
                'end_line': 2,
                'text': 'Vocabulary acquisition is the foundation of language learning.',
            },
            page_index,
        )
        self.assertTrue(result['passed'])
        self.assertGreaterEqual(result['overlap'], 0.8)

    def test_sample_grounding_checks_scores(self) -> None:
        page_index = {1: ['alpha beta gamma', 'delta epsilon']}
        annotations = [
            {
                'id': '1',
                'page': 1,
                'start_line': 1,
                'end_line': 1,
                'text': 'alpha beta gamma',
            },
            {
                'id': '2',
                'page': 1,
                'start_line': 2,
                'end_line': 2,
                'text': 'delta epsilon',
            },
            {
                'id': '3',
                'page': 1,
                'start_line': 1,
                'end_line': 1,
                'text': 'alpha beta gamma',
            },
        ]
        result = sample_grounding_checks(annotations, page_index, sample_size=3, seed='test')
        self.assertEqual(result['sample_size'], 3)
        self.assertEqual(result['passed'], 3)
        self.assertEqual(result['score'], 2)

    def test_evaluate_completed_job_if_present(self) -> None:
        completed_dirs = [
            child
            for child in self.jobs_root.iterdir()
            if child.is_dir() and (child / 'job.json').exists()
        ]
        target = None
        for child in completed_dirs:
            payload = json.loads((child / 'job.json').read_text(encoding='utf-8'))
            if payload.get('status') == 'completed':
                target = child
                break
        if target is None:
            self.skipTest('No completed job available for integration test')

        result = evaluate_job_dir(target)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['reliability']['completed'])
        self.assertIn('framework', result)
        self.assertIn('system_metrics', result)
        self.assertIn('M1_sr', result['system_metrics'])
        self.assertIn('system_verdict', result)
        self.assertIn(result['system_verdict']['verdict'], {'pass', 'fail'})
        self.assertIsNotNone(result['oqi'])
        self.assertGreaterEqual(int(result['oqi']['total']), 0)

        saved = evaluate_and_save_job(target.name)
        self.assertTrue((target / 'evaluation.json').exists())
        self.assertEqual(saved['job_id'], target.name)

    def test_aggregate_verdict_system_only(self) -> None:
        good_row = {
            'reliability': {'completed': True, 'tier1_pass': True},
            'efficiency': {'wall_clock_minutes': 20, 'tokens_total': 100000, 'tool_calls_total': 12},
            'system_verdict': {'verdict': 'pass'},
            'root_cause_bucket': None,
        }
        bad_row = {
            'reliability': {'completed': False, 'tier1_pass': False},
            'efficiency': {'wall_clock_minutes': None, 'tokens_total': 0, 'tool_calls_total': 0},
            'system_verdict': {'verdict': 'fail'},
            'root_cause_bucket': 'R2',
        }
        rows = [
            {**good_row, 'job_id': 'a'},
            {**good_row, 'job_id': 'b'},
            {**good_row, 'job_id': 'c'},
            {**bad_row, 'job_id': 'd'},
        ]
        summary = aggregate_verdict(rows)
        self.assertEqual(summary['verdict'], 'go-with-fixes')
        self.assertEqual(summary['job_count'], 4)
        self.assertEqual(summary['completion_rate'], 0.75)
        self.assertEqual(summary['deliverable_rate'], 0.75)
        self.assertNotIn('median_oqi', summary)

    def test_aggregate_verdict_and_report(self) -> None:
        rows = [
            {
                'job_id': 'a',
                'title': 'Paper A',
                'status': 'completed',
                'reliability': {'completed': True, 'tier1_pass': True},
                'efficiency': {'wall_clock_minutes': 30, 'tokens_total': 200000, 'tool_calls_total': 15},
                'system_metrics': {'M4_multi_round': {'pass': True}, 'M7_trace': {'pass': True}},
                'system_verdict': {'verdict': 'pass'},
                'root_cause_bucket': None,
                'tier2_needed': False,
            },
            {
                'job_id': 'b',
                'title': 'Paper B',
                'status': 'failed',
                'reliability': {'completed': False, 'tier1_pass': False},
                'efficiency': {'wall_clock_minutes': 5, 'tokens_total': 10000, 'tool_calls_total': 2},
                'system_metrics': {'M4_multi_round': {'pass': False}, 'M7_trace': {'pass': True}},
                'system_verdict': {'verdict': 'fail'},
                'root_cause_bucket': 'R2',
                'tier2_needed': True,
            },
        ]
        summary = aggregate_verdict(rows)
        self.assertIn(summary['verdict'], {'go', 'go-with-fixes', 'no-go'})
        self.assertEqual(summary['job_count'], 2)

        out_dir = self.repo_root / 'data' / 'evaluations' / '_test_output'
        paths = write_evaluation_report(rows, output_dir=out_dir)
        self.assertTrue(paths['json'].exists())
        self.assertTrue(paths['csv'].exists())
        self.assertTrue(paths['markdown'].exists())
        md_text = paths['markdown'].read_text(encoding='utf-8')
        self.assertIn('MINT', md_text)
        self.assertIn('AgentBoard', md_text)


if __name__ == '__main__':
    unittest.main()
