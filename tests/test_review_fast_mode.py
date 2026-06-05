from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from deepreview.config import Settings, get_settings
from deepreview.prompts.review_agent_prompt import (
    build_review_agent_system_prompt,
    resolve_review_min_annotation_count,
)


class ReviewFastModeTests(unittest.TestCase):
    def test_fast_profile_overrides(self) -> None:
        from deepreview.config import apply_review_fast_profile

        settings = apply_review_fast_profile(
            Settings(
                review_fast_mode=True,
                agent_max_turns=1000,
                min_annotations_for_final=10,
                paper_search_enabled=True,
            )
        )
        self.assertTrue(settings.review_fast_mode)
        self.assertLessEqual(settings.agent_max_turns, 40)
        self.assertEqual(settings.min_annotations_for_final, 8)
        self.assertFalse(settings.paper_search_enabled)

    def test_fast_prompt_is_compact(self) -> None:
        prompt = build_review_agent_system_prompt(
            source_file_id='job-1',
            source_file_name='paper.pdf',
            paper_markdown='# Title\n\nBody text.',
            review_fast_mode=True,
            max_markdown_chars=48000,
            review_fast_max_turns=18,
            review_fast_min_annotations=2,
        )
        self.assertIn('FAST REVIEW mode', prompt)
        self.assertIn('at most 18 tool-using', prompt)
        self.assertNotIn('Phase 1 Plan/Audit', prompt)
        self.assertLess(len(prompt), 8000)

    def test_resolve_min_annotations(self) -> None:
        self.assertEqual(resolve_review_min_annotation_count(review_fast_mode=True), 8)
        self.assertEqual(resolve_review_min_annotation_count(review_fast_mode=False), 10)

    def test_fast_final_report_section_order(self) -> None:
        from deepreview.tools.review_tools import _required_final_report_section_order

        self.assertEqual(
            _required_final_report_section_order(review_fast_mode=True, kg_criteria_active=True),
            [
                'summary',
                'strengths',
                'weaknesses',
                'key_issues',
                'actionable_suggestions',
                'claim_level_audit',
                'scores',
            ],
        )
        self.assertIn('priority_revision_plan', _required_final_report_section_order(review_fast_mode=False))

    def test_get_settings_cache_respects_env(self) -> None:
        get_settings.cache_clear()
        with patch.dict(os.environ, {'REVIEW_FAST_MODE': 'true'}, clear=False):
            settings = get_settings()
            self.assertTrue(settings.review_fast_mode)
        get_settings.cache_clear()


if __name__ == '__main__':
    unittest.main()
