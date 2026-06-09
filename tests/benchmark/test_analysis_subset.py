from pathlib import Path

import pytest

from benchmark.analysis_subset import (
    ANALYSIS_SUBSET_PATH,
    filter_paper_ids,
    load_analysis_subset,
)


def test_analysis_subset_path_exists():
    assert ANALYSIS_SUBSET_PATH.exists()


def test_load_analysis_subset_b2a():
    subset = load_analysis_subset()
    assert subset["version"] == "b2a-v1"
    assert len(subset["included_paper_ids"]) == 7
    assert "acl_2024.findings-acl.438" in subset["excluded_paper_ids"]
    assert "icml_2311.10263v2" in subset["excluded_paper_ids"]


def test_filter_paper_ids_default_included_only():
    all_ids = [
        "acl_2024.acl-short.8",
        "acl_2024.findings-acl.438",
        "icml_2311.10263v2",
    ]
    filtered = filter_paper_ids(all_ids, use_subset=True)
    assert filtered == ["acl_2024.acl-short.8"]


def test_filter_paper_ids_all_papers():
    all_ids = ["acl_2024.acl-short.8", "acl_2024.findings-acl.438"]
    assert filter_paper_ids(all_ids, use_subset=False) == all_ids
