"""The 30-second contract for a scan of distinct precomputed arms.

One end-to-end wall guard: assembly through verdict, at a scale whose linear
extrapolation to 128K arms is the budget in
``docs/_bench/pipeline_budget.py``. Opt-in via ``PULSERVER_BUDGET=1`` -- it
runs minutes and exists to be invoked when a stage lands, not on every run.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "_bench"))

pytestmark = pytest.mark.budget

if os.environ.get("PULSERVER_BUDGET") != "1":
    pytest.skip(
        "pipeline budget guard is opt-in: set PULSERVER_BUDGET=1",
        allow_module_level=True,
    )


def test_the_pipeline_meets_its_budget_at_target_scale():
    import pipeline_budget

    entry = pipeline_budget.run(32768)
    pipeline_budget.report(entry)
    at, budget = entry["at_target"], entry["budget"]

    assert entry["assembly_us_per_arm"] <= budget["assembly_us_per_arm"]
    assert at["declare_dedup_s"] <= budget["declare_dedup_s_at_target"]
    assert entry["write_mb_per_s"] >= budget["write_mb_per_s"]
    assert entry["parse_mb_per_s"] >= budget["parse_mb_per_s"]
    assert at["gate_s"] <= budget["gate_s_at_target"]
    assert at["end_to_end_s"] <= budget["end_to_end_s_at_target"]
