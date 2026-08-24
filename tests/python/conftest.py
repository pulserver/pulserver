from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pulserver.design import TypeinFloatParam, UIParam
from pulserver.design._params import Protocol

# Shared helper modules beside the tests (fixture_corpus, comparison helpers
# in test modules) and the generators/oracle under tests/utils import by
# bare name whatever directory pytest runs from.
for _helper_dir in (
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent / "recon",
    Path(__file__).resolve().parents[1] / "utils",
):
    if str(_helper_dir) not in sys.path:
        sys.path.insert(0, str(_helper_dir))


@pytest.fixture
def sample_protocol() -> Protocol:
    return {
        UIParam.TE: TypeinFloatParam(
            value=5.0, min=1.0, max=100.0, incr=0.1, unit="ms"
        ),
        UIParam.TR: TypeinFloatParam(
            value=20.0, min=1.0, max=1000.0, incr=0.1, unit="ms"
        ),
    }


@pytest.fixture
def bridge_test_plugin_module():
    plugin_path = (
        Path(__file__).resolve().parents[2] / "src" / "nim" / "tests" / "test_plugin.py"
    )
    spec = importlib.util.spec_from_file_location("bridge_test_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
