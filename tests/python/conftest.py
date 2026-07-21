from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pulserver import Protocol, TypeinFloatParam, UIParam


@pytest.fixture
def sample_protocol() -> Protocol:
    return {
        UIParam.TE: TypeinFloatParam(value=5.0, min=1.0, max=100.0, incr=0.1, unit="ms"),
        UIParam.TR: TypeinFloatParam(value=20.0, min=1.0, max=1000.0, incr=0.1, unit="ms"),
    }


@pytest.fixture
def bridge_test_plugin_module():
    plugin_path = Path(__file__).resolve().parents[2] / "bridge" / "tests" / "test_plugin.py"
    spec = importlib.util.spec_from_file_location("bridge_test_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
