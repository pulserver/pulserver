"""Public-API contract for the complete sequence example collection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pulserver.pypulseq as pp
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "sequences"
PLUGINS = sorted(path for path in EXAMPLES.glob("*.py") if path.name != "__init__.py")


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"public_example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", PLUGINS, ids=lambda path: path.stem)
def test_example_uses_only_public_pypulseq_imports(path):
    source = path.read_text()
    assert "from pulserver.pypulseq import _" not in source
    assert "from pulserver.pypulseq._" not in source
    assert "import pulserver.pypulseq._" not in source


@pytest.mark.parametrize("path", PLUGINS, ids=lambda path: path.stem)
def test_default_protocol_is_valid(path):
    module = _load(path)
    system = pp.Opts()
    result = module.validate_protocol(system, module.get_default_protocol(system))
    assert result["valid"], result["info"]


@pytest.mark.parametrize("name", ["bssfp_2d", "bssfp_3d", "epi_2d", "fse_3d", "gre_2d", "gre_noncart_2d"])
def test_representative_example_writes_sequence(name, tmp_path):
    module = _load(EXAMPLES / f"{name}.py")
    system = pp.Opts()
    output = tmp_path / f"{name}.seq"
    module.make_sequence(system, module.get_default_protocol(system), str(output))
    assert output.is_file()
    assert output.stat().st_size > 0
