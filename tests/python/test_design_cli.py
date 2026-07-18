"""Unit tests for pulserver.design.cli.run_cli."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pp = pytest.importorskip("pypulseq")

from pulserver.design import cli


@pytest.fixture
def gre_plugin():
    pulserver_interpreter_root = Path(__file__).resolve().parents[4]
    plugin_path = pulserver_interpreter_root / "package" / "pulserver" / "sequences" / "src" / "gre.py"
    if not plugin_path.is_file():
        pytest.skip(f"gre plugin not found: {plugin_path}")
    spec = importlib.util.spec_from_file_location("cli_test_gre_plugin", plugin_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_only_exits_zero(gre_plugin, capsys):
    rc = cli.run_cli(gre_plugin.PLUGIN, ["--validate-only"], arg_map=gre_plugin._ARG_MAP)
    assert rc == 0
    assert "TA" in capsys.readouterr().out


def test_infeasible_arg_exits_two(gre_plugin, capsys):
    rc = cli.run_cli(
        gre_plugin.PLUGIN, ["--validate-only", "--te-ms", "0.1"], arg_map=gre_plugin._ARG_MAP
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


def test_override_changes_validation_output(gre_plugin, capsys):
    rc = cli.run_cli(
        gre_plugin.PLUGIN, ["--validate-only", "--tr-ms", "500", "--ny", "16"],
        arg_map=gre_plugin._ARG_MAP,
    )
    assert rc == 0
    assert "TA = 8.00 s" in capsys.readouterr().out


def test_writes_sequence_file(gre_plugin, tmp_path, capsys):
    out = tmp_path / "cli_gre.seq"
    rc = cli.run_cli(
        gre_plugin.PLUGIN,
        ["-o", str(out), "--nx", "32", "--ny", "16", "--tr-ms", "150"],
        arg_map=gre_plugin._ARG_MAP,
    )
    assert rc == 0
    assert out.is_file()
    assert f"Wrote sequence: {out}" in capsys.readouterr().out


def test_const_flag_sets_value(gre_plugin, capsys):
    rc = cli.run_cli(
        gre_plugin.PLUGIN, ["--validate-only", "--swap-phase-freq"], arg_map=gre_plugin._ARG_MAP
    )
    assert rc == 0
