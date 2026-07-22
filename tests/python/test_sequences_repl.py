"""Public REPL sequence factories use the self-contained bridge plugins."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_repl_factories_have_explicit_controls_and_return_sequences():
    """The installed editable tree cannot mask the source tree under test."""
    code = """
import inspect
import sys
from pathlib import Path

sys.meta_path[:] = [finder for finder in sys.meta_path if 'editable' not in type(finder).__module__]
sys.path.insert(0, str(Path('python').resolve()))

import pulserver.sequence as singular
import pulserver.sequences as sequences

signature = inspect.signature(sequences.design_gre_2d)
assert 'protocol' not in signature.parameters
assert 'te_ms' in signature.parameters
assert 'system' in signature.parameters
assert singular.design_gre is sequences.design_gre

seq = sequences.design_gre_2d(nx=32, ny=16, tr_ms=150.0)
assert seq.get_definition('Name') == 'gre_2d'
assert len(seq.block_events) > 0
assert type(seq._seq).__module__.startswith('pypulseq.')

epi = sequences.design_epi_2d(nx=32, ny=16, etl=16)
assert epi.get_definition('Name') == 'epi_2d'
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
