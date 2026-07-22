"""Public REPL sequence factories use the self-contained bridge plugins."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

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


def test_advanced_sequence_variants_are_repl_designable():
    from pulserver import sequences

    sms = sequences.design_epi_2d(
        nx=32,
        ny=8,
        nslices=4,
        etl=8,
        te_ms=100.0,
        multiband=2,
        fat_sat=True,
        num_frames=2,
        ttl_output=True,
    )
    assert sms.get_definition("MultibandFactor") == 2
    assert sms.get_definition("NumSliceGroups") == 2
    assert sms.get_definition("FatSaturation") is True
    assert sms.get_definition("TTLExternalOutput") is True

    spsp = sequences.design_gre_3d(
        nx=32, ny=8, npartitions=2, te_ms=20.0, tr_ms=100.0, spsp=True
    )
    assert spsp.get_definition("SPSPExcitation") is True

    cine = sequences.design_bssfp_2d(
        nx=32, ny=8, num_frames=2, trigger="cardiac"
    )
    assert cine.get_definition("NumFrames") == 2
    assert cine.get_definition("TriggerType") == "physio2"

    dynamic = sequences.design_gre_noncart_3d(
        nx=32,
        npartitions=2,
        num_shots=4,
        num_frames=2,
        trigger="respiratory",
    )
    assert dynamic.get_definition("NumFrames") == 2
    assert dynamic.get_definition("TriggerType") == "physio1"

    radial = sequences.design_gre_radial_2d(
        nx=32, nslices=1, num_shots=8, num_frames=2, trigger="cardiac"
    )
    assert radial.get_definition("NumFrames") == 2
    assert radial.get_definition("TriggerType") == "physio2"


@pytest.mark.parametrize(
    ("factory_name", "controls"),
    [
        ("design_gre_3d", {"te_ms": 20.0, "tr_ms": 100.0}),
        (
            "design_gre_multiecho_3d",
            {"te_ms": 20.0, "echo_spacing_ms": 10.0, "tr_ms": 200.0},
        ),
        ("design_fse_3d", {"te_ms": 30.0, "tr_ms": 1000.0, "etl": 4}),
        (
            "design_mprage_3d",
            {"te_ms": 20.0, "tr_ms": 100.0, "ti_ms": 900.0, "etl": 4},
        ),
        ("design_epi_3d", {"te_ms": 80.0, "tr_ms": 2000.0, "etl": 8}),
    ],
)
def test_all_cartesian_3d_families_support_spsp(factory_name, controls):
    from pulserver import sequences

    sequence = getattr(sequences, factory_name)(
        nx=32,
        ny=8,
        npartitions=2,
        spsp=True,
        **controls,
    )
    assert sequence.get_definition("SPSPExcitation") is True
