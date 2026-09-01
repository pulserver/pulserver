"""The structure a scan declares needs no gradient samples.

``declare_tr``, ``tr_size``, ``num_trs`` and ``num_segments`` derive from a
structure-only conversion: gradient shapes travel without their samples,
the conversion sweeps no gradient statistics, and the collection refuses
what it cannot answer. The structure it reports must be the one the full
conversion reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.pypulseq._structure import _Structure

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = [
    "gre_2d.seq",
    "gre_2d_3sl.seq",
    "fse_2d.seq",
    "epi_2d.seq",
    "gre_spiral_2d.seq",
    "gre_radial_2d.seq",
    "zte_3d.seq",
    "mprage_stack_of_spirals_3d.seq",
]


def _system():
    return pp.Opts(
        max_grad=40.0,
        grad_unit="mT/m",
        max_slew=350.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=20e-6,
        block_duration_raster=20e-6,
    )


def _read(name: str) -> pp.Sequence:
    seq = pp.Sequence(_system())
    seq.read(str(FIXTURES / name))
    return seq


@pytest.mark.parametrize("name", CORPUS)
def test_the_light_structure_declares_the_same_scan_as_the_full_one(name):
    seq = _read(name)
    light = _Structure(seq, light=True)
    full = _Structure(seq)
    for key in ("tr_size", "num_trs", "tr_duration_us", "tr_start"):
        if key in full.tr:
            assert light.tr[key] == full.tr[key], key
    # The instance a segment is shown from is chosen by gradient energy,
    # which a light conversion does not compute; its boundaries, span and
    # kind are what the structure is.
    structural = (
        "index",
        "num_blocks",
        "duration_us",
        "pure_delay",
        "is_nav",
        "has_trigger",
    )
    assert len(light.segments) == len(full.segments)
    for a, b in zip(light.segments, full.segments, strict=True):
        assert {k: a[k] for k in structural} == {k: b[k] for k in structural}


def test_a_light_write_carries_no_readout_samples():
    system = pp.Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=350.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=4e-6,
        block_duration_raster=4e-6,
        rf_raster_time=2e-6,
    )
    t = np.linspace(0.0, 1.0, 4096)
    taper = 4.0 * t * (1.0 - t)
    rf = pp.make_block_pulse(flip_angle=0.17, duration=200e-6, system=system)
    adc = pp.make_adc(num_samples=4096, dwell=4e-6, system=system)
    seq = pp.Sequence(system)
    for k in range(16):
        phase = 2.0 * np.pi * k / 16
        gx = 0.6 * system.max_grad * np.sin(40 * np.pi * t + phase) * taper
        gy = 0.6 * system.max_grad * np.cos(40 * np.pi * t + phase) * taper
        seq.add_block(rf)
        seq.add_block(
            pp.make_arbitrary_grad(channel="x", waveform=gx, system=system),
            pp.make_arbitrary_grad(channel="y", waveform=gy, system=system),
            adc,
        )
    full = seq._to_binary()
    light = seq._to_binary(structure_only=True)
    assert len(light) < len(full) / 10


def test_a_light_collection_refuses_a_safety_check():
    from pulserver._ext.pulseg import _check_safety

    seq = _read("gre_2d.seq")
    light = _Structure(seq, light=True)
    with pytest.raises(RuntimeError, match="structure only"):
        _check_safety(light.collection, [], 4.25e8 / 0.333, 360.0, 100.0, False)


def test_the_sequence_keeps_light_and_full_structures_apart():
    seq = _read("gre_2d.seq")
    assert seq.num_trs > 1
    assert seq._light_structure is not None and seq._light_structure.light
    assert seq._structure is None
