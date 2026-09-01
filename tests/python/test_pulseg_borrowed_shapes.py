"""A collection read from bytes keeps its shape samples in those bytes.

The reader leaves the sample cells where they are instead of copying them,
and the collection object holds the bytes for as long as it lives -- so a
caller that drops its own reference still gets every waveform, and the same
waveforms a copy would have given.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

import pulserver.pypulseq as pp
from pulserver._ext.pulseg import _get_tr_waveforms, _PulseqCollection

FIXTURES = Path(__file__).parent / "fixtures"


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


def _collection(payload: bytes):
    s = _system()
    return _PulseqCollection(
        [payload],
        float(s.gamma),
        float(s.B0),
        float(s.max_grad),
        float(s.max_slew),
        float(s.rf_raster_time),
        float(s.grad_raster_time),
        float(s.adc_raster_time),
        float(s.block_duration_raster),
        True,
    )


def _waveforms(coll):
    w = _get_tr_waveforms(coll)
    return {
        k: np.asarray(v)
        for k, v in w.items()
        if isinstance(v, np.ndarray) or hasattr(v, "__array__")
    }


def test_a_collection_outlives_the_bytes_it_was_read_from():
    seq = pp.Sequence(_system())
    seq.read(str(FIXTURES / "gre_spiral_2d.seq"))
    held = seq._to_binary()
    reference = _waveforms(_collection(held))

    borrowed = _collection(bytes(held))  # a temporary the caller never keeps
    del seq
    gc.collect()
    junk = [bytearray(1 << 20) for _ in range(64)]  # churn the allocator
    del junk
    gc.collect()

    got = _waveforms(borrowed)
    assert got.keys() == reference.keys()
    for key in reference:
        assert np.array_equal(got[key], reference[key]), key
