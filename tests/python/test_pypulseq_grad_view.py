"""An arbitrary gradient holds a view of the caller's samples until a sequence takes it.

`make_arbitrary_grad` does not copy: the event's `waveform` is the array it
was given (as upstream PyPulseq's is), the normalised shape is divided out
where it is read, and `add_block` copies once, dividing on the way into the
shape library. Every row that reaches the library is bit-identical to the
one a copied-and-normalised event would register.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp


@pytest.fixture
def system() -> pp.Opts:
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        grad_raster_time=4e-6,
        rf_raster_time=2e-6,
        adc_raster_time=2e-6,
        block_duration_raster=4e-6,
    )


def _arm(seed: int, n: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 4e-6
    envelope = np.sin(np.pi * np.arange(n) / (n - 1)) ** 2
    w = np.sin(2 * np.pi * 300.0 * t)
    w += 0.3 * np.sin(2 * np.pi * (450.0 + 37.0 * seed) * t + rng.uniform(0, 2 * np.pi))
    return -0.2e6 * w * envelope


def _signed_peak(w: np.ndarray) -> float:
    """The largest magnitude, carrying the sign of the first nonzero sample."""
    peak = float(np.max(np.abs(w)))
    first = w[np.nonzero(w)[0][0]]
    return -peak if first < 0 else peak


def _rows(seq: pp.Sequence) -> list[np.ndarray]:
    """The gradient shapes the sequence writes, in registration order."""
    text = seq._to_text() if hasattr(seq, "_to_text") else None
    if isinstance(text, bytes):
        text = text.decode()
    if text is None:
        import pathlib
        import tempfile

        path = pathlib.Path(tempfile.mkdtemp()) / "s.seq"
        seq.write(str(path))
        text = path.read_text()
    import re

    block = re.search(r"\[SHAPES\](.*?)(?:\n\[|\Z)", text, re.S).group(1)
    rows = []
    for chunk in re.split(r"\n(?=shape_id)", block.strip()):
        m = re.match(r"shape_id\s+(\d+)\s*\nnum_samples\s+(\d+)\n(.*)", chunk, re.S)
        if m:
            rows.append(np.array([float(v) for v in m.group(3).split()]))
    return rows


def test_the_factory_hands_back_the_callers_array(system):
    w = _arm(1)
    g = pp.make_arbitrary_grad(channel="x", waveform=w, system=system)
    assert g.waveform is w
    assert len(g) == w.size
    assert g.amplitude == _signed_peak(w)


def test_the_shape_is_the_samples_divided_by_the_signed_peak(system):
    w = _arm(2)
    g = pp.make_arbitrary_grad(channel="x", waveform=w, system=system)
    peak = _signed_peak(w)
    assert np.array_equal(np.asarray(g.shape), w / peak)
    assert np.asarray(g.shape).max() == 1.0


def test_add_block_registers_the_row_a_copied_event_would(system):
    w = _arm(3)
    viewing = pp.make_arbitrary_grad(channel="x", waveform=w, system=system)
    copied = pp.make_arbitrary_grad(channel="x", waveform=w, system=system)
    copied.waveform = w.copy()  # the setter copies and normalises in place
    assert copied.waveform is not w

    seq_view = pp.Sequence(system=system)
    seq_view.add_block(viewing)
    seq_copy = pp.Sequence(system=system)
    seq_copy.add_block(copied)
    a, b = _rows(seq_view), _rows(seq_copy)
    assert len(a) == len(b) == 1
    assert np.array_equal(a[0], b[0])


def test_the_same_event_added_twice_registers_one_shape(system):
    w = _arm(4)
    g = pp.make_arbitrary_grad(channel="x", waveform=w, system=system)
    seq = pp.Sequence(system=system)
    seq.add_block(g)
    seq.add_block(g)
    assert len(_rows(seq)) == 1


def test_a_scaled_copy_keeps_the_view(system):
    w = _arm(5)
    g = pp.make_arbitrary_grad(channel="x", waveform=w, system=system)
    half = pp.scale_grad(g, 0.5)
    assert np.allclose(np.asarray(half.waveform), 0.5 * w)
    assert np.array_equal(np.asarray(half.shape), np.asarray(g.shape))
    del w
    seq = pp.Sequence(system=system)
    seq.add_block(g)
    seq.add_block(half)
    rows = _rows(seq)
    assert len(rows) == 2 and np.array_equal(rows[0], rows[1])


def test_the_view_follows_the_array_as_upstreams_does(system):
    w = _arm(6)
    g = pp.make_arbitrary_grad(channel="x", waveform=w, system=system)
    w *= 0.5
    assert np.array_equal(np.asarray(g.waveform), w)


def test_setting_the_waveform_owns_a_copy(system):
    w = _arm(7)
    g = pp.make_arbitrary_grad(channel="x", waveform=w, system=system)
    g.waveform = 2.0 * w
    assert g.waveform is not w
    assert np.allclose(np.asarray(g.waveform), 2.0 * w)
    w[:] = 0.0
    assert np.abs(np.asarray(g.waveform)).max() > 0.0


def test_the_written_file_does_not_depend_on_the_view(system, tmp_path):
    w = _arm(8)
    a = pp.Sequence(system=system)
    a.add_block(pp.make_arbitrary_grad(channel="x", waveform=w, system=system))
    b = pp.Sequence(system=system)
    owned = pp.make_arbitrary_grad(channel="x", waveform=w, system=system)
    owned.waveform = w.copy()
    b.add_block(owned)
    a.write(str(tmp_path / "a.seq"))
    b.write(str(tmp_path / "b.seq"))
    assert (tmp_path / "a.seq").read_bytes() == (tmp_path / "b.seq").read_bytes()
