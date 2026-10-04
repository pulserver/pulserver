"""A design checked and converted as written agrees with its files read back."""

from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _zoo import SMALL, designed

from pulserver import ir
from pulserver.design._plugin import _write
from pulserver.ir._convert import prescribe
from pulserver.ir._source import conversion_payload
from pulserver.mrd._sequence import designed_chain, read_chain

OFFSET = (0.02, -0.012, 0.005)
#: The payload columns holding a phase in rad.
PHASES = {"rf": (9,), "adc": (6,), "adc_offsets": (1,)}
ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _written(name, directory):
    built = designed(name)
    paths = _write(built, directory / "sequence.seq")
    chain = [built] if isinstance(built, pp.Sequence) else built
    return paths, designed_chain(list(zip(map(Path, paths), chain, strict=True)))


@pytest.mark.parametrize("name", sorted(SMALL))
def test_a_design_as_written_converts_as_its_files_read_back(name, tmp_path):
    """Moved to an offset, the structure is the same and every number agrees to float32.

    A binary file holds shape samples in float32, so what is computed from
    them differs in the digits past that.
    """
    paths, chain = _written(name, tmp_path)
    for (_, held), (_, read) in zip(chain, read_chain(paths[0]), strict=True):
        prescribe(held, OFFSET)
        prescribe(read, OFFSET)
        ours = conversion_payload(held, pp.Opts())
        theirs = conversion_payload(read, pp.Opts())
        assert ours.keys() == theirs.keys()
        for key, value in ours.items():
            _assert_agrees(value, theirs[key], key)


def _assert_agrees(ours, theirs, key):
    if isinstance(ours, dict):
        assert ours == theirs, key
    elif key == "shapes":
        assert [count for count, _ in ours] == [count for count, _ in theirs]
        for (_, a), (_, b) in zip(ours, theirs, strict=True):
            np.testing.assert_allclose(
                a, b, rtol=1e-7, atol=1e-7 * np.abs(b).max(initial=0)
            )
    elif isinstance(ours, list) and ours and isinstance(ours[0], np.ndarray):
        for a, b in zip(ours, theirs, strict=True):
            np.testing.assert_array_equal(a, b)
    else:
        a, b = np.asarray(ours), np.asarray(theirs)
        assert a.shape == b.shape, key
        for column in PHASES.get(key, ()):
            # A phase near a whole turn wraps either way.
            a = a.copy()
            a[:, column] = b[:, column] + np.angle(
                np.exp(1j * (a[:, column] - b[:, column]))
            )
        if a.dtype.kind == "f":
            scale = np.abs(b).max(initial=0)
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6 * scale, err_msg=key)
        else:
            np.testing.assert_array_equal(a, b, err_msg=key)


@pytest.mark.parametrize("name", sorted(SMALL))
def test_a_design_as_written_fails_the_checks_its_files_fail(name, tmp_path):
    paths, chain = _written(name, tmp_path)
    strict = pp.Opts(max_grad=5, grad_unit="mT/m", max_slew=20, slew_unit="T/m/s")
    read = ir.check(paths[0], strict, rotation=ROTATION)
    held = ir.check(paths[0], strict, rotation=ROTATION, designed=chain)
    assert held == read
