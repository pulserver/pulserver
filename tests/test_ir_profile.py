"""What a cache states its numbers are held as."""

import struct

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver import ir

SYSTEM = pp.Opts(B0=3.0)

#: One integer step of a quantity that spans its full scale over 32767 steps.
FULL_SCALE = 1.0 / 32767.0

#: Where the six (format, step) pairs sit, from the start of the COMMON region.
_QUANTITIES = 6


def written(tmp_path, **kwargs):
    seq = pp.Sequence(SYSTEM)
    seq.add_block(
        pp.make_sinc_pulse(
            flip_angle=np.deg2rad(30), duration=2e-3, system=SYSTEM, use="excitation"
        )
    )
    path = tmp_path / "one.seq"
    seq.write(path)
    return ir.convert(path, SYSTEM, cache_ext=".pseg", **kwargs)


def pairs_in(cache):
    """Return every (format, step) pair the cache holds, wherever they sit."""
    raw = cache.read_bytes()
    words = struct.unpack(f"={len(raw) // 4}i", raw[: (len(raw) // 4) * 4])
    floats = struct.unpack(f"={len(raw) // 4}f", raw[: (len(raw) // 4) * 4])
    return words, floats


def test_a_cache_states_a_float_in_si_by_default(tmp_path):
    """A machine that has stated no format reads what it always read."""
    cache = written(tmp_path)
    words, _ = pairs_in(cache)
    # Six formats of zero and six steps of zero sit together somewhere.
    assert any(
        all(words[at + i] == 0 for i in range(2 * _QUANTITIES))
        for at in range(len(words) - 2 * _QUANTITIES)
    )


def test_a_stated_format_and_step_reach_the_cache(tmp_path):
    """The profile is what a reader establishes the cache was built for."""
    profile = ir.VendorProfile(
        grad_sample=ir.Quantity(ir.Format.INT16, FULL_SCALE),
        rf_phase=ir.Quantity(ir.Format.INT32, 0.125),
    )
    cache = written(tmp_path, profile=profile)
    words, floats = pairs_in(cache)
    found = [
        at
        for at in range(len(words) - 2 * _QUANTITIES)
        if words[at] == int(ir.Format.INT16)
        and floats[at + 1] == pytest.approx(FULL_SCALE)
        and words[at + 8] == int(ir.Format.INT32)
        and floats[at + 9] == pytest.approx(0.125)
    ]
    assert found, "the formats and steps the profile stated are not in the cache"


def test_a_profile_is_six_quantities_of_two_values(tmp_path):
    """The extension takes them positionally, so the count is the contract."""
    assert len(ir.VendorProfile().as_pairs()) == 2 * _QUANTITIES


def test_a_quantity_stored_as_a_float_needs_no_step():
    assert ir.Quantity().format is ir.Format.FLOAT32
    assert ir.Quantity().step == 0.0
