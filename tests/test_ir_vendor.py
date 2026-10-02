"""The vendor file: one vendor's grouping and number formats, stated once and read."""

import shutil
from pathlib import Path

import pypulseqpp as pp
import pytest

from pulserver import ir

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s", B0=3.0)
FIXTURES = Path(__file__).parent / "fixtures" / "sequences"

VENDOR = """\
# A sequencer that plays integers and pays switching time per unit.
[Grouping]
boundary_gradient_hz_per_m: 2000
split_by_pulses: true
split_by_readouts: false
split_navigators: true
split_edge_delays: false
[Grouping End]

[VendorProfile]
grad_sample: int16 3.0517578125e-05
grad_amplitude: float32 0
rf_sample: int16 3.0517578125e-05
rf_amplitude: float32 0
rf_phase: int32 0.125
rf_frequency: float32 0
[VendorProfile End]
"""

PROFILE = ir.VendorProfile(
    grad_sample=ir.Quantity(ir.Format.INT16, 2.0**-15),
    rf_sample=ir.Quantity(ir.Format.INT16, 2.0**-15),
    rf_phase=ir.Quantity(ir.Format.INT32, 0.125),
)
GROUPING = ir.Grouping(
    boundary_gradient_hz_per_m=2000.0,
    split_by_readouts=False,
    split_edge_delays=False,
)


def _vendor(tmp_path, text=VENDOR):
    path = tmp_path / "vendor.txt"
    path.write_text(text)
    return path


def test_a_vendor_file_reads_as_the_profile_and_grouping_it_states(tmp_path):
    assert ir.read_vendor(_vendor(tmp_path)) == (PROFILE, GROUPING)


@pytest.mark.parametrize("name", ["gre_2d_3sl.seq", "epi_2d_main.seq"])
def test_a_conversion_under_a_vendor_file_is_the_conversion_under_its_arguments(
    name, tmp_path
):
    profile, grouping = ir.read_vendor(_vendor(tmp_path))
    caches = []
    for directory, kwargs in (
        ("file", {"profile": profile, "grouping": grouping}),
        ("arguments", {"profile": PROFILE, "grouping": GROUPING}),
    ):
        (tmp_path / directory).mkdir()
        seq = shutil.copy(FIXTURES / name, tmp_path / directory / name)
        caches.append(ir.convert(seq, SYSTEM, **kwargs).read_bytes())
    assert caches[0] == caches[1]


def _without(line):
    return VENDOR.replace(line + "\n", "")


@pytest.mark.parametrize(
    "text",
    [
        _without("split_navigators: true"),
        VENDOR.replace("rf_phase:", "rf_phase_offset:"),
        VENDOR.replace("split_by_pulses: true", "split_by_pulses: yes"),
        VENDOR.replace("int32 0.125", "int24 0.125"),
        VENDOR.replace("rf_amplitude: float32 0", "rf_amplitude: float32"),
        VENDOR.replace("[Grouping End]", "split_by_pulses: false\n[Grouping End]"),
        _without("[VendorProfile End]"),
        VENDOR.split("[VendorProfile]")[0],
        VENDOR + "[Limits]\nB0: 3\n[Limits End]\n",
        VENDOR + "boundary_gradient_hz_per_m: 100\n",
        VENDOR + VENDOR,
    ],
    ids=[
        "missing field",
        "unknown field",
        "unreadable flag",
        "unknown format",
        "missing step",
        "field stated twice",
        "unclosed block",
        "missing block",
        "unknown block",
        "line outside a block",
        "block stated twice",
    ],
)
def test_a_vendor_file_stating_anything_else_is_refused(text, tmp_path):
    """A grouping silently replaced by the default is a cache that plays the wrong units."""
    with pytest.raises(ValueError):
        ir.read_vendor(_vendor(tmp_path, text))
