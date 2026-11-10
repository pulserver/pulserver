"""The envelope of a design reads at least the nerve response and slew rates of the scan it bounds."""

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver import _plugins
from pulserver.design import load_plugin, repetition_envelope
from pulserver.host import _service as service
from pulserver.host._limits import design_system, split_limits
from pulserver.protocol import UIParam

SHIPPED = sorted(path.stem for path in _plugins.SEQUENCES.glob("*.py"))

#: The shipped sequences a block of whose repetition varies in more than a
#: plane -- a rotated blade and its phase encode, a 3D radial spoke: each is
#: its own envelope.
UNFOLDED = {
    "gre_propeller2d",
    "gre_stack_of_blades3d",
    "se_propeller2d",
    "se_stack_of_blades3d",
    "zte3d",
}

#: A 7 T head gradient designed at the coil's slew rate, and its nerve model.
SEVEN_T = {
    "B0": 7.0,
    "max_grad": 100.0,
    "grad_unit": "mT/m",
    "max_slew": 200.0,
    "slew_unit": "T/m/s",
    "design_max_grad": 100.0,
    "design_max_slew": 200.0,
    "rf_raster_time": 2e-6,
    "grad_raster_time": 4e-6,
    "adc_raster_time": 2e-6,
    "block_duration_raster": 4e-6,
    "max_b1": 25.0,
    "b1_unit": "uT",
    "design_rf_dead_time": 232e-6,
    "design_rf_ringdown_time": 4e-6,
    "pns_chronaxie": 359e-6,
    "pns_rheobase": 71.62162,
    "pns_limit": 0.98,
}

#: A double-oblique prescription, which mixes every logical axis into every
#: physical one.
OBLIQUE = np.array(
    [
        [0.8660254, 0.0, 0.5],
        [0.3535534, 0.7071068, -0.6123724],
        [-0.3535534, 0.7071068, 0.6123724],
    ]
)

#: The readings a lower slew rate eases.
EASED = ("pns", "slew", "vector_slew")

#: Smaller matrices than the defaults, so the scans design quickly.
SMALL = {UIParam.NY: 64, UIParam.NSLICES: 24}


def _readings(sequence, rotation):
    system, _, checked = split_limits(SEVEN_T)
    return service._representative_readings(
        sequence, system, checked, rotation, any_rotation=True, held=True
    )


def _designed(name):
    """Return the scan a shipped plugin designs at its default protocol, small, under the 7 T limits."""
    plugin = load_plugin(_plugins.SEQUENCES / f"{name}.py")
    system = design_system(SEVEN_T)
    listing = plugin.listing()
    request = {key: p.value for key, p in listing.items() if p.editable}
    request |= {key: value for key, value in SMALL.items() if key in listing}
    validation, protocol = plugin._validated(system, request)
    assert validation.valid, validation.info
    built = plugin.generate(system, protocol)
    return plugin, built if isinstance(built, list) else [built]


def _lines(views):
    """Return a phase-encoded line per view, each played by the same block definitions."""
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    seq = pp.Sequence(system)
    readout = pp.make_trapezoid("x", flat_area=2e3, flat_time=2e-3, system=system)
    encode = pp.make_trapezoid("y", area=1e3, duration=1e-3, system=system)
    pulse = pp.make_block_pulse(np.pi / 18, duration=0.5e-3, system=system)
    for view in views:
        seq.add_block(pulse)
        seq.add_block(pp.scale_grad(encode, view))
        seq.add_block(readout)
        seq.add_block(pp.scale_grad(encode, -view))
        seq.add_block(pp.make_delay(5e-3))
    return seq


def test_the_envelope_of_phase_encoded_lines_plays_whole_repetitions_fewer_than_the_scan():
    full = _lines(np.linspace(-1.0, 1.0, 256))
    envelope = repetition_envelope(full)
    blocks = len(envelope.block_definitions())
    assert blocks % full.repetition()[0] == 0
    assert blocks < len(full.block_definitions())
    for key in EASED:
        assert _readings(envelope, OBLIQUE)[key] >= _readings(full, OBLIQUE)[key]


def test_a_sequence_whose_repetitions_differ_in_a_block_is_its_own_envelope():
    full = _lines(np.linspace(-1.0, 1.0, 8))
    full.add_block(pp.make_trapezoid("z", area=1e3, duration=1e-3, system=full.system))
    assert repetition_envelope(full) is full


def test_rotated_spokes_are_bounded_on_every_axis_of_an_oblique_prescription():
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    seq = pp.Sequence(system)
    readout = pp.make_trapezoid("x", flat_area=2e3, flat_time=2e-3, system=system)
    pulse = pp.make_block_pulse(np.pi / 18, duration=0.5e-3, system=system)
    for angle in np.pi * np.arange(40) / 40:
        seq.add_block(pulse)
        seq.add_block(readout, pp.make_rotation(float(angle)))
        seq.add_block(pp.make_delay(5e-3))
    envelope = repetition_envelope(seq)
    assert len(envelope.block_definitions()) < len(seq.block_definitions())
    for key in EASED:
        assert _readings(envelope, OBLIQUE)[key] >= _readings(seq, OBLIQUE)[key]


@pytest.mark.parametrize("name", SHIPPED)
def test_the_envelope_of_a_shipped_scan_reads_at_least_what_the_scan_does(name):
    plugin, chain = _designed(name)
    for sequence in chain:
        envelope = plugin.envelope(sequence)
        folded = envelope is not sequence
        if sequence is chain[-1]:
            assert folded == (name not in UNFOLDED)
        if folded:
            assert len(envelope.block_definitions()) < len(sequence.block_definitions())
        for rotation in (None, OBLIQUE):
            scan = _readings(sequence, rotation)
            bound = _readings(envelope, rotation)
            for key in EASED:
                assert bound[key] >= scan[key] * (1.0 - 1e-9), (key, rotation)
