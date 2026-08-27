"""The spiral navigator: three orthogonal planes, flagged and then unflagged.

A navigator earns its place by making a rigid pose observable, which needs the
three planes to be genuinely orthogonal, and by not contaminating the scan it
rides in, which needs its flag to be turned off again.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver.design.readout.navigator import PLANES


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


@pytest.fixture
def navigator(system):
    return design.SpiralNavigator(system)


def test_the_three_planes_are_mutually_orthogonal(navigator):
    """The pose the tracking recovers has six degrees of freedom, and one
    plane constrains only four of them. Each plane must therefore be flat
    along a different axis, which is what makes the set span the pose."""
    samples = np.asarray(navigator.seq.calculate_kspace()[0])
    per_plane = samples.shape[1] // len(PLANES)

    normals = []
    for index in range(len(PLANES)):
        plane = samples[:, index * per_plane : (index + 1) * per_plane]
        extent = plane.max(axis=1) - plane.min(axis=1)
        normals.append(int(np.argmin(extent)))
        # The normal is flat, and the two in-plane axes are not.
        assert extent[normals[-1]] == pytest.approx(0.0, abs=1e-6)
        assert sorted(extent)[1] > 0.1 * extent.max()

    assert sorted(normals) == [0, 1, 2], "two planes share a normal"


def test_the_navigator_flag_is_cleared_before_the_scan_resumes(navigator, system):
    """A label is sequence state, not a property of the block that sets it. A
    navigator that left ``NAV`` set would have every imaging readout after it
    delivered as navigator data."""
    seq = pp.Sequence(system)
    for block in navigator.blocks:
        seq.add_block(*block)
    seq.add_block(
        pp.make_trapezoid(channel="x", area=1000, duration=2e-3, system=system),
        pp.make_adc(num_samples=64, duration=2e-3, system=system),
    )

    flags = seq.evaluate_labels(evolution="adc")["NAV"]
    assert list(flags) == [1] * len(PLANES) + [0]


def test_every_plane_acquires_exactly_once(navigator):
    acquiring = [
        block
        for block in navigator.blocks
        if any(getattr(event, "type", "") == "adc" for event in block)
    ]
    assert len(acquiring) == len(PLANES)


def test_one_designed_arm_serves_every_plane(navigator):
    """The planes are rotations of a single interleave, not three designs --
    which is what keeps a navigator train cheap in waveform memory."""
    assert navigator.readout.adc.num_samples > 0
    assert set(navigator.rotations) == set(PLANES)
    assert navigator.rotations["axial"] is None, "the designed plane is not turned"
    assert navigator.duration > len(PLANES) * navigator.plane_duration * 0.9


def test_a_navigator_tr_shorter_than_its_planes_is_refused(system):
    with pytest.raises(ValueError, match="shorter than the"):
        design.SpiralNavigator(system, navigator_tr=1e-3)


def test_the_navigator_tr_paces_the_planes(system):
    paced = design.SpiralNavigator(system, navigator_tr=100e-3)
    assert paced.duration == pytest.approx(100e-3, abs=1e-4)


@pytest.mark.parametrize("module", ["mprage3D_sequence", "fse3D_sequence"])
def test_a_navigator_train_costs_the_host_sequence_no_time(module):
    """Navigators ride in dead time the host sequence already had, so the
    interval that sets its contrast must come back unchanged."""
    import pulserver.app as app

    sequence = getattr(app, module)
    with_nav = sequence.main(navigator=True, n_z=8)
    without = sequence.main(n_z=8)

    assert with_nav.duration()[0] == pytest.approx(without.duration()[0], abs=1e-6)


@pytest.mark.parametrize("module", ["mprage3D_sequence", "fse3D_sequence"])
def test_only_the_navigator_readouts_are_flagged(module):
    """The imaging readouts must reach a reconstruction as imaging data, which
    is what the flag being cleared after each train buys."""
    import numpy as np

    import pulserver.app as app

    sequence = getattr(app, module)
    flags = np.asarray(
        sequence.main(navigator=True, n_z=8).evaluate_labels(evolution="adc")["NAV"]
    )
    imaging = np.asarray(
        sequence.main(n_z=8).evaluate_labels(evolution="adc").get("LIN")
    )

    assert flags.sum() > 0, "no navigator readouts were flagged"
    assert int((flags == 0).sum()) == imaging.size, (
        "the imaging readouts are not the unflagged ones"
    )
    assert int(flags.sum()) % len(PLANES) == 0, "a navigator lost a plane"


@pytest.mark.parametrize("module", ["mprage3D_sequence", "fse3D_sequence"])
def test_the_navigator_is_off_unless_a_script_turns_it_on(module):
    """The plugin gate is a constant a reader can see, and it is off."""
    import pulserver.app as app

    assert getattr(app, module).NAVIGATOR is False
