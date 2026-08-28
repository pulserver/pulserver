"""Wave encoding: the corkscrew a readout plays, and what it must not disturb.

A wave-encoded readout spreads every voxel along itself, so the aliasing
parallel imaging has to separate is spread with it. What makes that usable
rather than merely clever is everything the corkscrew leaves alone: k has to
end where it would have ended without it, the gradients have to stay inside
the system, and switching it off for one readout has to change nothing else
about the sequence.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver.design.readout._common import wave_gradients

SYSTEM = pp.Opts.default
FOV = (0.24, 0.24, 0.16)
MATRIX = (192, 192, 160)


def excitation():
    return design.SpatialSelectiveExcitation(SYSTEM, 10.0, FOV[2], 2e-3, is_slab=True)


def readout(**kwargs):
    """A 3D line readout, wave-encoded unless told otherwise."""
    pulse = excitation()
    return design.LineReadout3D(
        SYSTEM,
        pulse.rf,
        pulse.gz,
        fov=FOV,
        matrix=MATRIX,
        readout_bandwidth_hz=250e3,
        labels=("LIN", "PAR", "IMA", "SEG"),
        **{"wave": "both", "wave_cycles": 8, "wave_amplitude": 8e-3, **kwargs},
    )


def moment(event) -> float:
    """The zeroth moment of a gradient event, in 1/m.

    The samples sit at raster centres and the waveform reaches zero over half a
    raster at each end, so the ends are worth a quarter of an interior sample --
    the same rule :meth:`~pulserver.pypulseq.Sequence.calculate_kspace` applies,
    which is what makes this the area anything downstream would have to undo.
    """
    waveform = np.asarray(event.waveform)
    area = waveform.sum() - 0.25 * (waveform[0] + waveform[-1])
    return float(area) * SYSTEM.grad_raster_time


def steepest(event) -> float:
    """Its worst slew, in T/m/s.

    The samples sit at raster centres, so the steps into and out of zero cross
    half a raster: the same convention ``make_arbitrary_grad`` checks against.
    """
    waveform = np.asarray(event.waveform)
    steps = np.concatenate(
        [[2.0 * waveform[0]], np.diff(waveform), [-2.0 * waveform[-1]]]
    )
    return np.abs(steps).max() / SYSTEM.grad_raster_time / SYSTEM.gamma


# %% what the corkscrew must not disturb


@pytest.mark.parametrize("cycles", [4, 8, 13, 20])
@pytest.mark.parametrize("mode", ["phase", "partition", "both"])
def test_the_wave_returns_k_to_exactly_where_it_found_it(cycles, mode):
    """The event carries no net area, so nothing downstream compensates it.

    This is what lets a scan scale the wave to zero for one readout: the
    prewinder and the rewinder are the same events either way, because the
    corkscrew never moved k anywhere it did not put it back.
    """
    module = readout(wave=mode, wave_cycles=cycles)
    for channel in ("y", "z"):
        event = getattr(module, f"g{channel}_wave", None)
        if event is None:
            continue
        # A phase encode's own step is the scale that matters: a residual has
        # to be small against the k a line is placed at, not against one.
        step = 1.0 / FOV[1 if channel == "y" else 2]
        assert abs(moment(event)) < 1e-6 * step


@pytest.mark.parametrize("mode", ["phase", "partition", "both"])
def test_the_sequence_leaves_k_where_the_wave_free_readout_would_have(mode):
    """The same invariant, asked of the sequence rather than of the event.

    How much area a waveform carries depends on how k is integrated under it,
    and the readout and the integrator have to agree on that or the corkscrew
    leaves a line somewhere its counters do not say it is. So this asks the
    question anything downstream asks: where k ended up, against where it would
    have ended up had the corkscrew never been played.
    """
    module = readout(wave=mode)
    events = [
        event
        for event in (
            getattr(module, "gy_wave", None),
            getattr(module, "gz_wave", None),
        )
        if event is not None
    ]

    def after(scale):
        seq = pp.Sequence(SYSTEM)
        seq.add_block(module.rf, module.gz)
        seq.add_block(module.gx_pre, module.gy_pre, module.gz_pre)
        seq.add_block(module.gx, module.adc, *(pp.scale_grad(e, scale) for e in events))
        seq.add_block(pp.make_adc(num_samples=2, dwell=SYSTEM.grad_raster_time))
        return np.asarray(seq.calculate_kspace()[0])[:, -1]

    played, silent = after(1.0), after(0.0)
    for axis, extent in ((1, FOV[1]), (2, FOV[2])):
        assert abs(float(played[axis] - silent[axis])) < 1e-6 / extent


@pytest.mark.parametrize("cycles", [4, 8, 20])
def test_the_wave_stays_inside_the_gradient_system(cycles):
    module = readout(wave_cycles=cycles)
    for channel in ("y", "z"):
        event = getattr(module, f"g{channel}_wave")
        peak = np.abs(np.asarray(event.waveform)).max()
        assert peak <= SYSTEM.max_grad
        assert steepest(event) <= SYSTEM.max_slew / SYSTEM.gamma * (1 + 1e-9)


def test_the_amplitude_gives_way_to_the_slew_rate():
    """A sinusoid slews at its amplitude times its frequency, so a fast
    corkscrew cannot also be a strong one. The requested amplitude is a
    ceiling; what the readout gets is whichever limit binds first."""
    slow = readout(wave_cycles=4, wave_amplitude=4e-3)
    fast = readout(wave_cycles=20, wave_amplitude=8e-3)

    assert slow.wave_amplitude == pytest.approx(4e-3, rel=1e-9)
    assert fast.wave_amplitude < 8e-3
    assert steepest(fast.gy_wave) == pytest.approx(
        SYSTEM.max_slew / SYSTEM.gamma, rel=1e-6
    )


def test_the_two_axes_share_one_amplitude():
    """The corkscrew is round: the sinusoid and the cosinusoid reach the same
    peak, so a voxel is spread as far along one encoded axis as the other."""
    module = readout()
    peaks = [
        np.abs(np.asarray(getattr(module, f"g{channel}_wave").waveform)).max()
        for channel in ("y", "z")
    ]
    assert peaks[0] == pytest.approx(peaks[1], rel=0.05)


# %% what it is for


def test_the_corkscrew_reaches_k_space():
    """The spread is what a reconstruction undoes, and it is only worth
    undoing if it is there: several voxels along both encoded axes."""
    module = readout()
    for channel, extent in (("y", FOV[1]), ("z", FOV[2])):
        waveform = np.asarray(getattr(module, f"g{channel}_wave").waveform)
        excursion = np.ptp(np.cumsum(waveform) * SYSTEM.grad_raster_time)
        assert excursion * extent > 1.0


def test_a_readout_without_the_wave_has_no_wave_events():
    module = readout(wave=None)
    assert getattr(module, "gy_wave", None) is None
    assert getattr(module, "gz_wave", None) is None
    assert module.wave_amplitude == 0.0


# %% what it refuses


def test_more_cycles_than_the_flat_top_can_hold_are_refused():
    with pytest.raises(ValueError, match="gradient raster periods"):
        readout(wave_cycles=500)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="wave mode"):
        readout(wave="corkscrew")


@pytest.mark.parametrize("cycles", [0, -1])
def test_a_corkscrew_that_does_not_turn_is_refused(cycles):
    with pytest.raises(ValueError, match="at least one"):
        readout(wave_cycles=cycles)


def test_wave_needs_both_encoded_axes_free():
    """In two dimensions the second axis is the slice, and a gradient on it
    during the readout would undo the selection rather than encode."""
    pulse = excitation()
    with pytest.raises(ValueError, match="only one of them"):
        design.LineReadout2D(
            SYSTEM, pulse.rf, pulse.gz, fov=FOV[:2], matrix=MATRIX[:2], wave="phase"
        )


def test_the_flat_top_has_to_hold_a_quarter_period_of_edge():
    with pytest.raises(ValueError, match="gradient raster periods"):
        wave_gradients(
            SYSTEM, flat_time=1e-4, delay=0.0, cycles=8, amplitude=8e-3, mode="both"
        )


# %% switching it off


def sequence(scales):
    """One repetition per entry, its wave scaled by the entry -- or absent."""
    module = readout()
    seq = pp.Sequence(SYSTEM)
    for scale in scales:
        seq.add_block(module.rf, module.gz)
        seq.add_block(module.gx_pre, module.gy_pre, module.gz_pre)
        wave = (
            ()
            if scale is None
            else (
                pp.scale_grad(module.gy_wave, scale),
                pp.scale_grad(module.gz_wave, scale),
            )
        )
        seq.add_block(module.gx, module.adc, *wave)
        seq.add_block(module.gx_spoil, module.gy_rew, module.gz_rew)
    return seq


def structure(seq):
    detected = seq._structure_for("plot")
    return seq.tr_size, [
        (segment["num_blocks"], segment["pure_delay"]) for segment in detected.segments
    ]


def test_scaling_the_wave_to_zero_leaves_the_structure_alone():
    """How a calibration line is acquired wave-free.

    Scaling the event keeps the readout's block *definition*, so the sequence
    is still one repeating unit and its TR and its segments are what they
    would have been. Leaving the event out instead is a different block, and
    the sequence fragments into several segments around it -- which is what
    the scan loop and the interpreter would then have to carry.
    """
    everywhere = structure(sequence([1.0] * 6))
    scaled_off = structure(sequence([1.0, 1.0, 0.0, 0.0, 1.0, 1.0]))
    left_out = structure(sequence([1.0, 1.0, None, None, 1.0, 1.0]))

    assert scaled_off == everywhere
    assert left_out != everywhere


def test_a_wave_scaled_to_zero_encodes_nothing():
    module = readout()
    for channel in ("y", "z"):
        off = pp.scale_grad(getattr(module, f"g{channel}_wave"), 0.0)
        assert not np.any(np.asarray(off.waveform))
