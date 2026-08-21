"""The chain: each step against the primitive it wraps.

A gadget is the per-acquisition half of a reconstruction, and what makes it
worth having is that a plugin no longer writes it. So what is checked here is
that each one does exactly what the library function it composes does, that the
ones which consume an acquisition consume it, and that a gadget's state belongs
to the stream it was learned from.
"""

from __future__ import annotations

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

from pulserver import ExamCache, ReconContext, ReconPlugin
from pulserver.recon import (
    AcquisitionFlag,
    CoilCompression,
    EpiPhaseCorrection,
    NoiseAdjust,
    RampSampling,
    coil_compress,
    correct_lines,
    epi_ramp_operator,
    estimate_epi_phase,
    noise_prewhiten,
)

COILS = 4
READOUT = 32


def header(readout: int = READOUT):
    """The encoded and reconstructed spaces a gadget sizes itself from."""
    space = SimpleNamespace(matrixSize=SimpleNamespace(x=readout, y=8, z=1))
    return SimpleNamespace(
        encoding=[SimpleNamespace(encodedSpace=space, reconSpace=space)],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


def context():
    return ReconContext(header=header(), exam=ExamCache("gadgets"))


def acquisition(data, *, flags=(), trajectory=None):
    """One acquisition carrying ``data``, and the flags it was given."""
    item = ismrmrd.Acquisition()
    item.resize(
        data.shape[-1],
        data.shape[0],
        0 if trajectory is None else np.shape(trajectory)[-1],
    )
    item.data[:] = data.astype(np.complex64)
    if trajectory is not None:
        item.traj[:] = np.asarray(trajectory, dtype=np.float32)
    for flag in flags:
        item.setFlag(flag)
    return item


def readouts(seed: int = 0, count: int = 1):
    generator = np.random.default_rng(seed)
    return [
        (
            generator.standard_normal((COILS, READOUT))
            + 1j * generator.standard_normal((COILS, READOUT))
        ).astype(np.complex64)
        for _ in range(count)
    ]


# %% NoiseAdjust


def test_a_noise_scan_is_consumed_and_whitens_what_follows():
    """It measures the receiver, not the object, so nothing of it is placed --
    and what it leaves behind is what every later readout is whitened by."""
    (noise,) = readouts(seed=1)
    (line,) = readouts(seed=2)

    gadget = NoiseAdjust()
    gadget.startup(context())

    consumed = gadget(
        acquisition(noise, flags=[ismrmrd.ACQ_IS_NOISE_MEASUREMENT]), noise
    )
    assert consumed is None

    whitened = gadget(acquisition(line), line)
    np.testing.assert_allclose(
        whitened, noise_prewhiten(line, noise, coil_axis=0), rtol=1e-6
    )


def test_a_scan_with_no_noise_measurement_passes_through():
    """Not every scanner sends one, and the readout is still a readout."""
    (line,) = readouts(seed=3)
    gadget = NoiseAdjust()
    gadget.startup(context())
    np.testing.assert_array_equal(gadget(acquisition(line), line), line)


# %% CoilCompression


def test_coil_compression_waits_for_a_basis_and_then_applies_it():
    """Until the calibration has closed there is nothing to project onto, so a
    readout keeps every channel it arrived with."""
    (line,) = readouts(seed=4)
    gadget = CoilCompression(2, key="basis")
    gadget.startup(context())

    np.testing.assert_array_equal(gadget(acquisition(line), line), line)

    calibration = np.stack(readouts(seed=5, count=1))[0]
    basis = gadget.learn(calibration[:, None, :])
    assert basis.shape == (2, COILS)
    np.testing.assert_allclose(gadget(acquisition(line), line), basis @ line, rtol=1e-6)


def test_the_lines_the_basis_is_learned_from_are_not_compressed():
    """There would be nothing to learn it from if they were."""
    (line,) = readouts(seed=6)
    gadget = CoilCompression(2, key="basis")
    gadget.startup(context())
    gadget.learn(np.stack(readouts(seed=7, count=1))[0][:, None, :])

    calibration = acquisition(line, flags=[ismrmrd.ACQ_IS_PARALLEL_CALIBRATION])
    np.testing.assert_array_equal(gadget(calibration, line), line)


def test_the_basis_is_the_one_coil_compress_finds():
    """The gadget is the primitive, applied per readout rather than in bulk."""
    calibration = np.stack(readouts(seed=8, count=1))[0]
    gadget = CoilCompression(3, key="basis")
    gadget.startup(context())

    _, expected = coil_compress(calibration, 3)
    np.testing.assert_allclose(
        gadget.learn(calibration[:, None, :]), expected, rtol=1e-6
    )


def test_the_basis_is_left_where_the_next_stream_of_the_exam_finds_it():
    """A prescan may be its own stream, and the exam cache is what carries an
    artifact from one to the next."""
    shared = context()
    gadget = CoilCompression(2, key="epi_basis")
    gadget.startup(shared)
    basis = gadget.learn(np.stack(readouts(seed=9, count=1))[0][:, None, :])
    np.testing.assert_array_equal(shared.exam.get("epi_basis"), basis)


# %% EpiPhaseCorrection


def test_the_navigator_triplet_is_consumed_and_fits_the_phase():
    """Three blip-nulled lines measure the readout, not the object."""
    lines = readouts(seed=10, count=3)
    gadget = EpiPhaseCorrection(order=1)
    gadget.startup(context())

    for index, line in enumerate(lines):
        flags = [ismrmrd.ACQ_IS_PHASECORR_DATA]
        if index == 1:
            flags.append(ismrmrd.ACQ_IS_REVERSE)
        assert gadget(acquisition(line, flags=flags), line) is None
        # The fit exists only once all three have arrived.
        assert (gadget.phase is not None) is (index == 2)

    expected = estimate_epi_phase(
        [lines[0], lines[1][..., ::-1], lines[2]], polynomial_order=1
    )
    np.testing.assert_allclose(gadget.phase, expected, rtol=1e-6)


def test_a_reversed_line_is_flipped_and_demodulated_by_the_fit():
    """Which is what ``correct_lines`` does, and the gadget is that call."""
    navigator = readouts(seed=11, count=3)
    (line,) = readouts(seed=12)

    gadget = EpiPhaseCorrection(order=1)
    gadget.startup(context())
    for item in navigator:
        gadget(acquisition(item, flags=[ismrmrd.ACQ_IS_PHASECORR_DATA]), item)

    reversed_line = acquisition(line, flags=[ismrmrd.ACQ_IS_REVERSE])
    (expected,) = correct_lines([(line, True)], gadget.phase)
    np.testing.assert_allclose(gadget(reversed_line, line), expected, rtol=1e-6)


# %% RampSampling


def test_a_readout_with_no_trajectory_was_sampled_uniformly():
    """A train that waits for its plateau carries none, and needs no regridding."""
    (line,) = readouts(seed=13)
    gadget = RampSampling()
    gadget.startup(context())
    np.testing.assert_array_equal(gadget(acquisition(line), line), line)


def test_a_ramp_sampled_readout_is_resampled_onto_the_grid():
    """Where the samples fell is what the acquisition says, and the change of
    basis onto the grid is ``epi_ramp_operator``."""
    (line,) = readouts(seed=14)
    # A sinusoidal sweep: sampled across the ramps rather than the plateau.
    taken = np.sin(np.linspace(-np.pi / 2, np.pi / 2, READOUT)).astype(np.float32) / 2

    gadget = RampSampling()
    gadget.startup(context())
    regridded = gadget(acquisition(line, trajectory=taken[:, None]), line)

    grid = (np.arange(READOUT) - READOUT // 2) / READOUT
    expected = (
        line @ epi_ramp_operator(taken / (2 * np.abs(taken).max()), grid, READOUT).T
    )
    np.testing.assert_allclose(regridded, expected, rtol=1e-5, atol=1e-6)


def test_the_operator_is_built_once_for_a_readout_length():
    """One lobe is played for every readout of the train."""
    lines = readouts(seed=15, count=2)
    taken = np.linspace(-0.5, 0.5, READOUT, dtype=np.float32)

    gadget = RampSampling()
    gadget.startup(context())
    gadget(acquisition(lines[0], trajectory=taken[:, None]), lines[0])
    first = gadget.operator
    gadget(acquisition(lines[1], trajectory=taken[:, None]), lines[1])
    assert gadget.operator is first


# %% the chain, and whose state it is


def test_every_stream_gets_its_own_gadgets():
    """What a gadget learned belongs to the scan it learned it from, so two
    concurrent connections cannot see each other's noise or phase fit."""

    class Sink(ReconPlugin):
        def recon(self, branch, context):
            del branch, context

    template = Sink(chain=[NoiseAdjust()])
    first, second = template.spawn(), template.spawn()

    assert first.chain[0] is not second.chain[0]
    assert first.chain[0] is not template.chain[0]

    (noise,) = readouts(seed=16)
    first.startup(context())
    second.startup(context())
    first.chain[0](acquisition(noise, flags=[ismrmrd.ACQ_IS_NOISE_MEASUREMENT]), noise)

    assert first.chain[0].noise is not None
    assert second.chain[0].noise is None


def test_a_consumed_acquisition_never_reaches_a_buffer():
    """``receive`` ends where a step returns nothing."""

    class Sink(ReconPlugin):
        def recon(self, branch, context):
            del branch, context

    plugin = Sink(chain=[NoiseAdjust()], buffered=False).spawn()
    scan = context()
    plugin.startup(scan)

    (noise,) = readouts(seed=17)
    item = acquisition(noise, flags=[ismrmrd.ACQ_IS_NOISE_MEASUREMENT])
    assert plugin.receive(item, scan) is None
    # And the acquisition the plugin holds is still the last accepted one.
    assert plugin.acquisition is None


def test_the_chain_runs_in_the_order_it_was_declared():
    """Each step is handed what the one before it produced."""
    seen: list[str] = []

    class Note(NoiseAdjust):
        def __init__(self, name: str) -> None:
            self.name = name

        def startup(self, context):
            del context

        def __call__(self, acquisition, data):
            del acquisition
            seen.append(self.name)
            return data + 1

    class Sink(ReconPlugin):
        def recon(self, branch, context):
            del branch, context

    plugin = Sink(chain=[Note("first"), Note("second")], buffered=False).spawn()
    plugin.startup(context())
    (line,) = readouts(seed=18)
    assert plugin.process(acquisition(line), line)[0, 0] == pytest.approx(
        line[0, 0] + 2
    )
    assert seen == ["first", "second"]


def test_a_plugin_reaches_its_own_gadget_by_kind():
    """A calibration branch hands its filled buffer to the compression, and has
    to ask this stream for it rather than hold the template's."""

    class Sink(ReconPlugin):
        def recon(self, branch, context):
            del branch, context

    template = Sink(chain=[NoiseAdjust(), CoilCompression(2, key="basis")])
    stream = template.spawn()

    assert stream.gadget(CoilCompression) is stream.chain[1]
    assert stream.gadget(CoilCompression) is not template.chain[1]
    with pytest.raises(LookupError):
        stream.gadget(RampSampling)


def test_a_declared_branch_is_what_recon_is_given():
    """The routing is the mapping, read in the order it was written."""
    branch: list[str] = []

    class Sink(ReconPlugin):
        def recon(self, name, context):
            del context
            branch.append(name)

    plugin = Sink(
        branches={AcquisitionFlag.LAST_IN_SLICE: "imaging"}, buffered=False
    ).spawn()
    scan = context()
    plugin.startup(scan)

    (line,) = readouts(seed=19)
    plugin.receive(acquisition(line), scan)
    assert branch == []
    plugin.receive(acquisition(line, flags=[ismrmrd.ACQ_LAST_IN_SLICE]), scan)
    assert branch == ["imaging"]
