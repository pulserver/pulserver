"""The subspace reconstruction of shuffled trains: the basis it simulates from the streamed description, and the echoes it solves for."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("bartorch")
blochsim = pytest.importorskip("blochsim")
pp = pytest.importorskip("pypulseqpp")

import torch  # noqa: E402
from blochsim import sequence as S  # noqa: E402

from pulserver.proxy._seqdesc import describe, waveforms  # noqa: E402
from pulserver.recon import ExamCache, ReconContext  # noqa: E402
from pulserver.recon.handlers.subspace import (  # noqa: E402
    FseSubspaceRecon,
    _by_contrast,
    _by_echo,
    _echoes,
)

SYSTEM = pp.Opts(B0=7.0)
ESP, ECHOES, TR = 8e-3, 16, 1.0


def cpmg_train():
    """A CPMG train of ``ECHOES`` hard refocusing pulses ``ESP`` apart, as pypulseqpp plays it, with no crushers."""
    seq = pp.Sequence(SYSTEM)
    excitation = pp.make_block_pulse(
        np.pi / 2,
        duration=0.5e-3,
        phase_offset=np.pi / 2,
        system=SYSTEM,
        use="excitation",
    )
    refocusing = pp.make_block_pulse(
        np.pi, duration=1e-3, system=SYSTEM, use="refocusing"
    )
    adc = pp.make_adc(num_samples=32, duration=1e-3, system=SYSTEM)
    seq.add_block(excitation)
    seq.add_block(pp.make_delay(ESP / 2 - 0.75e-3))
    for _ in range(ECHOES):
        seq.add_block(refocusing)
        seq.add_block(pp.make_delay(ESP / 2 - 1e-3))
        seq.add_block(adc)
        seq.add_block(pp.make_delay(ESP / 2 - 1e-3))
    seq.add_block(pp.make_delay(TR - seq.duration()[0]))
    return seq


def context():
    return ReconContext(header=SimpleNamespace(encoding=[]), exam=ExamCache("exam"))


def test_every_refocusing_pulse_and_no_other_event_sits_between_crushers():
    (description,) = blochsim.description_from_waveforms(
        waveforms([describe(cpmg_train())])
    )
    interpreted = FseSubspaceRecon().interpret(description)
    crushed = S.EventAction.CRUSH_BEFORE | S.EventAction.CRUSH_AFTER
    for event in interpreted.events:
        refocusing = (
            event.type == S.EventType.RF and event.params[1] == S.RfUse.REFOCUSING
        )
        assert (event.action == crushed) == refocusing


def test_the_basis_simulated_from_the_streamed_train_is_that_of_the_ideal_cpmg_train():
    plugin = FseSubspaceRecon(rank=4)
    streamed = plugin.basis(context(), tuple(waveforms([describe(cpmg_train())])))
    ideal = S.fse_description(torch.full((ECHOES,), np.pi), ESP, repetition_time_s=TR)
    expected = S.simulate_subspace(
        ideal, plugin.tissue(), rank=4, repetitions=3
    ).basis.T

    shared = torch.linalg.svdvals(expected.conj() @ streamed.T)

    assert streamed.shape == (4, ECHOES)
    assert float(shared.min()) > 0.999


def test_a_stream_without_a_description_is_refused_with_its_reason():
    with pytest.raises(ValueError, match="no sequence description"):
        FseSubspaceRecon().basis(context(), ())


class Placed:
    """A buffer stand-in: lines placed at ``(z, y)`` with the readouts ``kspace``."""

    def __init__(self, kspace, placed):
        self.kspace = kspace
        self.mask = np.zeros(kspace.shape[1:], bool)
        for z, y in placed:
            self.mask[z, y] = True

    def position(self, acquisition):
        return acquisition.where


def test_each_line_is_solved_in_the_frame_of_the_echo_it_was_read_at():
    kspace = np.arange(2 * 3 * 4 * 5, dtype=np.complex64).reshape(2, 3, 4, 5)
    placed = {(0, 1): 2, (2, 3): 0, (1, 0): 2}
    buffer = Placed(kspace, placed)
    acquisitions = [
        SimpleNamespace(where=where, idx=SimpleNamespace(contrast=echo))
        for where, echo in placed.items()
    ]

    positions, samples = _by_echo(buffer, _echoes(buffer, acquisitions), frames=3)

    assert positions.shape == (3, 2, 2)
    assert positions[0, 0].tolist() == [2, 3] and positions[0, 1].tolist() == [-1, -1]
    assert positions[1].tolist() == [[-1, -1], [-1, -1]]
    assert sorted(map(tuple, positions[2].tolist())) == [(0, 1), (1, 0)]
    for frame in range(3):
        for shot, (z, y) in enumerate(positions[frame].tolist()):
            if z >= 0:
                np.testing.assert_array_equal(samples[:, frame, shot], kspace[:, z, y])


def test_a_buffer_laid_out_by_contrast_is_solved_one_frame_per_contrast():
    kspace = np.arange(2 * 3 * 2 * 4 * 5, dtype=np.complex64).reshape(2, 3, 2, 4, 5)
    mask = np.zeros(kspace.shape[1:], bool)
    mask[0, 1, 2] = mask[0, 0, 3] = mask[2, 1, 0] = True
    buffer = SimpleNamespace(kspace=kspace, mask=mask)

    positions, samples = _by_contrast(buffer, frames=4)

    assert positions.shape == (4, 2, 2)
    assert sorted(map(tuple, positions[0].tolist())) == [(0, 3), (1, 2)]
    assert positions[1].tolist() == [[-1, -1], [-1, -1]]
    assert positions[2, 0].tolist() == [1, 0]
    assert positions[3].tolist() == [[-1, -1], [-1, -1]]
    for frame in range(3):
        for shot, (z, y) in enumerate(positions[frame].tolist()):
            if z >= 0:
                np.testing.assert_array_equal(
                    samples[:, frame, shot], kspace[:, frame, z, y]
                )
