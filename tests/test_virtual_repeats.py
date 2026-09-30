"""Runs of repetitions the virtual scanner plays from each isochromat's map, against the same blocks played one by one."""

import shutil
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _zoo import SMALL
from pypulseqpp import sequences

from pulserver import ir, virtual
from pulserver.virtual import _bloch
from pulserver.virtual import _isochromats as _engine

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SYSTEM = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
RNG = np.random.default_rng(11)
POSITIONS = RNG.uniform(-0.05, 0.05, size=(60, 3))
TISSUE = {"t1": 0.9, "t2": 0.09, "off_resonance": RNG.uniform(-30.0, 30.0, 60)}
#: Sequences whose imaging repeats one repetition with other phase offsets and
#: phase encodings, a spin echo and an inversion-prepared train among them.
REPEATING = [
    "bssfp3D_sequence",
    "gre3D_sequence",
    "gre_multiecho2D_sequence",
    "se2D_sequence",
    "fse3D_sequence",
    "mprage3D_sequence",
]


def _designed(name, tmp_path_factory):
    path = tmp_path_factory.mktemp(name) / "scan.seq"
    getattr(sequences, name)(**SMALL[name]).write(str(path))
    ir.convert(path, pp.Opts())
    return path


@pytest.fixture(scope="module", params=REPEATING)
def repeating(request, tmp_path_factory):
    return _designed(request.param, tmp_path_factory)


def _isochromats():
    return virtual.Isochromats(POSITIONS, **TISSUE)


def _written(seq, path):
    seq.write(str(path))
    ir.convert(path, pp.Opts(B0=3.0))
    return path


def _readout():
    gx = pp.make_trapezoid("x", flat_area=320.0, flat_time=1.28e-3, system=SYSTEM)
    adc = pp.make_adc(32, duration=gx.flat_time, delay=gx.rise_time, system=SYSTEM)
    return gx, adc


def _balanced(path, rewinder=1.0, count=24):
    """A balanced gradient echo whose phase encodings are rewound at ``rewinder`` times their amplitude."""
    seq = pp.Sequence(SYSTEM)
    rf = pp.make_block_pulse(np.deg2rad(30), duration=0.2e-3, system=SYSTEM)
    gx, adc = _readout()
    pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=0.8e-3, system=SYSTEM)
    lobe = pp.make_trapezoid(
        "y", area=10.0 * (count // 2), duration=0.8e-3, system=SYSTEM
    )
    for n in range(count):
        gy = pp.scale_grad(lobe, (n - count // 2) / (count // 2))
        rf.phase_offset = adc.phase_offset = np.pi * (n % 2)
        seq.add_block(rf)
        seq.add_block(pre, gy)
        seq.add_block(gx, adc)
        seq.add_block(pre, pp.scale_grad(gy, -rewinder))
    return _written(seq, path)


def _encoded_before_refocusing(path, count=16):
    """A spin echo phase-encoded between its excitation and refocusing pulses, rewound after its readout."""
    seq = pp.Sequence(SYSTEM)
    excitation = pp.make_block_pulse(np.pi / 2, duration=0.2e-3, system=SYSTEM)
    refocusing = pp.make_block_pulse(np.pi, duration=0.4e-3, system=SYSTEM)
    gx, adc = _readout()
    pre = pp.make_trapezoid("x", area=gx.area / 2, duration=0.8e-3, system=SYSTEM)
    lobe = pp.make_trapezoid(
        "y", area=10.0 * (count // 2), duration=0.8e-3, system=SYSTEM
    )
    for n in range(count):
        gy = pp.scale_grad(lobe, (n - count // 2) / (count // 2))
        seq.add_block(excitation)
        seq.add_block(pre, gy)
        seq.add_block(refocusing)
        seq.add_block(gx, adc)
        seq.add_block(pp.scale_grad(gy, -1.0))
        seq.add_block(pp.make_delay(20e-3))
    return _written(seq, path)


def _one_by_one(path, monkeypatch, **options):
    """Every readout of the cache played block by block: the player finding no run."""
    with monkeypatch.context() as patched:
        patched.setattr(_bloch, "runs", lambda *args, **kwargs: [])
        return virtual.simulate(path, _isochromats(), **options)


def test_runs_of_repetitions_sample_what_their_blocks_played_one_by_one_sample(
    repeating, monkeypatch
):
    player = _bloch.Player(repeating, _isochromats())
    assert player.runs
    repeated = np.concatenate(virtual.simulate(repeating, _isochromats()), axis=1)
    played = np.concatenate(_one_by_one(repeating, monkeypatch), axis=1)
    # The cache's single-precision amplitudes leave the phase encodings'
    # net areas at rounding, which repetitions take as zero.
    np.testing.assert_allclose(
        repeated, played, rtol=0, atol=1e-6 * np.abs(played).max()
    )


def test_a_balanced_sequence_plays_as_runs_of_one_repetition_but_for_its_preparation(
    tmp_path_factory,
):
    path = _designed("bssfp3D_sequence", tmp_path_factory)
    player = _bloch.Player(path, _isochromats())
    covered = sum(run.count * run.size for run in player.runs)
    assert covered >= 0.9 * player.blocks
    assert {run.size for run in player.runs} == {3}


def test_phase_encodings_rewound_to_the_precision_of_the_pulseq_format_play_as_one_run_to_a_tolerance(
    tmp_path,
):
    """The six significant digits a Pulseq file keeps of an amplitude leave a repetition's encoding a net area of a few millionths of its largest."""
    path = _balanced(tmp_path / "rounded.seq", rewinder=1.0 + 3e-6)
    assert all(run.count < 24 for run in _bloch.Player(path, _isochromats()).runs)
    player = _bloch.Player(path, _isochromats(), tolerance=1e-4)
    assert [(run.first, run.size, run.count) for run in player.runs] == [(0, 4, 24)]


def test_a_phase_encoding_played_across_a_refocusing_pulse_plays_block_by_block(
    tmp_path, monkeypatch
):
    """A repetition's maps take its encodings' area at each window as all they do, which a pulse between the encoding and the window breaks."""
    path = _encoded_before_refocusing(tmp_path / "se.seq")
    assert not _bloch.Player(path, _isochromats()).runs
    np.testing.assert_array_equal(
        np.concatenate(virtual.simulate(path, _isochromats()), axis=1),
        np.concatenate(_one_by_one(path, monkeypatch), axis=1),
    )


def _windows_held(player):
    adc = player.played["adc"]
    return [int(adc[run.first : run.first + run.size].sum()) for run in player.runs]


def test_repetitions_hold_no_more_windows_than_their_maps_have_memory_for(
    tmp_path_factory, monkeypatch
):
    """With room for the maps of one window, a three-echo gradient echo's echoes play block by block."""
    path = _designed("gre_multiecho2D_sequence", tmp_path_factory)
    spins = _isochromats()
    assert max(_windows_held(_bloch.Player(path, spins))) == 3
    monkeypatch.setattr(_bloch, "MEMORY", len(spins) * (264 + 236 + 16 * spins.coils))
    assert _bloch._windows(spins) == 1
    assert max(_windows_held(_bloch.Player(path, spins)), default=0) <= 1


def test_a_run_keeps_no_maps_once_its_last_repetition_has_played(tmp_path_factory):
    """Consecutive runs, dummy excitations and then the imaging, hold one run's maps at a time."""
    path = _designed("gre3D_sequence", tmp_path_factory)
    player = _bloch.Player(path, _isochromats())
    assert len(player.runs) > 1
    for run in player.runs:
        list(player.readouts(run.first, run.stop))
        assert player._playing is None


@pytest.mark.parametrize("memory", [_engine.MEMORY, 0], ids=["columns", "no-room"])
def test_runs_played_to_a_tolerance_sample_to_it(tmp_path_factory, monkeypatch, memory):
    """Without room for the fixed points' column sums, every transient is carried."""
    monkeypatch.setattr(_engine, "MEMORY", memory)
    path = _designed("bssfp3D_sequence", tmp_path_factory)
    exact = np.concatenate(virtual.simulate(path, _isochromats()), axis=1)
    within = np.concatenate(
        virtual.simulate(path, _isochromats(), tolerance=1e-4), axis=1
    )
    assert np.linalg.norm(within - exact) < 1e-3 * np.linalg.norm(exact)


def test_pulses_whose_phases_step_unevenly_within_a_repetition_play_block_by_block(
    tmp_path,
):
    """An interleaved multislice scan whose RF spoiling steps each slice's pulse by its own increment."""
    shutil.copy(FIXTURES / "gre_2d_3sl.seq", tmp_path)
    ir.convert(tmp_path / "gre_2d_3sl.seq", pp.Opts(B0=3.0))
    assert not _bloch.Player(tmp_path / "gre_2d_3sl.seq", _isochromats()).runs


def test_readout_gradients_that_vary_play_block_by_block(tmp_path_factory):
    """Radial spokes: the dummy scans before them repeat, the spokes do not."""
    path = _designed("gre_radial2D_sequence", tmp_path_factory)
    played = ir.playout(path)["blocks"]
    runs = _bloch.Player(path, _isochromats()).runs
    assert runs
    assert not any(played["adc"][run.first : run.stop].any() for run in runs)


def test_a_run_played_across_spans_samples_what_it_samples_played_whole(
    tmp_path_factory,
):
    """Spans end where a repetition starts, so a run's repetitions play whole in any of them."""
    path = _designed("bssfp3D_sequence", tmp_path_factory)
    scan = virtual.Scan(path, _isochromats())
    chunks = list(scan.chunks(0.02))
    assert len(chunks) > 2 * len(scan._player.runs)
    streamed = [readout for chunk in chunks for readout in chunk.readouts]
    whole = virtual.simulate(path, _isochromats())
    assert len(streamed) == len(whole)
    assert all(np.array_equal(a, b) for a, b in zip(streamed, whole, strict=True))
