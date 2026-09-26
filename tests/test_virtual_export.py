"""The blocks a cache plays, written as a Pulseq file: what a simulator reads plays what the cache plays."""

import math
import re
import shutil
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _virtual import ORIENTATIONS
from _zoo import SMALL
from pypulseqpp import sequences

from pulserver import ir, virtual

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SYSTEM = pp.Opts(B0=3.0)
# A small fraction of the k-space spacing of every sequence here.
K_TOLERANCE = 1e-3
# The text format writes a gradient's amplitude to six significant figures,
# and a turned gradient's amplitude is written as the rotation leaves it.
K_PRECISION = 1e-5
STANDARD_SECTIONS = {
    "VERSION",
    "DEFINITIONS",
    "BLOCKS",
    "RF",
    "GRADIENTS",
    "TRAP",
    "ADC",
    "SHAPES",
    "SIGNATURE",
}


def _exported(path, target, rotation=None):
    """Convert ``path``, export its cache to ``target`` and read the file back."""
    ir.convert(path, SYSTEM)
    phases = virtual.export(path, target, SYSTEM, rotation=rotation)
    back = pp.Sequence(SYSTEM)
    back.read(str(target))
    return back, phases


def _assert_same_trajectory(exported, played):
    atol = K_TOLERANCE + K_PRECISION * np.abs(played).max()
    np.testing.assert_allclose(exported, played, rtol=0, atol=atol)


def _fixture(name, tmp_path):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    return tmp_path / name


@pytest.fixture(scope="module", params=sorted(SMALL))
def design(request, tmp_path_factory):
    name = request.param
    path = tmp_path_factory.mktemp(name) / "scan.seq"
    getattr(sequences, name)(**SMALL[name]).write(str(path))
    return path


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
def test_every_shipped_sequence_exports_the_trajectory_its_cache_plays(
    design, rotation, tmp_path
):
    back, _ = _exported(design, tmp_path / "exported.seq", rotation)

    played = np.concatenate(virtual.trajectory(design, rotation=rotation), axis=1)
    _assert_same_trajectory(back.calculate_kspace()[0], played)


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
@pytest.mark.parametrize(
    "name", sorted(p.name for p in FIXTURES.glob("*.seq") if not p.stem.endswith("_b"))
)
def test_every_fixture_exports_the_trajectory_its_cache_plays(name, rotation, tmp_path):
    path = _fixture(name, tmp_path)
    back, _ = _exported(path, tmp_path / "exported.seq", rotation)

    played = np.concatenate(virtual.trajectory(path, rotation=rotation), axis=1)
    _assert_same_trajectory(back.calculate_kspace()[0], played)


def test_every_shipped_sequence_exports_each_block_as_long_as_its_cache_plays_it(
    design, tmp_path
):
    back, _ = _exported(design, tmp_path / "exported.seq")

    played = ir.play(design)["duration_us"]
    exported = [back.block_durations[i] for i in range(1, len(back.block_events) + 1)]
    np.testing.assert_allclose(1e6 * np.array(exported), played, atol=1e-6)


def test_a_pulse_keeps_its_samples_offsets_centre_and_use(tmp_path):
    """A sinc pulse on the raster, a block pulse drawn by its two corners, a
    pulse sampled every two rasters and a fat saturation pulse offset in ppm."""
    raster = SYSTEM.rf_raster_time
    pulses = [
        pp.make_sinc_pulse(
            math.pi / 6,
            duration=2e-3,
            freq_offset=1500.0,
            phase_offset=0.7,
            use="excitation",
            system=SYSTEM,
        ),
        pp.make_block_pulse(
            math.pi / 6,
            duration=1e-3,
            freq_offset=-800.0,
            phase_offset=-1.1,
            use="excitation",
            system=SYSTEM,
        ),
        pp.make_arbitrary_rf(
            np.sinc(np.linspace(-2.0, 2.0, 200)).astype(complex),
            math.pi / 6,
            dwell=2 * raster,
            delay=SYSTEM.rf_dead_time,
            use="excitation",
            system=SYSTEM,
        ),
        pp.make_gauss_pulse(
            math.pi / 2, duration=4e-3, freq_ppm=-3.45, use="saturation", system=SYSTEM
        ),
    ]
    seq = pp.Sequence(SYSTEM)
    for pulse in pulses:
        seq.add_block(pulse)
    path = tmp_path / "pulses.seq"
    seq.write(str(path))

    back, _ = _exported(path, tmp_path / "exported.seq")

    for block, pulse in enumerate(pulses, start=1):
        rf = back.get_block(block).rf
        frequency, phase = pp.calc_absolute_offsets(pulse, system=SYSTEM)
        np.testing.assert_allclose(rf.t, pulse.t, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            rf.signal, pulse.signal, rtol=0, atol=1e-5 * np.abs(pulse.signal).max()
        )
        assert (rf.freq_offset, rf.phase_offset) == pytest.approx((frequency, phase))
        assert rf.freq_ppm == rf.phase_ppm == 0.0
        assert (rf.delay, rf.center) == pytest.approx((pulse.delay, pulse.center))
        assert rf.use == pulse.use


def test_a_ptx_pulse_plays_the_sum_of_its_channels(tmp_path):
    raster = SYSTEM.rf_raster_time
    envelope = 200.0 * np.hanning(100)
    channels = np.stack([envelope * np.exp(0.3j), 0.5 * envelope * np.exp(-1.2j)])
    seq = pp.Sequence(SYSTEM)
    seq.add_block(
        pp.make_ptx_pulse(
            channels, delay=SYSTEM.rf_dead_time, use="excitation", system=SYSTEM
        )
    )
    path = tmp_path / "ptx.seq"
    seq.write(str(path))

    back, _ = _exported(path, tmp_path / "exported.seq")

    rf = back.get_block(1).rf
    np.testing.assert_allclose(rf.t, raster * (np.arange(100) + 0.5), atol=1e-12)
    np.testing.assert_allclose(
        rf.signal, channels.sum(axis=0), rtol=0, atol=1e-5 * envelope.max()
    )


def test_a_gradient_that_starts_and_ends_away_from_zero_inside_its_block_steps_there(
    tmp_path,
):
    """A lobe held at its first and last samples out to its edges, beside a trapezoid spanning it."""
    raster = SYSTEM.grad_raster_time
    count = 20
    lobe = pp.make_arbitrary_grad(
        "x",
        2e4 * np.sin(np.pi * (np.arange(count) + 0.5) / count),
        first=0,
        last=0,
        delay=50 * raster,
        system=SYSTEM,
    )
    spanning = pp.make_trapezoid(
        "y", amplitude=1e4, rise_time=1e-4, flat_time=1.8e-3, system=SYSTEM
    )
    seq = pp.Sequence(SYSTEM)
    seq.add_block(spanning, lobe)
    seq.add_block(pp.make_adc(8, duration=8e-5, system=SYSTEM))
    path = tmp_path / "lobe.seq"
    seq.write(str(path))

    back, _ = _exported(path, tmp_path / "exported.seq")

    played = np.concatenate(virtual.trajectory(path), axis=1)
    _assert_same_trajectory(back.calculate_kspace()[0], played)


def test_the_receiver_phase_is_the_adc_offsets_advancing_from_its_start_and_its_modulation(
    tmp_path,
):
    samples, dwell = 32, 1e-5
    modulation = np.linspace(0.0, 2.0, samples)
    adc = pp.make_adc(
        samples,
        dwell=dwell,
        delay=1e-4,
        freq_offset=250.0,
        phase_offset=0.4,
        phase_modulation=modulation,
        system=SYSTEM,
    )
    seq = pp.Sequence(SYSTEM)
    seq.add_block(pp.make_block_pulse(math.pi / 12, duration=1e-3, system=SYSTEM))
    seq.add_block(adc)
    path = tmp_path / "readout.seq"
    seq.write(str(path))

    back, (phase,) = _exported(path, tmp_path / "exported.seq")

    since = dwell * (np.arange(samples) + 0.5)
    np.testing.assert_allclose(
        phase, 0.4 + 2 * math.pi * 250.0 * since + modulation, atol=1e-5
    )
    window = back.get_block(2).adc
    assert window.freq_offset == window.phase_offset == 0.0
    assert window.num_samples == samples
    assert window.delay == pytest.approx(1e-4)


def test_the_file_is_pulseq_1_5_1_with_standard_sections_alone(tmp_path):
    path = _fixture("epi_2d_main.seq", tmp_path)
    target = tmp_path / "exported.seq"

    _exported(path, target)

    text = target.read_text()
    assert re.search(r"\[VERSION\]\s+major 1\s+minor 5\s+revision 1", text)
    assert set(re.findall(r"^\[(\w+)\]", text, re.MULTILINE)) <= STANDARD_SECTIONS
