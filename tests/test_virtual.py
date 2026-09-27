"""The virtual scanner: the cache played as each file designs it, and phantoms scanned through it."""

import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import ismrmrd
import numpy as np
import pypulseqpp as pp
import pytest
from _host import ANY_ORIENTATION, generate
from _virtual import (
    OBLIQUE,
    OFF_RESONANCE_HZ,
    OFFSET,
    ORIENTATIONS,
    REFLECTED,
    phantom,
    posed,
    precession,
    water_and_fat,
)
from pypulseqpp import sequences
from pypulseqpp.sequences.preparation.fatsat import FAT_SHIFT_PPM
from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp
from pypulseqpp.sequences.sequence.se2D_sequence import Se2DApp

from pulserver import ir, virtual
from pulserver.host import DesignStore
from pulserver.mrd import read_chain
from pulserver.protocol import FOV_OFFSET, FOV_ROTATION
from pulserver.proxy import ReconProxy, SequenceTable

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "sequences"
RECON_PLUGINS = ROOT / "tests" / "recon_plugins"
SEQUENCES = [
    "dedup_gre_pair.seq",
    "epi_2d_main.seq",
    "gre_2d_3sl.seq",
    "mprage_stack_of_spirals_3d.seq",
    "zte_3d.seq",
]
SYSTEM = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=170.0,
    slew_unit="T/m/s",
    B0=3.0,
    rf_raster_time=1e-6,
    grad_raster_time=1e-5,
    adc_raster_time=1e-7,
    block_duration_raster=1e-5,
)
# A small fraction of the k-space spacing of every fixture.
K_TOLERANCE = 1e-3
MATRIX = 32
DEADLINE = 180.0


def _written(path, sequence):
    sequence.write(str(path))
    ir.convert(path, SYSTEM)
    return sequence


def _copy(name, directory):
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    return directory / name


def _designed_trajectory(seq, rotation=None):
    """The k-space location of every ADC sample of a chain as its files design it, turned by ``rotation`` as the checks turn it."""
    files = [sequence for _, sequence in read_chain(seq)]
    if rotation is not None:
        files = [pp.TransformFOV(rotation=rotation).apply_to_sequence(s) for s in files]
    return np.concatenate(
        [sequence.calculate_kspace()[0] for sequence in files], axis=1
    )


def _ideal(seq, phantom, field_t=None, off_resonance_hz=0.0, longitudinal=None):
    """What an unshifted, unrotated prescription acquires of ``phantom``, one array per readout.

    Each chemical shift precesses at its frequency as the files time their RF
    pulses, with ``longitudinal[shift]`` of its magnetization, all of it by
    default.
    """
    k = _designed_trajectory(seq)
    accrued = np.concatenate([precession(sequence) for _, sequence in read_chain(seq)])
    per_ppm = 0.0 if field_t is None else 1e-6 * pp.Opts().gamma * field_t
    signal = sum(
        (1.0 if longitudinal is None else longitudinal[shift])
        * phantom.kspace(k, shift)
        * np.exp(
            -2j * np.pi * (per_ppm * shift + off_resonance_hz) * np.nan_to_num(accrued)
        )
        for shift in phantom.shifts_ppm
    )
    sizes = [
        int(sequence.get_block(index).adc.num_samples)
        for _, sequence in read_chain(seq)
        for index in range(1, len(sequence) + 1)
        if sequence.get_block(index).adc is not None
    ]
    return np.split(signal, np.cumsum(sizes)[:-1], axis=1)


def _residual(acquired, ideal):
    """The misfit of ``acquired`` to ``ideal`` scaled by one complex factor, relative to ``ideal``."""
    a, b = np.concatenate(acquired, axis=1), np.concatenate(ideal, axis=1)
    factor = np.vdot(b, a) / np.vdot(b, b)
    return np.linalg.norm(a - factor * b) / np.linalg.norm(b), abs(factor)


def _excited(acquired, ideal):
    """The readouts after the first excitation of their file, of both."""
    excited = [i for i, samples in enumerate(acquired) if np.abs(samples).any()]
    return [acquired[i] for i in excited], [ideal[i] for i in excited]


def _fat_saturated_epi(path, system):
    """A 2D EPI that saturates fat before every shot, written to ``path`` and converted at ``system.B0``."""
    sequences.epi2D_sequence(n_x=32, n_y=16, n_dummy=0, fat_saturation=True).write(
        str(path)
    )
    ir.convert(path, system)
    design = pp.Sequence()
    design.read(str(path))
    return design


def _left_by_saturation(design, system, field_t, phantom):
    """The z component the designed saturation pulse, resolved at ``system.B0``, leaves of each shift at ``field_t``.

    From pypulseqpp's ``sim_rf`` of the pulse as the design holds it.
    """
    rf = next(
        block.rf
        for block in (design.get_block(i) for i in range(1, len(design) + 1))
        if block.rf is not None and block.rf.use == "saturation"
    )
    frequency, phase = pp.calc_absolute_offsets(rf, system=system)
    resolved = SimpleNamespace(
        t=np.asarray(rf.t),
        signal=np.asarray(rf.signal),
        shape_dur=rf.shape_dur,
        center=rf.center,
        delay=rf.delay,
        freq_offset=frequency,
        phase_offset=phase,
        freq_ppm=0.0,
        phase_ppm=0.0,
        use=rf.use,
    )
    longitudinal, _, frequencies = pp.sim_rf(resolved, df=0.1, dt=1e-6)[:3]
    per_ppm = 1e-6 * pp.Opts().gamma * field_t
    return {
        shift: float(np.interp(per_ppm * shift, frequencies, longitudinal))
        for shift in phantom.shifts_ppm
    }


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
@pytest.mark.parametrize("name", SEQUENCES)
def test_the_played_trajectory_is_the_one_each_file_designs_turned_as_it_is_checked(
    name, rotation, tmp_path
):
    seq = _copy(name, tmp_path)
    ir.convert(seq, SYSTEM)
    played = np.concatenate(virtual.trajectory(seq, rotation=rotation), axis=1)
    np.testing.assert_allclose(
        played, _designed_trajectory(seq, rotation), atol=K_TOLERANCE
    )


def test_the_prescription_turns_a_block_after_its_own_rotation_unless_it_is_labelled_norot(
    tmp_path,
):
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rf = pp.make_block_pulse(np.pi / 2, duration=2e-4, system=SYSTEM)
    prephaser = pp.make_trapezoid("x", area=-500.0, duration=1e-3, system=SYSTEM)
    readout = pp.make_trapezoid("x", flat_area=1000.0, flat_time=2e-3, system=SYSTEM)
    adc = pp.make_adc(
        num_samples=64,
        duration=readout.flat_time,
        delay=readout.rise_time,
        system=SYSTEM,
    )
    seq = pp.Sequence(SYSTEM)
    # The prephaser plays unturned before a turned readout, then turned before
    # an unturned one; the readout carries a rotation of its own throughout.
    for norot in (1, 0):
        seq.add_block(rf, pp.make_label(label="NOROT", type="SET", value=0))
        seq.add_block(prephaser, pp.make_label(label="NOROT", type="SET", value=norot))
        seq.add_block(
            readout,
            adc,
            pp.make_rotation(quarter_turn),
            pp.make_label(label="NOROT", type="SET", value=1 - norot),
        )
    path = tmp_path / "scan.seq"
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    played = np.concatenate(virtual.trajectory(path, rotation=REFLECTED), axis=1)
    np.testing.assert_allclose(
        played, _designed_trajectory(path, REFLECTED), atol=K_TOLERANCE
    )
    logical = _designed_trajectory(path)
    assert not np.allclose(played, REFLECTED @ logical, atol=1.0)


@pytest.mark.parametrize("name", SEQUENCES)
def test_the_enrichment_states_the_trajectory_the_scanner_plays(name, tmp_path):
    seq = _copy(name, tmp_path)
    ir.convert(seq, SYSTEM)
    table = SequenceTable.read(seq)
    played = virtual.trajectory(seq)
    assert len(table) == len(played)
    for index, k in enumerate(played):
        np.testing.assert_allclose(table.readout_k(index), k, atol=K_TOLERANCE)


def test_every_spiral_interleave_plays_its_own_gradient_shape(tmp_path):
    seq = _copy("mprage_stack_of_spirals_3d.seq", tmp_path)
    ir.convert(seq, SYSTEM)
    played = ir.play(seq, waveforms=True)
    design = next(iter(read_chain(seq)))[1]
    shapes = set()
    for block in np.flatnonzero(played["adc"]):
        designed = design.get_block(int(block) + 1).gx
        start, stop = played["gradient_span"][block, 0]
        # The raster waveform's end values are held over its outer half intervals.
        waveform = played["gradient_waveform_hz_per_m"][start + 1 : stop - 1]
        np.testing.assert_allclose(waveform, designed.waveform, rtol=1e-5, atol=1.0)
        shapes.add(tuple(np.round(designed.waveform[:4])))
    assert len(shapes) > 1


def test_the_scanner_plays_an_rf_pulse_at_the_centre_its_design_records(tmp_path):
    """A centre away from the magnitude peak is the one the cache carries."""
    rf = pp.make_slr_pulse(np.pi / 2, duration=2e-3, center_pos=0.3, system=SYSTEM)
    seq = pp.Sequence(SYSTEM)
    seq.add_block(rf)
    seq.add_block(pp.make_delay(1e-3))
    path = tmp_path / "scan.seq"
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    played = ir.play(path, waveforms=True)
    peak = float(np.asarray(rf.t)[np.argmax(np.abs(np.asarray(rf.signal)))])
    assert abs(peak - rf.center) > 1e-4
    assert played["rf_center_us"][0] == pytest.approx(
        1e6 * (rf.delay + rf.center), abs=1e-3
    )


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
@pytest.mark.parametrize("application", [Gre2DApp, Se2DApp])
def test_an_object_posed_as_prescribed_is_acquired_as_at_the_isocentre(
    application, rotation, tmp_path
):
    seq = tmp_path / "scan.seq"
    application(SYSTEM, n_x=MATRIX, n_y=MATRIX).design().write(seq)
    ir.convert(seq, SYSTEM, fov_offset=OFFSET)
    residual, gain = _residual(
        virtual.acquire(seq, posed(rotation), rotation=rotation),
        _ideal(seq, phantom()),
    )
    assert residual < 1e-4
    assert gain == pytest.approx(1.0, rel=1e-4)


def _precessed(seq):
    """How far each sample of a chain, acquired off resonance, has precessed from its sample on resonance."""
    tissue = phantom(coils=1)
    on = np.concatenate(virtual.acquire(seq, tissue), axis=1)
    off = np.concatenate(
        virtual.acquire(seq, tissue, off_resonance_hz=OFF_RESONANCE_HZ), axis=1
    )
    accrued = np.concatenate([precession(sequence) for _, sequence in read_chain(seq)])
    expected = on * np.exp(-2j * np.pi * OFF_RESONANCE_HZ * np.nan_to_num(accrued))
    return np.linalg.norm(off - expected) / np.linalg.norm(on)


@pytest.mark.parametrize("name", SEQUENCES)
def test_off_resonance_accrues_as_each_file_times_its_excitation(name, tmp_path):
    seq = _copy(name, tmp_path)
    ir.convert(seq, SYSTEM)
    assert _precessed(seq) < 1e-4


@pytest.mark.parametrize("application", [Gre2DApp, Se2DApp])
def test_off_resonance_accrues_from_the_excitation_and_refocuses_at_the_echo(
    application, tmp_path
):
    seq = tmp_path / "scan.seq"
    application(SYSTEM, n_x=MATRIX, n_y=MATRIX).design().write(seq)
    ir.convert(seq, SYSTEM)
    assert _precessed(seq) < 1e-4


def test_fat_precesses_at_its_chemical_shift_at_the_field_of_the_magnet(tmp_path):
    seq = tmp_path / "scan.seq"
    Gre2DApp(SYSTEM, n_x=MATRIX, n_y=MATRIX).design().write(seq)
    ir.convert(seq, SYSTEM)
    tissue = water_and_fat()
    acquired = virtual.acquire(seq, tissue, field_t=SYSTEM.B0)
    residual, gain = _residual(acquired, _ideal(seq, tissue, field_t=SYSTEM.B0))
    assert residual < 1e-4
    assert gain == pytest.approx(1.0, rel=1e-4)
    unshifted, _ = _residual(acquired, _ideal(seq, tissue))
    assert unshifted > 0.1


def test_a_phantom_with_a_chemical_shift_is_scanned_at_a_field(tmp_path):
    seq = tmp_path / "scan.seq"
    Gre2DApp(SYSTEM, n_x=MATRIX, n_y=MATRIX).design().write(seq)
    ir.convert(seq, SYSTEM)
    with pytest.raises(ValueError, match="field_t"):
        virtual.acquire(seq, water_and_fat())


@pytest.mark.parametrize(
    "build",
    [
        lambda path: _fat_saturated_epi(path, pp.Opts(B0=3.0)),
        lambda path: _written(path, Se2DApp(SYSTEM, n_x=MATRIX, n_y=MATRIX).design()),
    ],
    ids=["fat-saturated-epi", "spin-echo"],
)
def test_the_cache_plays_every_rf_pulse_as_its_design_draws_it(build, tmp_path):
    path = tmp_path / "scan.seq"
    design = build(path)
    played = ir.play(path, waveforms=True)
    pulses = np.flatnonzero(played["rf_amp_hz"])
    assert pulses.size
    for block in pulses:
        rf = design.get_block(int(block) + 1).rf
        start, stop = played["rf_span"][block]
        signal = np.asarray(rf.signal)
        np.testing.assert_allclose(
            played["rf_waveform_hz"][start:stop],
            signal,
            atol=1e-6 * np.abs(signal).max(),
        )
        np.testing.assert_allclose(
            played["rf_time_us"][start:stop], 1e6 * (rf.delay + rf.t), atol=1e-3
        )


def test_fat_saturation_leaves_each_shift_what_its_designed_pulse_leaves_it(tmp_path):
    system = pp.Opts(B0=3.0)
    path = tmp_path / "scan.seq"
    design = _fat_saturated_epi(path, system)
    tissue = water_and_fat(coils=1)
    left = _left_by_saturation(design, system, system.B0, tissue)
    # Without relaxation, fat keeps the cosine of the flip angle at its
    # resonance, and water, a stopband away, nearly all of its magnetization.
    assert left[FAT_SHIFT_PPM] == pytest.approx(np.cos(np.radians(110.0)), abs=1e-2)
    assert left[0.0] == pytest.approx(1.0, abs=1e-2)
    acquired = virtual.acquire(path, tissue, field_t=system.B0)
    ideal = _ideal(path, tissue, field_t=system.B0, longitudinal=left)
    residual, _ = _residual(*_excited(acquired, ideal))
    assert residual < 1e-4


def test_a_fat_saturation_converted_at_another_field_misses_the_fat(tmp_path):
    scanned, converted = pp.Opts(B0=3.0), pp.Opts(B0=1.5)
    path = tmp_path / "scan.seq"
    design = _fat_saturated_epi(path, converted)
    tissue = water_and_fat(coils=1)
    missed = _left_by_saturation(design, converted, scanned.B0, tissue)
    assert missed[FAT_SHIFT_PPM] > 0.9
    acquired = virtual.acquire(path, tissue, field_t=scanned.B0)
    residual, _ = _residual(
        *_excited(
            acquired, _ideal(path, tissue, field_t=scanned.B0, longitudinal=missed)
        )
    )
    assert residual < 1e-4
    saturated = _left_by_saturation(design, scanned, scanned.B0, tissue)
    misfit, _ = _residual(
        *_excited(
            acquired, _ideal(path, tissue, field_t=scanned.B0, longitudinal=saturated)
        )
    )
    assert misfit > 0.1


def test_a_hard_saturation_pulse_acts_over_its_duration(tmp_path):
    """A block pulse is held as the two corners of a time shape, one duration apart."""
    system = pp.Opts(B0=3.0)
    seq = pp.Sequence(system)
    seq.add_block(
        pp.make_block_pulse(
            np.pi / 2,
            duration=8e-3,
            freq_ppm=FAT_SHIFT_PPM,
            use="saturation",
            system=system,
        )
    )
    seq.add_block(pp.make_block_pulse(np.pi / 2, duration=0.5e-3, system=system))
    seq.add_block(pp.make_adc(num_samples=64, duration=6.4e-3, system=system))
    path = tmp_path / "scan.seq"
    seq.write(str(path))
    ir.convert(path, system)
    tissue = water_and_fat(coils=1)
    left = _left_by_saturation(seq, system, system.B0, tissue)
    assert abs(left[FAT_SHIFT_PPM]) < 1e-3
    acquired = virtual.acquire(path, tissue, field_t=system.B0)
    residual, _ = _residual(
        acquired, _ideal(path, tissue, field_t=system.B0, longitudinal=left)
    )
    assert residual < 1e-4


def test_a_readout_before_the_first_excitation_acquires_nothing(tmp_path):
    adc = pp.make_adc(num_samples=64, duration=3.2e-3, system=SYSTEM)
    seq = pp.Sequence(SYSTEM)
    seq.add_block(
        pp.make_block_pulse(np.pi, duration=1e-3, use="refocusing", system=SYSTEM)
    )
    seq.add_block(adc)
    seq.add_block(pp.make_block_pulse(np.pi / 2, duration=0.5e-3, system=SYSTEM))
    seq.add_block(adc)
    path = tmp_path / "scan.seq"
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    before, after = virtual.acquire(path, phantom(coils=1))
    assert not np.abs(before).any()
    assert np.abs(after).all()


def test_a_saturation_band_selected_in_space_is_refused(tmp_path):
    system = pp.Opts(B0=3.0)
    band = sequences.FatSaturation(system, thickness_m=0.05)
    rf = pp.make_block_pulse(np.pi / 2, duration=0.5e-3, system=system)
    adc = pp.make_adc(num_samples=64, duration=3.2e-3, system=system)
    seq = pp.Sequence(system)
    for block in band.blocks:
        seq.add_block(*block)
    seq.add_block(rf)
    seq.add_block(adc)
    path = tmp_path / "scan.seq"
    seq.write(str(path))
    ir.convert(path, system)
    with pytest.raises(ValueError, match="band in space"):
        virtual.acquire(path, phantom(coils=1))


def test_an_object_off_the_prescription_is_not_acquired_centred(tmp_path):
    seq = tmp_path / "scan.seq"
    Gre2DApp(SYSTEM, n_x=MATRIX, n_y=MATRIX).design().write(seq)
    ir.convert(seq, SYSTEM)
    residual, _ = _residual(
        virtual.acquire(seq, phantom(OFFSET)), _ideal(seq, phantom())
    )
    assert residual > 0.5


@pytest.mark.parametrize(
    ("rotation", "turned"),
    [(OBLIQUE, np.eye(3)), (REFLECTED, OBLIQUE)],
    ids=["unturned", "unreflected"],
)
def test_an_object_turned_otherwise_than_prescribed_is_not_acquired_as_at_the_isocentre(
    rotation, turned, tmp_path
):
    seq = tmp_path / "scan.seq"
    Gre2DApp(SYSTEM, n_x=MATRIX, n_y=MATRIX).design().write(seq)
    ir.convert(seq, SYSTEM, fov_offset=OFFSET)
    residual, _ = _residual(
        virtual.acquire(seq, phantom(rotation @ OFFSET, turned), rotation=rotation),
        _ideal(seq, phantom()),
    )
    assert residual > 0.3


@pytest.mark.parametrize(
    "name",
    [
        "epi_2d_main.seq",
        "mprage_stack_of_spirals_3d.seq",
        "zte_3d.seq",
    ],
)
def test_a_readout_under_a_varying_gradient_is_acquired_centred_off_the_isocentre(
    name, tmp_path
):
    seq = _copy(name, tmp_path)
    ir.convert(seq, SYSTEM, fov_offset=OFFSET)
    acquired = virtual.acquire(seq, phantom(OFFSET, coils=1))
    ideal = _ideal(seq, phantom(coils=1))
    excited = [i for i, samples in enumerate(acquired) if np.abs(samples).any()]
    residual, _ = _residual([acquired[i] for i in excited], [ideal[i] for i in excited])
    assert residual < 1e-4


def test_the_cache_carries_the_phase_modulation_of_every_prescribed_readout(tmp_path):
    seq = _copy("epi_2d_main.seq", tmp_path)
    ir.convert(seq, SYSTEM, fov_offset=OFFSET)
    played = ir.play(seq, waveforms=True)
    readouts = np.flatnonzero(played["adc"])
    designed = []
    for _, sequence in read_chain(seq):
        ir.prescribe(sequence, OFFSET)
        designed += [
            np.asarray(block.adc.phase_modulation, dtype=float)
            for block in (sequence.get_block(i) for i in range(1, len(sequence) + 1))
            if block.adc is not None
        ]
    assert len(designed) == readouts.size
    assert any(modulation.size for modulation in designed)
    for block, modulation in zip(readouts, designed, strict=True):
        start, stop = played["adc_modulation_span"][block]
        np.testing.assert_allclose(
            played["adc_phase_modulation_rad"][start:stop], modulation, atol=1e-6
        )


def test_a_phase_modulation_without_one_phase_per_sample_is_refused(tmp_path):
    seq = _copy("zte_3d.seq", tmp_path)
    text = seq.read_text()
    head, adc = text.split("[ADC]\n", 1)
    rows, rest = adc.split("\n\n", 1)
    rows = [
        row if row.startswith("#") else " ".join([*row.split()[:8], "51"])
        for row in rows.splitlines()
    ]
    seq.write_text(head + "[ADC]\n" + "\n".join(rows) + "\n\n" + rest)
    with pytest.raises(ValueError, match="phase modulation has 90"):
        ir.convert(seq, SYSTEM, verify_signature=False)


def _image(readouts, matrix):
    """The root-sum-of-squares image of Cartesian lines in play order, as the FFT plugin makes it."""
    lines = np.stack(readouts, axis=-1)
    coils = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(lines, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)
    )
    combined = np.sqrt((np.abs(coils) ** 2).sum(axis=0))
    start = (combined.shape[0] - matrix) // 2
    return combined[start : start + matrix].T


@pytest.fixture
def proxy(tmp_path):
    running = ReconProxy(tmp_path / "designs", RECON_PLUGINS, slots=1)
    running.bind(0)
    thread = threading.Thread(target=running.serve, daemon=True)
    thread.start()
    yield running
    running.close()
    thread.join(timeout=DEADLINE)


def _scan(proxy, tmp_path, rotation, readouts=None):
    """Design a 2D gradient echo prescribed at ``OFFSET`` and ``rotation``, and scan the phantom posed there."""
    store = DesignStore(tmp_path / "designs")
    values = {"TE": 5000, "nx": MATRIX, "ny": MATRIX}
    values.update(zip(FOV_OFFSET, 1e3 * OFFSET, strict=True))
    values.update(zip(FOV_ROTATION, rotation.ravel(), strict=True))
    design = generate(store, "gre2d", values, limits=ANY_ORIENTATION)
    seq = store.directory(design) / "sequence.seq"
    acquired = virtual.acquire(seq, posed(rotation), rotation=rotation)
    received = virtual.send(
        ("127.0.0.1", proxy.port),
        design,
        acquired if readouts is None else acquired[:readouts],
        position_mm=1e3 * rotation @ OFFSET,
        rotation=rotation,
        timeout=DEADLINE,
    )
    return seq, received


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
def test_a_virtual_scan_reconstructs_the_phantom_where_it_is_prescribed(
    proxy, tmp_path, rotation
):
    """The phantom posed as prescribed is imaged as at the isocentre and placed where it is."""
    seq, received = _scan(proxy, tmp_path, rotation)
    (image,) = [item for item in received if isinstance(item, ismrmrd.Image)]
    reconstructed = np.squeeze(np.abs(image.data)).astype(float)
    expected = _image(_ideal(seq, phantom()), MATRIX)
    reconstructed /= reconstructed.max()
    expected /= expected.max()
    assert np.linalg.norm(reconstructed - expected) / np.linalg.norm(expected) < 1e-3
    np.testing.assert_allclose(image.position, 1e3 * rotation @ OFFSET, atol=1e-4)
    for direction, axis in zip(
        (image.read_dir, image.phase_dir, image.slice_dir), rotation.T, strict=True
    ):
        np.testing.assert_allclose(direction, axis, atol=1e-6)
    fov = pp.Sequence()
    fov.read(str(seq))
    np.testing.assert_allclose(
        image.field_of_view[:2], 1e3 * np.asarray(fov.definitions["FOV"][:2]), rtol=1e-6
    )


def test_a_series_streamed_as_it_is_acquired_is_reconstructed_as_one_sent_whole(
    proxy, tmp_path
):
    store = DesignStore(tmp_path / "designs")
    design = generate(store, "gre2d", {"TE": 5000, "nx": MATRIX, "ny": MATRIX})
    seq = store.directory(design) / "sequence.seq"
    scan = virtual.Scan(seq, phantom().isochromats(2e-3))
    streamed = (readout for chunk in scan.chunks() for readout in chunk.readouts)
    whole = virtual.simulate(seq, phantom().isochromats(2e-3))
    images = []
    for readouts in (streamed, whole):
        received = virtual.send(
            ("127.0.0.1", proxy.port), design, readouts, timeout=DEADLINE
        )
        (image,) = [item for item in received if isinstance(item, ismrmrd.Image)]
        images.append(np.squeeze(np.abs(image.data)).astype(float))
    np.testing.assert_array_equal(images[0], images[1])
    assert images[0].max() > 0.0


def test_a_virtual_series_short_of_a_readout_is_refused(proxy, tmp_path):
    _, received = _scan(proxy, tmp_path, np.eye(3), readouts=MATRIX - 1)
    assert not [item for item in received if isinstance(item, ismrmrd.Image)]
    assert any(isinstance(item, str) and "pulserver:" in item for item in received)
