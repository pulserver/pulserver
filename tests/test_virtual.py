"""The virtual scanner: the cache played as each file designs it, and phantoms scanned through it."""

import shutil
import threading
from pathlib import Path

import ismrmrd
import numpy as np
import pypulseqpp as pp
import pytest
from _host import generate
from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp
from pypulseqpp.sequences.sequence.se2D_sequence import Se2DApp

from pulserver import ir, virtual
from pulserver.host import DesignStore
from pulserver.mrd import read_chain
from pulserver.vre import ReconProxy, SequenceTable

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
OFFSET = np.array([0.02, -0.012, 0.0])
MATRIX = 32
DEADLINE = 180.0


def _copy(name, directory):
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    return directory / name


def _designed_trajectory(seq):
    """The k-space location of every ADC sample of a chain, as its files design it."""
    return np.concatenate(
        [sequence.calculate_kspace()[0] for _, sequence in read_chain(seq)], axis=1
    )


def _phantom(centre, coils=2):
    """An object without a mirror symmetry, centred at ``centre``."""
    centre = np.asarray(centre, dtype=float)
    return virtual.Phantom(
        [
            virtual.Ellipse(tuple(centre), (0.08, 0.06), 0.3),
            virtual.Ellipse(
                tuple(centre + np.array([0.03, 0.02, 0.0])), (0.02, 0.015), 0.0, 0.5
            ),
        ],
        coils=coils,
    )


def _ideal(seq, phantom, offset):
    """What a scanner acquires of ``phantom``, demodulated to ``offset``, one array per readout."""
    k = _designed_trajectory(seq)
    signal = phantom.kspace(k) * np.exp(2j * np.pi * (np.asarray(offset) @ k))
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


@pytest.mark.parametrize("name", SEQUENCES)
def test_the_played_trajectory_is_the_one_each_file_designs(name, tmp_path):
    seq = _copy(name, tmp_path)
    ir.convert(seq, SYSTEM)
    played = np.concatenate(virtual.trajectory(seq), axis=1)
    np.testing.assert_allclose(played, _designed_trajectory(seq), atol=K_TOLERANCE)


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


@pytest.mark.parametrize("application", [Gre2DApp, Se2DApp])
def test_an_object_at_the_prescribed_offset_is_acquired_centred(application, tmp_path):
    seq = tmp_path / "scan.seq"
    application(SYSTEM, n_x=MATRIX, n_y=MATRIX).design().write(seq)
    ir.convert(seq, SYSTEM, fov_offset=OFFSET)
    phantom = _phantom(OFFSET)
    residual, gain = _residual(
        virtual.acquire(seq, phantom), _ideal(seq, phantom, OFFSET)
    )
    assert residual < 1e-4
    assert gain == pytest.approx(1.0, rel=1e-4)


def test_an_object_off_the_prescription_is_not_acquired_centred(tmp_path):
    seq = tmp_path / "scan.seq"
    Gre2DApp(SYSTEM, n_x=MATRIX, n_y=MATRIX).design().write(seq)
    ir.convert(seq, SYSTEM)
    phantom = _phantom(OFFSET)
    residual, _ = _residual(virtual.acquire(seq, phantom), _ideal(seq, phantom, OFFSET))
    assert residual > 0.5


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
    phantom = _phantom(OFFSET, coils=1)
    acquired = virtual.acquire(seq, phantom)
    ideal = _ideal(seq, phantom, OFFSET)
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


def _scan(proxy, tmp_path, offset_mm, readouts=None):
    store = DesignStore(tmp_path / "designs")
    values = {"TE": 5000, "nx": MATRIX, "ny": MATRIX}
    values.update(
        zip(("fov_offset_x", "fov_offset_y", "fov_offset_z"), offset_mm, strict=True)
    )
    design = generate(store, "gre2d", values)
    seq = store.directory(design) / "sequence.seq"
    phantom = _phantom(np.asarray(offset_mm) * 1e-3)
    acquired = virtual.acquire(seq, phantom)
    received = virtual.send(
        ("127.0.0.1", proxy.port),
        design,
        acquired if readouts is None else acquired[:readouts],
        position_mm=offset_mm,
        timeout=DEADLINE,
    )
    return seq, phantom, received


def test_a_virtual_scan_reconstructs_the_phantom_at_its_prescription(proxy, tmp_path):
    offset_mm = tuple(1e3 * OFFSET)
    seq, phantom, received = _scan(proxy, tmp_path, offset_mm)
    (image,) = [item for item in received if isinstance(item, ismrmrd.Image)]
    reconstructed = np.squeeze(np.abs(image.data)).astype(float)
    expected = _image(_ideal(seq, phantom, OFFSET), MATRIX)
    reconstructed /= reconstructed.max()
    expected /= expected.max()
    assert np.linalg.norm(reconstructed - expected) / np.linalg.norm(expected) < 1e-3
    np.testing.assert_allclose(image.position, offset_mm, atol=1e-4)
    fov = pp.Sequence()
    fov.read(str(seq))
    np.testing.assert_allclose(
        image.field_of_view[:2], 1e3 * np.asarray(fov.definitions["FOV"][:2]), rtol=1e-6
    )


def test_a_virtual_series_short_of_a_readout_is_refused(proxy, tmp_path):
    _, _, received = _scan(proxy, tmp_path, (0.0, 0.0, 0.0), readouts=MATRIX - 1)
    assert not [item for item in received if isinstance(item, ismrmrd.Image)]
    assert any(isinstance(item, str) and "pulserver:" in item for item in received)
