"""The virtual scanner's Fourier engine: its timeline, its event streams, and what it acquires."""

import shutil
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
import torch
from _analytic import trajectory
from _virtual import ORIENTATIONS
from bartorch import linop
from pypulseqpp.sequences.preparation.fatsat import FAT_SHIFT_PPM

from pulserver import ir, virtual
from pulserver.virtual import _fourier
from pulserver.virtual._timeline import Timeline

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
#: Converged Bloch simulations of the cases below, the readouts of each joined
#: in play order: isochromats four to a voxel, at the centres of the cells
#: filling it, played block by block through the Bloch equation.
BLOCH = Path(__file__).parent / "fixtures" / "bloch"
SEQUENCES = [
    "dedup_gre_pair.seq",
    "epi_2d_main.seq",
    "gre_2d_3sl.seq",
    "mprage_stack_of_spirals_3d.seq",
    "zte_3d.seq",
]
SYSTEM = pp.Opts(B0=3.0)
#: Steps, in m, between the sheets a slab phantom stacks through its slices:
#: fine enough that the Bloch simulation dephases the orders a spoiler winds.
SHEETS = 2e-5
#: Relative residuals at which the engine agrees with the Bloch simulation on a
#: phantom filling its slices: the interleaved 12 degree slices, fat included;
#: and the 70 degree EPI, whose fat saturation the Bloch simulation spoils only
#: as finely as its
#: isochromats sample the spoiler, on water, the excitation's phase across the
#: slice beyond an ideal rotation's.
AGREEMENT = {"gre_2d_3sl.seq": (True, 0.05), "epi_2d_main.seq": (False, 0.08)}


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Each fixture beside the cache it converts to, at the magnet's field."""
    directory = tmp_path_factory.mktemp("fixtures")
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    for name in SEQUENCES:
        ir.convert(directory / name, SYSTEM)
    return directory


def _slab(half_thickness=8e-3, radius=0.02, fat=True):
    """A water disk, and a fat disk beside it, as sheets stacked through ``half_thickness`` either side of the isocentre."""
    sheets = np.arange(-half_thickness, half_thickness + 0.5 * SHEETS, SHEETS)
    weight = SHEETS / 1e-3
    disks = []
    for z in sheets:
        disks.append(
            virtual.Ellipse(
                (0.0, 0.0, z), (radius, radius), t1=1.0, t2=0.1, intensity=weight
            )
        )
        if not fat:
            continue
        disks.append(
            virtual.Ellipse(
                (0.035, 0.0, z),
                (0.008, 0.008),
                t1=0.35,
                t2=0.07,
                shift_ppm=FAT_SHIFT_PPM,
                intensity=weight,
            )
        )
    return virtual.Phantom(disks)


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
@pytest.mark.parametrize("name", SEQUENCES)
def test_the_timeline_samples_where_the_cache_plays_its_trajectory(
    name, rotation, converted, device
):
    timeline = Timeline(converted / name, rotation=rotation, device=device)
    _, k, _ = timeline.kspace(0, len(timeline.readouts))
    played = np.concatenate(trajectory(converted / name, rotation=rotation), 1)
    excited = np.isfinite(k).all(axis=1)

    assert k.shape == (played.shape[1], 3)
    turned = k[excited] @ rotation.T
    assert np.abs(turned - played.T[excited]).max() < 1e-6 * np.abs(played).max()
    assert np.isnan(played.T[~excited]).all() or not (~excited).any()


def test_a_readouts_echo_is_its_sample_nearest_the_centre_of_k_space(converted):
    timeline = Timeline(converted / "gre_2d_3sl.seq")
    readouts = timeline.readouts
    _, k, _ = timeline.kspace(0, len(readouts))
    for at in range(len(readouts)):
        window = np.linalg.norm(k[readouts.first[at] : readouts.first[at + 1]], axis=1)
        assert readouts.echo[at] == int(np.argmin(window))


def test_a_spoiled_train_shifts_the_states_after_each_readout_and_nowhere_else(
    converted,
):
    timeline = Timeline(converted / "gre_2d_3sl.seq")
    events = _fourier._events(timeline)
    voxel = _fourier._voxel(
        timeline,
        _fourier._Entries.of(_slab().tissue(4e-3, field_t=3.0), timeline.device),
        4e-3,
    )
    shifted = _fourier._shifted(timeline, events, voxel, np.eye(3))

    after_readout = events.kind[:-1] == 2
    assert (shifted[after_readout] == 1).all()
    pulse_to_readout = (events.kind[:-1] == 1) & (events.kind[1:] == 2)
    assert not shifted[pulse_to_readout].any()


def test_a_flat_phantom_is_not_dephased_along_the_axis_it_does_not_span(converted):
    timeline = Timeline(converted / "gre_2d_3sl.seq")
    flat = virtual.Phantom([virtual.Ellipse((0.0, 0.0, 0.0), (0.02, 0.02))]).tissue(
        2e-3, field_t=3.0
    )
    entries = _fourier._Entries.of(flat, timeline.device)
    voxel = _fourier._voxel(timeline, entries, flat.spacing)

    spanned = _fourier._spanned(entries, flat.axes, timeline.rotation, voxel)

    np.testing.assert_array_equal(np.diag(spanned), [1.0, 1.0, 0.0])


def test_each_slice_is_a_station_of_its_own_readouts(converted):
    timeline = Timeline(converted / "gre_2d_3sl.seq")
    selector_of, _ = _fourier._selectors(timeline.pulses)

    stations, station_of = _fourier._stations(timeline, selector_of)

    assert len(stations) == 3
    assert (np.bincount(station_of[station_of >= 0]) > 0).sum() == 3


def test_a_pulse_played_without_a_gradient_selects_by_frequency(converted):
    timeline = Timeline(converted / "epi_2d_main.seq")
    pulses = timeline.pulses
    saturation = pulses.use == 4
    water = virtual.Phantom([virtual.Ellipse((0.0, 0.0, 0.0), (0.02, 0.02))])
    fat = virtual.Phantom(
        [virtual.Ellipse((0.0, 0.0, 0.0), (0.02, 0.02), shift_ppm=FAT_SHIFT_PPM)]
    )
    _, selectors = _fourier._selectors(pulses)
    column = np.flatnonzero(selectors == np.flatnonzero(saturation)[0])

    flips = {
        name: _fourier._profiles(
            _fourier._Entries.of(phantom.tissue(2e-3, field_t=3.0), timeline.device),
            selectors,
            pulses,
        )[:, column]
        for name, phantom in (("water", water), ("fat", fat))
    }

    assert not pulses.gradient[saturation].any()
    assert (flips["water"] == 0).all()
    assert (flips["fat"] > 0).all()


@pytest.mark.parametrize("name", AGREEMENT)
def test_the_fourier_engine_acquires_what_a_bloch_simulation_does_of_a_phantom_filling_its_slices(
    name, converted, device
):
    fat, agreement = AGREEMENT[name]
    phantom = _slab(fat=fat)
    sequence = converted / name
    player = virtual.FourierPlayer(
        sequence, phantom.tissue(3e-3, field_t=3.0), device=device
    )
    fourier = np.concatenate(
        [readout.ravel() for readout in player.readouts(0, player.blocks)]
    )
    bloch = np.load(BLOCH / f"{Path(name).stem}.npy")

    assert fourier.shape == bloch.shape
    assert np.linalg.norm(fourier - bloch) < agreement * np.linalg.norm(bloch)


def test_a_scan_on_a_tissue_plays_through_the_fourier_engine(converted):
    sequence = converted / "gre_2d_3sl.seq"
    tissue = _slab(half_thickness=2e-3).tissue(4e-3, field_t=3.0)

    chunks = list(virtual.Scan(sequence, tissue).chunks(0.1, sound=False))

    readouts = [readout for chunk in chunks for readout in chunk.readouts]
    assert len(readouts) == len(Timeline(sequence).readouts)
    assert all(readout.dtype == np.complex64 for readout in readouts)
    assert chunks[-1].stop == pytest.approx(virtual.Scan(sequence, tissue).duration)


def _steady(kind, path, *, selective, lines=16, dummies=64, matrix=32, fov=0.12):
    """A 2D SSFP whose repetitions wind one z crusher: after the readout ("fid"), before it ("echo") or between two ("dess")."""
    system = pp.Opts(
        max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s", B0=3.0
    )
    seq = pp.Sequence(system)
    flip = np.radians(30.0)
    if selective:
        rf, gz, rephaser = pp.make_sinc_pulse(
            flip,
            duration=2e-3,
            slice_thickness=5e-3,
            time_bw_product=4,
            system=system,
            return_gz=True,
        )
    else:
        rf, gz = pp.make_block_pulse(flip, duration=2e-4, system=system), None
        rephaser = pp.make_trapezoid("z", area=1e-3, duration=4e-4, system=system)
    read = pp.make_trapezoid(
        "x", flat_area=matrix / fov, flat_time=1.6e-3, system=system
    )
    adc = pp.make_adc(
        matrix, duration=read.flat_time, delay=read.rise_time, system=system
    )
    prephaser = pp.make_trapezoid("x", area=-read.area / 2, system=system)
    crusher = pp.make_trapezoid("z", area=4.0 / 5e-3, system=system)
    span = pp.calc_duration(prephaser)

    def readout(area, acquiring):
        encode = pp.make_trapezoid("y", area=area, duration=span, system=system)
        rewind = pp.make_trapezoid("y", area=-area, duration=span, system=system)
        seq.add_block(prephaser, encode)
        seq.add_block(read, adc) if acquiring else seq.add_block(read)
        seq.add_block(prephaser, rewind)

    for at in range(dummies + lines):
        area = (max(at - dummies, lines // 2) - lines // 2) / fov
        seq.add_block(rf, gz) if gz is not None else seq.add_block(rf)
        seq.add_block(rephaser)
        if kind == "echo":
            seq.add_block(crusher)
        readout(area, at >= dummies)
        if kind == "dess":
            seq.add_block(crusher)
            readout(area, at >= dummies)
        if kind == "fid":
            seq.add_block(crusher)
        seq.add_block(pp.make_trapezoid("z", area=rephaser.area, system=system))
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    return path


@pytest.mark.parametrize(
    "kind, pathways", [("fid", {0}), ("echo", {1}), ("dess", {0, 1})]
)
def test_each_readout_reads_the_pathway_that_passes_the_centre_during_it(
    kind, pathways, tmp_path
):
    sequence = _steady(kind, tmp_path / f"{kind}.seq", selective=True)

    assert set(Timeline(sequence).readouts.pathway.tolist()) == pathways


@pytest.mark.parametrize(
    "kind, selective, offset_hz",
    [
        ("fid", False, 0.0),
        ("echo", False, 0.0),
        ("dess", False, 0.0),
        ("echo", True, 0.0),
        ("echo", False, 30.0),
        ("dess", False, 30.0),
    ],
)
def test_the_fourier_engine_acquires_what_a_bloch_simulation_does_of_an_unspoiled_steady_state(
    kind, selective, offset_hz, tmp_path, device
):
    sequence = _steady(kind, tmp_path / f"{kind}.seq", selective=selective)
    water = _slab(half_thickness=6e-3, radius=0.03, fat=False)
    player = virtual.FourierPlayer(
        sequence,
        water.tissue(2e-3, field_t=3.0, off_resonance_hz=offset_hz),
        device=device,
    )
    fourier = np.concatenate(
        [readout.ravel() for readout in player.readouts(0, player.blocks)]
    )
    profile = "selective" if selective else "hard"
    bloch = np.load(BLOCH / f"steady_{kind}_{profile}_{int(offset_hz)}hz.npy")

    assert np.linalg.norm(fourier - bloch) < 0.05 * np.linalg.norm(bloch)


def _balanced(path, *, matrix=32, fov=0.12, lines=32, dummies=200):
    """A 2D bSSFP: hard pulses alternating in phase, every gradient of a repetition balanced."""
    system = pp.Opts(
        max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s", B0=3.0
    )
    seq = pp.Sequence(system)
    rf = pp.make_block_pulse(np.radians(40.0), duration=2e-4, system=system)
    read = pp.make_trapezoid(
        "x", flat_area=matrix / fov, flat_time=1.6e-3, system=system
    )
    adc = pp.make_adc(
        matrix, duration=read.flat_time, delay=read.rise_time, system=system
    )
    prephaser = pp.make_trapezoid("x", area=-read.area / 2, system=system)
    span = pp.calc_duration(prephaser)
    for at in range(dummies + lines):
        rf.phase_offset = adc.phase_offset = np.pi * (at % 2)
        area = (at - dummies - lines // 2) / fov if at >= dummies else 0.0
        seq.add_block(rf)
        seq.add_block(
            prephaser, pp.make_trapezoid("y", area=area, duration=span, system=system)
        )
        seq.add_block(read, adc) if at >= dummies else seq.add_block(read)
        seq.add_block(
            prephaser, pp.make_trapezoid("y", area=-area, duration=span, system=system)
        )
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    return path


def test_a_balanced_steady_state_reads_the_free_induction_as_a_bloch_simulation_does(
    tmp_path, device
):
    sequence = _balanced(tmp_path / "bssfp.seq")
    water = virtual.Phantom(
        [virtual.Ellipse((0.0, 0.0, 0.0), (0.03, 0.03), t1=0.8, t2=0.08)]
    )
    player = virtual.FourierPlayer(
        sequence, water.tissue(2e-3, field_t=3.0), device=device
    )
    fourier = np.concatenate(
        [readout.ravel() for readout in player.readouts(0, player.blocks)]
    )
    bloch = np.load(BLOCH / "bssfp.npy")

    assert not np.any(player._timeline.readouts.pathway)
    assert np.linalg.norm(fourier - bloch) < 0.05 * np.linalg.norm(bloch)


def test_a_stream_plays_its_repetitions_until_they_settle_and_reads_the_rest_off_the_last(
    tmp_path, device, monkeypatch
):
    sequence = _steady("fid", tmp_path / "fid.seq", selective=False)
    tissue = virtual.Phantom(
        [virtual.Ellipse((0.0, 0.0, 0.0), (0.03, 0.03), t1=0.01, t2=0.005)]
    ).tissue(2e-3, field_t=3.0)
    periodic = _fourier._periodic
    counts = {}

    def counted(stream, settle_us):
        played, column = periodic(stream, settle_us)
        counts.update(events=stream.kind.size, played=played.kind.size)
        return played, column

    def acquired():
        player = virtual.FourierPlayer(sequence, tissue, device=device)
        return np.concatenate(
            [readout.ravel() for readout in player.readouts(0, player.blocks)]
        )

    monkeypatch.setattr(_fourier, "_periodic", counted)
    settled = acquired()
    monkeypatch.setattr(
        _fourier,
        "_periodic",
        lambda stream, settle_us: (
            stream,
            np.arange(np.count_nonzero(stream.kind == 2)),
        ),
    )
    whole = acquired()

    assert counts["played"] < counts["events"] / 2
    assert np.linalg.norm(settled - whole) < 1e-3 * np.linalg.norm(whole)


def test_finer_than_its_cubes_a_scan_reads_their_spectrum_off_the_lattice_they_lie_on(
    tmp_path, device
):
    sequence = _steady("fid", tmp_path / "fid.seq", selective=False, matrix=96)
    tissue = virtual.Phantom(
        [virtual.Ellipse((0.01, -0.005, 0.0), (0.03, 0.02), t1=0.8, t2=0.08)],
        coils=4,
    ).tissue(2e-3, field_t=3.0)
    player = virtual.FourierPlayer(sequence, tissue, device=device)
    grid = player._grids[0]
    k = player._timeline.kspace(0, len(player._timeline.readouts))[1]
    k = k[np.isfinite(k).all(axis=1)][::7]

    at = torch.as_tensor(k, device=device)
    spectrum = grid.spectrum(at).cpu().numpy()
    traj = grid.trajectory(at).to(torch.float32)
    held = grid.sensitivity(1, 2) * grid.images[:1]
    read = linop.NUFFT(traj[None], (1, *grid.shape), toeplitz=False)(held)
    read = read.reshape(-1).cpu().numpy() * np.sqrt(grid.points) * spectrum
    read *= np.exp(-2j * np.pi * (k @ grid.centre))
    cells = torch.nonzero(held[0] != 0)
    position = np.tile(grid.centre, (cells.shape[0], 1))
    for at, axis in enumerate(grid.axes):
        index = cells[:, held.dim() - 2 - at].cpu().numpy()
        position[:, axis] += (index - grid.size[at] // 2) * grid.delta[at]
    values = held[0][tuple(cells.T)].cpu().numpy()
    summed = np.exp(-2j * np.pi * k @ position.T) @ values * spectrum

    assert grid.lattice == 2e-3
    assert np.linalg.norm(read - summed) < 1e-2 * np.linalg.norm(summed)


def test_a_scan_read_span_by_span_reads_what_it_reads_whole(
    tmp_path, device, monkeypatch
):
    monkeypatch.setattr(_fourier, "_SPAN_SAMPLES", 64)
    monkeypatch.setattr(_fourier, "_AHEAD_SAMPLES", 512)
    sequence = _steady("fid", tmp_path / "fid.seq", selective=False)
    tissue = virtual.Phantom(
        [virtual.Ellipse((0.0, 0.0, 0.0), (0.03, 0.03), t1=0.8, t2=0.08)], coils=2
    ).tissue(2e-3, field_t=3.0)
    player = virtual.FourierPlayer(sequence, tissue, device=device)
    spans, first = [], 0
    while first < player.blocks:
        last = player.boundary(first + 1)
        spans.extend(player.readouts(first, last))
        first = last
    whole = list(
        virtual.FourierPlayer(sequence, tissue, device=device).readouts(
            0, player.blocks
        )
    )

    assert len(spans) == len(whole) > 2
    scale = max(np.abs(readout).max() for readout in whole)
    for span, readout in zip(spans, whole, strict=True):
        np.testing.assert_allclose(span, readout, rtol=0, atol=1e-5 * scale)


def test_a_rare_group_of_flip_angles_joins_the_nearest_one_the_same_pulses_turn(
    device,
):
    groups = torch.tensor(
        [[5, 20], [5, 21], [5, 23], [0, 20]], dtype=torch.int8, device=device
    )
    group_of = torch.tensor([0] * 9000 + [1] * 995 + [2] * 4 + [3], device=device)
    density = torch.ones(group_of.numel(), device=device)

    kept, joined = _fourier._merged(groups, group_of, density)

    assert kept.cpu().tolist() == [[5, 20], [5, 21], [0, 20]]
    assert torch.equal(
        kept[joined], groups[torch.tensor([0] * 9000 + [1] * 999 + [3], device=device)]
    )
