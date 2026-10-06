"""ADC windows read and runs of repetitions carried by a device in the engine's place: windows on the isochromats' lattice and sample by sample, runs a tile of repetitions at a time."""

import gc
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pulserver import virtual
from pulserver.virtual import Isochromats, RigidMotion

RNG = np.random.default_rng(20261001)


def _area(times, values, at):
    """The integral of the piecewise-linear waveform through the corners, up to ``at``."""
    fine = np.union1d(times, [at])
    fine = fine[fine <= at]
    return np.sum(
        0.5
        * np.diff(fine)
        * (np.interp(fine[1:], times, values) + np.interp(fine[:-1], times, values))
    )


def _gradients(trajectory: str, duration: float) -> list[np.ndarray | None]:
    """Corners of gradients whose k moves along one, two or three axes throughout ``duration``, held along one for ``"held"``."""
    if trajectory == "held":
        return [np.array([[0.0, duration], [2e4, 2e4]]), None, None]
    if trajectory == "ramps":
        ramps = np.array([[0.0, 0.4, 1.6, 2.0], [0.0, 2e4, 2e4, 0.0]])
        return [ramps * [[duration / 2.0], [1.0]], None, None]
    times = np.linspace(0.0, duration, 81)
    spiral = 3e4 * (times / duration) * np.exp(6j * np.pi * times / duration)
    ramp = np.array([times, 1e4 * times / duration])
    return [
        np.array([times, spiral.real]),
        np.array([times, spiral.imag]),
        ramp if trajectory == "cone" else None,
    ]


def _window(trajectory: str, coils: int | None):
    """Isochromats several to a lattice point 4 mm apart along the axes the k of ``_gradients`` moves along, anywhere along the others, and the window's samples."""
    shape = {
        "held": (24, 1, 1),
        "ramps": (24, 1, 1),
        "spiral": (12, 10, 1),
        "cone": (8, 6, 4),
    }[trajectory]
    index = np.indices(shape).reshape(3, -1).T - np.array(shape) // 2
    positions = np.repeat(4e-3 * index.astype(float), 3, axis=0)
    free = np.array(shape) == 1
    positions[:, free] = RNG.uniform(-0.02, 0.02, size=(len(positions), free.sum()))
    n = len(positions)
    receive = (
        None
        if coils is None
        else RNG.normal(size=(n, coils)) + 1j * RNG.normal(size=(n, coils))
    )
    properties = {
        "t1": 0.9,
        "t2": RNG.choice([0.03, 0.08, 0.3], n),
        "off_resonance": RNG.uniform(-100.0, 100.0, n),
        "receive": receive,
    }
    start = RNG.normal(size=n) + 1j * RNG.normal(size=n)
    return positions, properties, start


def _transverse(positions, properties, start, gradients, times):
    """Each isochromat's transverse magnetisation at ``times``, (times, isochromats)."""
    times = np.atleast_1d(times)
    areas = np.array(
        [[0.0 if g is None else _area(*g, t) for g in gradients] for t in times]
    )
    phase = areas @ positions.T + np.outer(times, properties["off_resonance"])
    return start * np.exp(-times[:, None] / properties["t2"] - 2j * np.pi * phase)


def _signal(positions, properties, start, gradients, times):
    """What each coil receives at ``times``, summed over the isochromats exactly, (coils, times), and the largest sum of the magnitudes of a sample's terms."""
    transverse = _transverse(positions, properties, start, gradients, times)
    receive = properties["receive"]
    sensitivities = np.ones((len(start), 1)) if receive is None else receive
    return (transverse @ sensitivities).T, (np.abs(start) @ np.abs(sensitivities)).max()


def _scattered(trajectory: str, coils: int | None):
    """The isochromats of :func:`_window` turned off their lattice about z and x."""
    positions, properties, start = _window(trajectory, coils)
    return positions @ _rotation(0.3, 0.2).T, properties, start


def _rotation(about_z: float, about_x: float) -> np.ndarray:
    c, s = np.cos(about_z), np.sin(about_z)
    z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    c, s = np.cos(about_x), np.sin(about_x)
    return z @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _play(spins, start, trajectory, tolerance):
    duration = 2e-3
    spins.magnetization = np.column_stack(
        [start.real, start.imag, np.full(len(start), 0.2)]
    )
    gradients = _gradients(trajectory, duration)
    adc = np.linspace(0.05e-3, duration - 0.05e-3, 120)
    return (
        spins.play(duration, gradients=gradients, adc=adc, tolerance=tolerance),
        gradients,
        adc,
    )


def _read_directly(window) -> bool:
    """Read a window as the engine describes it, its lattice sums and their transform summed directly."""
    order, starts = window["order"], window["starts"]
    n = len(order)
    if window["receive_re"] is None:
        receive = np.ones((1, n))
    else:
        receive = (window["receive_re"] + 1j * window["receive_im"])[:, order]
    magnetization = (window["mx"] + 1j * window["my"])[order]
    turn = np.exp(
        2j
        * np.pi
        * np.outer(window["frequency"] - window["off_resonance"], window["nodes"])
    )
    values = magnetization[:, None] * window["decay"][window["decay_of"]] * turn
    point = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    sums = np.zeros((len(receive), len(window["nodes"]), len(starts) - 1), complex)
    for c in range(len(receive)):
        np.add.at(sums[c].T, point, receive[c][:, None] * values)
    rest, phase = np.arange(len(starts) - 1), 0.0
    for modes, x in zip(window["modes"], window["x"], strict=True):
        phase = phase + np.outer(x, rest % modes - modes // 2)
        rest = rest // modes
    transformed = sums @ np.exp(-1j * phase).T
    window["out"][:] = np.einsum("cls,sl->cs", transformed, window["basis"])
    return True


def _sum_directly(window) -> bool:
    """Read a window as the engine describes it to a device reading it sample by sample: every isochromat's term at every sample, summed."""
    positions = np.column_stack([window["x"], window["y"], window["z"]])
    cycles = window["k"] @ positions.T + np.outer(
        window["time"], window["off_resonance"]
    )
    rate = window["rates"][window["decay_of"]]
    terms = (window["mx"] + 1j * window["my"]) * np.exp(
        -np.outer(window["time"], rate) - 2j * np.pi * cycles
    )
    if window["receive_re"] is None:
        receive = np.ones((1, len(positions)))
    else:
        receive = window["receive_re"] + 1j * window["receive_im"]
    window["out"][:] = receive @ terms.T
    return True


@pytest.mark.parametrize(
    ("trajectory", "tolerance", "coils"),
    [("ramps", 1e-4, 3), ("spiral", 0.0, 3), ("spiral", 1e-4, None), ("cone", 1e-4, 3)],
)
def test_a_device_reading_the_window_the_engine_describes_reads_the_window(
    trajectory, tolerance, coils
):
    """The lattice, the isochromats' order on it, their values at the Chebyshev points, the basis and the samples' coordinates the engine hands a device describe the window to within its tolerance."""
    positions, properties, start = _window(trajectory, coils)
    spins = Isochromats(positions, **properties)
    spins._native.use_device(SimpleNamespace(lattice=_read_directly))
    signal, gradients, adc = _play(spins, start, trajectory, tolerance)

    assert (spins.lattice_windows, spins.device_windows) == (1, 1)
    expected, terms = _signal(positions, properties, start, gradients, adc)
    np.testing.assert_allclose(
        signal, expected, rtol=0, atol=max(tolerance, 1e-11) * terms
    )


def test_a_window_a_device_declines_is_read_by_the_engine():
    """Declining leaves the window to the engine, which reads it as it would without a device."""
    positions, properties, start = _window("spiral", 3)
    offered = []
    spins = Isochromats(positions, **properties)
    spins._native.use_device(
        SimpleNamespace(lattice=lambda window: offered.append(window["axes"]) or False)
    )
    declined, _, _ = _play(spins, start, "spiral", 1e-4)
    alone = Isochromats(positions, **properties)
    signal, _, _ = _play(alone, start, "spiral", 1e-4)

    assert offered == [0b011]
    assert spins.device_windows == 0
    np.testing.assert_array_equal(declined, signal)
    np.testing.assert_array_equal(spins.magnetization, alone.magnetization)


def test_an_error_in_a_device_reaches_the_caller():
    positions, properties, start = _window("spiral", 3)
    spins = Isochromats(positions, **properties)

    def fail(window):
        raise RuntimeError("the device failed")

    spins._native.use_device(SimpleNamespace(lattice=fail, samples=fail))
    with pytest.raises(RuntimeError, match="the device failed"):
        _play(spins, start, "spiral", 1e-4)
    scattered = Isochromats(_scattered("spiral", 3)[0], **properties)
    scattered._native.use_device(SimpleNamespace(lattice=fail, samples=fail))
    with pytest.raises(RuntimeError, match="the device failed"):
        _play(scattered, start, "spiral", 1e-4)


@pytest.mark.parametrize(
    ("trajectory", "coils"),
    [("spiral", 3), ("held", None), ("ramps", 1), ("cone", 3)],
)
def test_a_device_summing_the_window_the_engine_describes_sample_by_sample_reads_the_window(
    trajectory, coils
):
    """Off any lattice, the positions, properties, magnetisation at the first sample, and each sample's k and time from it that the engine hands a device describe the window exactly; the engine leaves the isochromats as the window does."""
    positions, properties, start = _scattered(trajectory, coils)
    spins = Isochromats(positions, **properties)
    spins._native.use_device(SimpleNamespace(samples=_sum_directly))
    signal, gradients, adc = _play(spins, start, trajectory, 0.0)

    assert (spins.lattice_windows, spins.device_windows) == (0, 1)
    expected, terms = _signal(positions, properties, start, gradients, adc)
    np.testing.assert_allclose(signal, expected, rtol=0, atol=1e-11 * terms)
    m = spins.magnetization
    np.testing.assert_allclose(
        m[:, 0] + 1j * m[:, 1],
        _transverse(positions, properties, start, gradients, 2e-3)[0],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("tolerance", "single"), [(1e-4, True), (1e-5, True), (1e-7, False), (0.0, False)]
)
def test_a_window_is_summed_sample_by_sample_in_single_precision_at_tolerances_it_reaches(
    tolerance, single
):
    positions, properties, start = _scattered("spiral", 3)
    handed = []
    spins = Isochromats(positions, **properties)
    spins._native.use_device(
        SimpleNamespace(
            samples=lambda w: handed.append((w["single"], w["tolerance"])) or False
        )
    )
    _play(spins, start, "spiral", tolerance)

    assert handed == [(single, tolerance or 1e-11)]


def test_a_long_window_of_a_wide_spread_of_off_resonances_is_summed_in_double_precision():
    """Each term's turn by its off-resonance is rounded in proportion to it."""
    positions, properties, start = _scattered("spiral", 3)
    properties["off_resonance"] = np.linspace(-3e4, 3e4, len(positions))
    handed = []
    spins = Isochromats(positions, **properties)
    spins._native.use_device(
        SimpleNamespace(samples=lambda w: handed.append(w["single"]) or False)
    )
    _play(spins, start, "spiral", 1e-4)

    assert handed == [False]


def _window_device(device, **options):
    """The Triton device, reading every window however small, or a skip where it cannot run here."""
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    if device == "cuda":
        pytest.importorskip("cufinufft")
    from pulserver.virtual._device import Device, _interpreted

    if device == "cpu" and not _interpreted():
        pytest.skip("Triton runs on the CPU only under its interpreter")
    options.setdefault("smallest", 0)
    options.setdefault("smallest_run", 0)
    return Device(device, **options)


@pytest.mark.parametrize(
    ("trajectory", "tolerance", "coils"),
    [
        ("spiral", 1e-4, 3),
        ("spiral", 1e-7, 3),
        ("spiral", 0.0, 3),
        ("spiral", 1e-4, None),
        ("ramps", 1e-4, 3),
        ("cone", 1e-4, 3),
    ],
)
def test_a_window_read_on_a_device_is_read_to_within_its_tolerance(
    device, trajectory, tolerance, coils
):
    """Summed by the Triton kernel and transformed by cuFINUFFT, or FINUFFT on the CPU, in the precision the engine would read it in."""
    positions, properties, start = _window(trajectory, coils)
    spins = Isochromats(positions, **properties, device=_window_device(device))
    signal, gradients, adc = _play(spins, start, trajectory, tolerance)

    assert (spins.lattice_windows, spins.device_windows) == (1, 1)
    expected, terms = _signal(positions, properties, start, gradients, adc)
    np.testing.assert_allclose(
        signal, expected, rtol=0, atol=max(tolerance, 1e-11) * terms
    )
    # The engine leaves each isochromat as the window does, whoever read it.
    m = spins.magnetization
    np.testing.assert_allclose(
        m[:, 0] + 1j * m[:, 1],
        _transverse(positions, properties, start, gradients, 2e-3)[0],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(("coils", "memory"), [(20, None), (3, 1)])
def test_a_device_sums_any_number_of_coils_as_many_at_once_as_its_memory_holds(
    device, coils, memory
):
    """More coils than the kernel sums together, and coils summed one at a time where the memory holds one coil's sums."""
    positions, properties, start = _window("spiral", coils)
    options = {} if memory is None else {"memory": memory}
    spins = Isochromats(
        positions, **properties, device=_window_device(device, **options)
    )
    signal, gradients, adc = _play(spins, start, "spiral", 1e-4)

    assert spins.device_windows == 1
    expected, terms = _signal(positions, properties, start, gradients, adc)
    np.testing.assert_allclose(signal, expected, rtol=0, atol=1e-4 * terms)


def test_a_phantom_s_isochromats_read_their_windows_on_the_device_they_are_given(
    device,
):
    """The phantom's grid is the lattice its windows are summed onto."""
    phantom = virtual.Phantom(
        [virtual.Ellipse((0.0, 0.0, 0.0), (0.03, 0.02), 0.3, 0.08)], coils=3
    )
    spins = phantom.isochromats(2e-3, device=_window_device(device))
    engine = phantom.isochromats(2e-3)
    start = np.exp(1j * np.linspace(0.0, 1.0, len(spins)))
    signal, _, _ = _play(spins, start, "spiral", 1e-4)
    alone, _, _ = _play(engine, start, "spiral", 1e-4)

    assert spins.device_windows == 1
    # Each term is at most a sensitivity, 1 + depth = 1.5 for the phantom's coils.
    np.testing.assert_allclose(signal, alone, rtol=0, atol=2e-4 * 1.5 * len(spins))


def test_a_profiled_device_times_each_stage_of_every_window(device):
    positions, properties, start = _window("spiral", 3)
    profiled = _window_device(device, profile=True)
    spins = Isochromats(positions, **properties, device=profiled)
    _play(spins, start, "spiral", 1e-4)

    assert profiled.stages["windows"] == 1
    assert set(profiled.stages) == {
        "upload",
        "sums",
        "transform",
        "basis",
        "download",
        "windows",
    }
    assert all(seconds >= 0.0 for seconds in profiled.stages.values())


def test_a_device_reads_isochromats_moved_on_the_lattice_they_were_moved_onto(device):
    """The lattice a device holds is that of the positions the window was read at."""
    positions, properties, start = _window("spiral", 3)
    spins = Isochromats(positions, **properties, device=_window_device(device))
    alone = Isochromats(positions, **properties)
    moved = positions * [1.0, -1.0, 1.0] + [1e-3, 0.0, 0.0]
    read = []
    for each in (spins, alone):
        _play(each, start, "spiral", 1e-4)
        each.positions = moved
        read.append(_play(each, start, "spiral", 1e-4)[0])

    assert spins.device_windows == 2
    expected, terms = _signal(
        moved, properties, start, *_play(alone, start, "spiral", 1e-4)[1:]
    )
    np.testing.assert_allclose(read[0], expected, rtol=0, atol=2e-4 * terms)
    np.testing.assert_allclose(read[1], expected, rtol=0, atol=2e-4 * terms)


@pytest.mark.parametrize(
    ("trajectory", "tolerance", "coils"),
    [
        ("spiral", 1e-4, 3),
        ("spiral", 0.0, 3),
        ("spiral", 1e-4, None),
        ("held", 1e-4, 20),
        ("held", 1e-7, 1),
        ("ramps", 1e-4, 1),
        ("cone", 1e-4, 3),
    ],
)
def test_a_window_off_the_lattice_summed_on_a_device_is_read_to_within_its_tolerance(
    device, trajectory, tolerance, coils
):
    """Summed by the Triton kernel sample by sample, in the precision the engine asks for; a window under a held gradient too, which the engine would read by its own transform."""
    positions, properties, start = _scattered(trajectory, coils)
    spins = Isochromats(positions, **properties, device=_window_device(device))
    signal, gradients, adc = _play(spins, start, trajectory, tolerance)

    assert (spins.lattice_windows, spins.device_windows) == (0, 1)
    expected, terms = _signal(positions, properties, start, gradients, adc)
    np.testing.assert_allclose(
        signal, expected, rtol=0, atol=max(tolerance, 1e-11) * terms
    )
    m = spins.magnetization
    np.testing.assert_allclose(
        m[:, 0] + 1j * m[:, 1],
        _transverse(positions, properties, start, gradients, 2e-3)[0],
        rtol=0,
        atol=1e-12,
    )


def test_a_window_smaller_than_a_device_takes_is_read_by_the_engine(device):
    positions, properties, start = _scattered("spiral", 3)
    spins = Isochromats(
        positions, **properties, device=_window_device(device, smallest=10**12)
    )
    alone = Isochromats(positions, **properties)
    signal, _, _ = _play(spins, start, "spiral", 1e-4)
    expected, _, _ = _play(alone, start, "spiral", 1e-4)

    assert spins.device_windows == 0
    np.testing.assert_array_equal(signal, expected)
    np.testing.assert_array_equal(spins.magnetization, alone.magnetization)


def test_a_moving_subject_s_windows_are_summed_where_each_block_placed_it(device):
    """The device sums each window at the positions of its block, which the motion changes from one block to the next."""
    positions, properties, start = _window("spiral", 3)
    motion = RigidMotion(
        lambda t: _rotation(0.2 + 40.0 * t, 0.0), lambda t: [0.0, 2.0 * t, 0.0]
    )
    moving = Isochromats(
        positions, **properties, motion=motion, device=_window_device(device)
    )
    alone = Isochromats(positions, **properties, motion=motion)
    signals = [[], []]
    for spins, read in zip((moving, alone), signals, strict=True):
        for _ in range(3):
            read.append(_play(spins, start, "spiral", 1e-4)[0])

    assert (moving.lattice_windows, moving.device_windows) == (0, 3)
    terms = (np.abs(start) @ np.abs(properties["receive"])).max()
    np.testing.assert_allclose(signals[0], signals[1], rtol=0, atol=1e-4 * terms)


def test_engines_sharing_a_device_each_read_their_own_isochromats(device):
    shared = _window_device(device)
    first = _scattered("spiral", 3)
    second = _scattered("held", 2)
    engines = [Isochromats(p, **q, device=shared) for p, q, _ in (first, second)]
    alone = [Isochromats(p, **q) for p, q, _ in (first, second)]
    for which in (0, 1, 0, 1):
        trajectory, start = ("spiral", first[2]) if which == 0 else ("held", second[2])
        signal, _, _ = _play(engines[which], start, trajectory, 1e-4)
        expected, _, _ = _play(alone[which], start, trajectory, 0.0)
        terms = (np.abs(start) @ np.abs((first, second)[which][1]["receive"])).max()
        np.testing.assert_allclose(signal, expected, rtol=0, atol=1e-4 * terms)

    assert [spins.device_windows for spins in engines] == [2, 2]


def test_a_profiled_device_times_each_stage_of_a_window_summed_sample_by_sample(
    device,
):
    positions, properties, start = _scattered("spiral", 3)
    profiled = _window_device(device, profile=True)
    spins = Isochromats(positions, **properties, device=profiled)
    _play(spins, start, "spiral", 1e-4)

    assert profiled.stages["windows"] == 1
    assert set(profiled.stages) == {"upload", "samples", "download", "windows"}
    assert all(seconds >= 0.0 for seconds in profiled.stages.values())


# Runs of repetitions carried on the device.

#: Repetitions of a run, played in two parts that end inside a tile.
RUN = 20


def _lattice_spins(coils, scattered=False):
    """Isochromats on a 6 x 4 x 3 lattice 4 mm apart, or scattered about it, and their properties."""
    index = np.indices((6, 4, 3)).reshape(3, -1).T - np.array([3, 2, 1])
    positions = 4e-3 * index.astype(float)
    if scattered:
        positions = positions + RNG.uniform(-1e-3, 1e-3, positions.shape)
    n = len(positions)
    receive = (
        None
        if coils is None
        else RNG.normal(size=(n, coils)) + 1j * RNG.normal(size=(n, coils))
    )
    properties = {
        "proton_density": RNG.uniform(0.5, 1.0, n),
        "t1": RNG.choice([0.8, 1.2, 4.0], n),
        "t2": RNG.choice([0.05, 0.08, 0.3], n),
        "off_resonance": RNG.normal(0.0, 30.0, n),
        "receive": receive,
    }
    return positions, properties


def _repetition(windows=1, samples=48, single_sample=False):
    """One repetition: a pulse, ``windows`` readouts under held gradients, the first prephased, and a pause."""
    import pypulseqpp as pp

    dwell = 10e-6
    span = samples * dwell
    readout = 1.0 / (0.24 * dwell)
    blocks = [
        {"duration": 0.3e-3, "rf": pp.make_block_pulse(np.pi / 6, duration=0.2e-3)}
    ]
    for w in range(windows):
        if single_sample:
            blocks.append(
                {"duration": 0.3e-3, "adc": pp.make_adc(1, dwell=dwell, delay=0.1e-3)}
            )
            continue
        pre = -readout * span / 2 / 0.3e-3 if w == 0 else 0.0
        level = readout if w % 2 == 0 else -readout
        times = [0.0, 0.05e-3, 0.35e-3, 0.4e-3, 0.4e-3 + span, 0.45e-3 + span]
        gx = np.array([times, [0.0, pre, pre, level, level, 0.0]])
        blocks.append(
            {
                "duration": 0.45e-3 + span,
                "adc": pp.make_adc(samples, dwell=dwell, delay=0.4e-3),
                "gradients": [gx, None, None],
            }
        )
    blocks.append({"duration": 0.4e-3})
    return blocks


def _schedule(kind, windows=1):
    """The RF phases, phase-encoding areas and net areas of a run of ``kind``."""
    k = np.arange(RUN)
    phases = np.pi * k if kind == "split" else np.deg2rad(117.0) * k * (k + 1) / 2
    areas = np.zeros((RUN, windows, 3))
    areas[:, :, 1] = ((k % 4 - 2) / 0.024)[:, None]
    areas[:, :, 2] = ((k // 4 % 3 - 1) / 0.012)[:, None]
    nets = None
    if kind == "netted":
        nets = np.zeros((RUN, 3))
        nets[:, 2] = 7.0 * (k % 3)
    return phases, areas, nets


def _carried(spins, kind, tolerance, windows=1, single_sample=False, readouts=None):
    phases, areas, nets = _schedule(kind, windows)
    run = spins.repetitions(
        _repetition(windows, single_sample=single_sample),
        phases,
        areas,
        nets=nets,
        readouts=readouts,
        tolerance=tolerance,
    )
    signal = np.concatenate([run.play(7), run.play()])
    return signal, spins.magnetization.copy(), run


@pytest.mark.parametrize(
    ("kind", "coils", "tolerance", "windows", "single_sample"),
    [
        ("spoiled", 20, 1e-4, 1, False),
        ("spoiled", 3, 0.0, 1, False),
        ("spoiled", None, 1e-4, 1, False),
        ("scattered", 17, 1e-4, 1, False),
        ("netted", 16, 1e-4, 1, False),
        ("spoiled", 16, 1e-4, 2, False),
        ("spoiled", 5, 1e-4, 1, True),
        ("split", 20, 1e-4, 1, False),
    ],
)
def test_a_run_carried_on_a_device_plays_what_the_engine_plays(
    device, kind, coils, tolerance, windows, single_sample
):
    """Coils taken by matrix products (16 and more) or not, exact or in single precision, phase encodings tabulated on the lattice or computed off it, net areas, two windows, a window of one sample, and a run split about its fixed points."""
    positions, properties = _lattice_spins(coils, scattered=kind == "scattered")
    carrier = _window_device(device, profile=True)
    on_device = Isochromats(positions, **properties, device=carrier)
    alone = Isochromats(positions, **properties)
    signal, settled, run = _carried(on_device, kind, tolerance, windows, single_sample)
    expected, left, _ = _carried(alone, kind, tolerance, windows, single_sample)

    assert carrier.stages["tiles"] == 2
    assert run._native.divided == (kind == "split")
    within = 1e-5 if tolerance > 0.0 else 1e-12
    scale = np.abs(expected).max()
    np.testing.assert_allclose(signal, expected, rtol=0, atol=within * scale)
    np.testing.assert_allclose(settled, left, rtol=0, atol=within)


def test_a_run_whose_transients_drop_is_carried_on_by_the_slots_left(device):
    """A split run drops each transient that falls to its limit; the device drops what the engine drops and carries the rest."""
    positions, properties = _lattice_spins(16)
    properties["t1"] = np.full(len(positions), 4e-3)
    properties["t2"] = np.full(len(positions), 3e-3)
    carrier = _window_device(device)
    on_device = Isochromats(positions, **properties, device=carrier)
    alone = Isochromats(positions, **properties)
    signal, settled, run = _carried(on_device, "split", 1e-2)
    expected, left, engine = _carried(alone, "split", 1e-2)

    (held,) = carrier._runs.values()
    assert run.carried == engine.carried == held.slots == 0
    scale = np.abs(expected).max()
    np.testing.assert_allclose(signal, expected, rtol=0, atol=1e-2 * scale)
    np.testing.assert_allclose(settled, left, rtol=0, atol=1e-2)


@pytest.mark.parametrize(
    ("kind", "coils", "tolerance"),
    [("spoiled", 20, 1e-4), ("spoiled", 3, 0.0), ("netted", 16, 1e-4)],
)
def test_a_run_whose_windows_turn_is_carried_on_a_device_as_the_engine_carries_it(
    device, kind, coils, tolerance
):
    """Each repetition reads the window along its own direction, its slots spread anew; coils taken by matrix products or not, exact or in single precision, with net areas."""
    positions, properties = _lattice_spins(coils)
    carrier = _window_device(device, profile=True)
    on_device = Isochromats(positions, **properties, device=carrier)
    alone = Isochromats(positions, **properties)
    k = np.arange(RUN)
    readouts = np.zeros((RUN, 1, 3))
    readouts[:, 0, 1] = 1e3 * np.sin(0.1 * k)
    readouts[:, 0, 2] = 5e2 * np.cos(0.3 * k)
    signal, settled, _ = _carried(on_device, kind, tolerance, readouts=readouts)
    expected, left, _ = _carried(alone, kind, tolerance, readouts=readouts)

    assert carrier.stages["tiles"] == 2
    within = 1e-5 if tolerance > 0.0 else 1e-12
    scale = np.abs(expected).max()
    np.testing.assert_allclose(signal, expected, rtol=0, atol=within * scale)
    np.testing.assert_allclose(settled, left, rtol=0, atol=within)


def _spoke(along, after, window=True):
    """A ZTE spoke along ``along``: a hard pulse on the readout gradient, held through its block, then the window under it, where given, and the turn onto ``after``."""
    import pypulseqpp as pp

    readout = 1.0 / (0.24 * 10e-6)
    rf = pp.make_block_pulse(np.deg2rad(10.0), duration=10e-6, delay=10e-6)
    hold = [np.array([[0.0, 60e-6], [readout * a] * 2]) for a in along]
    read = [
        np.array([[0.0, 0.6e-3, 0.8e-3], [readout * a] * 2 + [readout * b]])
        for a, b in zip(along, after, strict=True)
    ]
    second = {"duration": 0.8e-3, "gradients": read}
    if window:
        second["adc"] = pp.make_adc(48, dwell=10e-6, delay=20e-6)
    return [{"duration": 60e-6, "rf": rf, "gradients": hold}, second]


def _saturate(spins):
    """Play a pulse of 60 degrees, then a spoiler along z, on ``spins``."""
    import pypulseqpp as pp

    spoiler = np.array([[0.0, 0.1e-3, 0.9e-3, 1.0e-3], [0.0, 4e3, 4e3, 0.0]])
    spins.play(0.3e-3, rf=pp.make_block_pulse(np.pi / 3, duration=0.2e-3))
    spins.play(1.0e-3, gradients=[None, None, spoiler])


def _spokes(spins, tolerance, window=True, between=False):
    """A run of ZTE spokes on a spiral over a hemisphere, played in two parts; with ``between``, a saturation played between them and the run resumed after it."""
    readout = 1.0 / (0.24 * 10e-6)
    n = np.arange(RUN + 1)
    z = 1.0 - (n + 0.5) / (RUN + 1)
    angle = n * np.pi * (3.0 - np.sqrt(5.0))
    along = np.column_stack(
        [np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z]
    )
    change = readout * (along[:-1] - along[0])
    turned = readout * (along[1:] - along[1])
    k = np.arange(RUN)
    run = spins.repetitions(
        _spoke(along[0], along[1], window),
        np.deg2rad(117.0) * k * (k + 1) / 2,
        25e-6 * change[:, None] if window else np.zeros((RUN, 0, 3)),
        readouts=change[:, None] if window else None,
        nets=0.6e-3 * change + 0.1e-3 * (change + turned),
        pulse_gradients=change,
        tolerance=tolerance,
        split=False,
    )
    first = run.play(7)
    if between:
        _saturate(spins)
        run.resume()
    signal = np.concatenate([first, run.play()])
    return signal, spins.magnetization.copy()


@pytest.mark.parametrize(
    ("coils", "transmit", "tolerance", "window"),
    [
        (3, False, 1e-4, True),
        (20, True, 1e-4, True),
        (3, True, 0.0, True),
        (None, False, 1e-4, True),
        (3, True, 1e-4, False),
    ],
)
def test_a_run_whose_pulses_play_under_their_own_gradients_is_carried_on_a_device_as_the_engine_carries_it(
    device, coils, transmit, tolerance, window
):
    """ZTE spokes: each pulse read off its tables at each slot's field under its spoke's gradient, with a transmit map or without, exact or in single precision; a run of them without windows is taken too."""
    positions, properties = _lattice_spins(coils, scattered=True)
    if transmit:
        properties["transmit"] = RNG.uniform(0.7, 1.2, len(positions))
    carrier = _window_device(device, profile=True)
    on_device = Isochromats(positions, **properties, device=carrier)
    alone = Isochromats(positions, **properties)
    signal, settled = _spokes(on_device, tolerance, window)
    expected, left = _spokes(alone, tolerance, window)

    assert carrier.stages["tiles"] == 2
    within = 1e-5 if tolerance > 0.0 else 1e-12
    if window:
        scale = np.abs(expected).max()
        np.testing.assert_allclose(signal, expected, rtol=0, atol=within * scale)
    np.testing.assert_allclose(settled, left, rtol=0, atol=within)


def _resumed(spins, tolerance):
    """A run played in two parts, a saturation played between them and the run resumed after it."""
    phases, areas, nets = _schedule("spoiled")
    run = spins.repetitions(
        _repetition(), phases, areas, nets=nets, tolerance=tolerance, split=False
    )
    first = run.play(7)
    _saturate(spins)
    run.resume()
    signal = np.concatenate([first, run.play()])
    return signal, spins.magnetization.copy()


@pytest.mark.parametrize(
    ("kind", "coils", "tolerance"),
    [("spoiled", 20, 1e-4), ("spoiled", 3, 0.0), ("spokes", 3, 1e-4)],
)
def test_a_run_resumed_after_blocks_played_between_its_repetitions_is_carried_on_a_device_as_the_engine_carries_it(
    device, kind, coils, tolerance
):
    """The device takes the magnetisation the blocks leave as its slots' before carrying the repetitions after them."""
    positions, properties = _lattice_spins(coils, scattered=kind == "spokes")
    carrier = _window_device(device, profile=True)
    on_device = Isochromats(positions, **properties, device=carrier)
    alone = Isochromats(positions, **properties)
    if kind == "spokes":
        signal, settled = _spokes(on_device, tolerance, between=True)
        expected, left = _spokes(alone, tolerance, between=True)
    else:
        signal, settled = _resumed(on_device, tolerance)
        expected, left = _resumed(alone, tolerance)

    assert carrier.stages["tiles"] == 2
    within = 1e-5 if tolerance > 0.0 else 1e-12
    scale = np.abs(expected).max()
    np.testing.assert_allclose(signal, expected, rtol=0, atol=within * scale)
    np.testing.assert_allclose(settled, left, rtol=0, atol=within)


@pytest.mark.parametrize("kind", ["spoiled", "dropping", "turned", "spokes", "resumed"])
def test_a_run_larger_than_its_memory_is_carried_in_parts_as_the_engine_carries_it(
    device, kind, monkeypatch
):
    """Cut into parts most of which wait off the device between tiles: their spreading adds up to each window, transients drop part by part, pulses are read off tables in each, and a resumed run takes the state of every part."""
    from pulserver.virtual import _carry

    made = []

    class Recorded(_carry.Run):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(_carry, "Run", Recorded)
    # Parts of half the memory: a few, one staying, so that the interpreter
    # carries them in reasonable time.
    monkeypatch.setattr(_carry, "PARTS", 2)
    coils = 3 if kind in ("turned", "spokes") else 20
    positions, properties = _lattice_spins(coils, scattered=kind == "spokes")
    if kind == "dropping":
        properties["t1"] = np.full(len(positions), 4e-3)
        properties["t2"] = np.full(len(positions), 3e-3)
    carrier = _window_device(device, run_memory=200 * len(positions))
    on_device = Isochromats(positions, **properties, device=carrier)
    alone = Isochromats(positions, **properties)
    tolerance = 1e-2 if kind == "dropping" else 1e-4
    if kind == "spokes":
        signal, settled = _spokes(on_device, tolerance)
        expected, left = _spokes(alone, tolerance)
    elif kind == "resumed":
        signal, settled = _resumed(on_device, tolerance)
        expected, left = _resumed(alone, tolerance)
    else:
        readouts = None
        if kind == "turned":
            k = np.arange(RUN)
            readouts = np.zeros((RUN, 1, 3))
            readouts[:, 0, 1] = 1e3 * np.sin(0.1 * k)
            readouts[:, 0, 2] = 5e2 * np.cos(0.3 * k)
        schedule = "split" if kind == "dropping" else "spoiled"
        signal, settled, _ = _carried(on_device, schedule, tolerance, readouts=readouts)
        expected, left, _ = _carried(alone, schedule, tolerance, readouts=readouts)

    (run,) = made
    assert len(run.parts) > 1
    assert any(part.stays for part in run.parts)
    assert not all(part.stays for part in run.parts)
    within = 1e-2 if kind == "dropping" else 1e-5
    scale = np.abs(expected).max()
    np.testing.assert_allclose(signal, expected, rtol=0, atol=within * scale)
    np.testing.assert_allclose(settled, left, rtol=0, atol=within)


def test_a_run_without_windows_is_carried_by_the_engine(device):
    positions, properties = _lattice_spins(3)
    carrier = _window_device(device, profile=True)
    spins = Isochromats(positions, **properties, device=carrier)
    blocks = [block for block in _repetition() if "adc" not in block]
    spins.repetitions(blocks, np.pi * np.arange(RUN)).play()

    assert "tiles" not in carrier.stages
    assert not carrier._runs


def test_a_device_frees_a_run_when_its_repetitions_go(device):
    positions, properties = _lattice_spins(3)
    carrier = _window_device(device)
    spins = Isochromats(positions, **properties, device=carrier)
    _, _, run = _carried(spins, "spoiled", 1e-4)
    assert len(carrier._runs) == 1

    del run
    gc.collect()

    assert not carrier._runs


def test_an_error_in_carrying_a_run_reaches_the_caller():
    positions, properties = _lattice_spins(3)

    def carry(tile):
        raise RuntimeError("the device failed")

    spins = Isochromats(positions, **properties)
    spins._native.use_device(
        SimpleNamespace(
            begin_run=lambda run: True,
            carry=carry,
            write_state=lambda state: None,
            load_state=lambda state: None,
        )
    )
    with pytest.raises(RuntimeError, match="the device failed"):
        _carried(spins, "spoiled", 1e-4)


def test_an_error_in_freeing_a_run_is_reported_rather_than_raised(monkeypatch):
    positions, properties = _lattice_spins(3)
    reported = []
    monkeypatch.setattr(sys, "unraisablehook", reported.append)

    def end_run(run):
        raise RuntimeError("the device kept the run")

    spins = Isochromats(positions, **properties)
    spins._native.use_device(
        SimpleNamespace(
            begin_run=lambda run: True,
            carry=lambda tile: 0,
            write_state=lambda state: None,
            load_state=lambda state: None,
            end_run=end_run,
        )
    )
    _, _, run = _carried(spins, "spoiled", 1e-4)
    del run
    gc.collect()

    assert [str(report.exc_value) for report in reported] == ["the device kept the run"]


def test_a_device_handed_a_run_reads_what_the_engine_hands_it():
    """The run's slots and each tile, as the binding views them, describe the run: one slot per isochromat, each window's grid, and the repetitions of the tile."""
    positions, properties = _lattice_spins(4)
    handed = {}

    def begin_run(run):
        handed["run"] = {
            "slots": run["slots"],
            "coils": run["coils"],
            "cells": run["cells"].copy(),
            "pack": run["pack"].shape,
            "start": run["start"].shape,
            "factor": run["factor"].shape,
            "decay": run["decay"].shape,
            "lattice": run["lattice"],
            "turned": run["turned"].tolist(),
            "place": run["place"],
            "pulse": run["pulse"],
        }
        return False

    spins = Isochromats(positions, **properties)
    spins._native.use_device(
        SimpleNamespace(
            begin_run=begin_run,
            carry=lambda tile: 0,
            write_state=lambda state: None,
            load_state=lambda state: None,
        )
    )
    _carried(spins, "spoiled", 1e-4)

    run = handed["run"]
    n = len(positions)
    assert (run["slots"], run["coils"]) == (n, 4)
    assert run["cells"].tolist() == [96]
    assert run["pack"][0] * run["pack"][2] >= n
    assert run["start"] == (n, 1)
    assert run["factor"] == (n, 1, 4, 2)
    assert run["decay"] == (n,)
    # Tabulated along y and z, which the phase encodings run along.
    assert run["lattice"] == (0, 4, 3)
    # No window turns and no pulse is read off tables, so no slot needs
    # where it lies.
    assert (run["turned"], run["place"], run["pulse"]) == ([0], None, None)


def test_a_profiled_device_times_each_stage_of_a_run(device):
    positions, properties = _lattice_spins(16)
    profiled = _window_device(device, profile=True)
    spins = Isochromats(positions, **properties, device=profiled)
    _carried(spins, "spoiled", 1e-4)

    assert profiled.stages["tiles"] == 2
    assert set(profiled.stages) == {
        "upload",
        "carry",
        "spread",
        "download",
        "state",
        "tiles",
    }
    assert all(seconds >= 0.0 for seconds in profiled.stages.values())
