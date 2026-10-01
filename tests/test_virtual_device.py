"""ADC windows read by a device in the engine's place: on the isochromats' lattice, and sample by sample."""

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
    from pulserver.virtual._device import WindowDevice, _interpreted

    if device == "cpu" and not _interpreted():
        pytest.skip("Triton runs on the CPU only under its interpreter")
    options.setdefault("smallest", 0)
    return WindowDevice(device, **options)


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
