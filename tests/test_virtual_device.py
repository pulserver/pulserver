"""Windows under a changing gradient read on the isochromats' lattice by a device in the engine's place."""

import numpy as np
import pytest

from pulserver import virtual
from pulserver.virtual import Isochromats

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
    """Corners of gradients whose k moves along one, two or three axes throughout ``duration``."""
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
    shape = {"ramps": (24, 1, 1), "spiral": (12, 10, 1), "cone": (8, 6, 4)}[trajectory]
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
    spins._native.use_lattice_device(_read_directly)
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
    spins._native.use_lattice_device(
        lambda window: offered.append(window["axes"]) or False
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

    spins._native.use_lattice_device(fail)
    with pytest.raises(RuntimeError, match="the device failed"):
        _play(spins, start, "spiral", 1e-4)


def _lattice_device(device, **options):
    """The Triton device, or a skip where it cannot run here."""
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    if device == "cuda":
        pytest.importorskip("cufinufft")
    from pulserver.virtual._device import LatticeDevice, _interpreted

    if device == "cpu" and not _interpreted():
        pytest.skip("Triton runs on the CPU only under its interpreter")
    return LatticeDevice(device, **options)


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
    spins = Isochromats(positions, **properties, device=_lattice_device(device))
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
        positions, **properties, device=_lattice_device(device, **options)
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
    spins = phantom.isochromats(2e-3, device=_lattice_device(device))
    engine = phantom.isochromats(2e-3)
    start = np.exp(1j * np.linspace(0.0, 1.0, len(spins)))
    signal, _, _ = _play(spins, start, "spiral", 1e-4)
    alone, _, _ = _play(engine, start, "spiral", 1e-4)

    assert spins.device_windows == 1
    # Each term is at most a sensitivity, 1 + depth = 1.5 for the phantom's coils.
    np.testing.assert_allclose(signal, alone, rtol=0, atol=2e-4 * 1.5 * len(spins))


def test_a_profiled_device_times_each_stage_of_every_window(device):
    positions, properties, start = _window("spiral", 3)
    profiled = _lattice_device(device, profile=True)
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
    spins = Isochromats(positions, **properties, device=_lattice_device(device))
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
