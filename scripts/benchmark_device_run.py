"""Time a run of repetitions carried by the engine and by a device: an RF-spoiled 3D gradient echo's, phase-encoded along y and z, with ``--radial`` a stack of stars', or with ``--zte`` a ZTE scan's spokes.

The isochromats lie on a lattice 2 mm apart, at off-resonances and T2s spread
as over a head at 3 T, and are received by coils of smooth sensitivities; the
default lattice holds about as many as a console keeps of a head's 3D slab.
The RF phase grows quadratically, so that no isochromat rests at a fixed point
and every one is carried through every repetition. Each repetition reads a
line under a held gradient: along x, or with ``--radial`` a spoke turned about
z by the golden angle from the last partition's, phase-encoded along z. With
``--zte`` each repetition plays a hard pulse on its spoke's own readout
gradient, held through the pulse's block, as a ZTE shell's spokes do, under a
transmit field that varies smoothly by a fifth across the head; the pulse's
map is read off its tables at each isochromat's field and drive. With
``--shells``, the spokes are those of as many shells, each shell's spiral
turned about z from the last's, and each shell's closing spoke, ramped to
zero, and the ramp onto the next shell's first spoke play alone between them,
after which the run resumes. The times are of the repetitions after the first
tile, whose play hands the run to the device and, on a CUDA device, times the
kernels' configurations, and of the blocks between shells with the resumption
after them; ``--profile`` waits for the device between stages to time each::

    python scripts/benchmark_device_run.py --device cuda --coils 32 --profile
    python scripts/benchmark_device_run.py --device cuda --coils 32 --radial --profile
    python scripts/benchmark_device_run.py --device cuda --coils 2 --zte --profile
    python scripts/benchmark_device_run.py --device cuda --coils 2 --zte --shells 4
    python scripts/benchmark_device_run.py --device cuda --coils 2 --repetitions 2048
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pypulseqpp as pp

from pulserver.virtual import Isochromats

DWELL = 4e-6
#: When a window's first sample is taken, from the start of its block, in s.
DELAY = 0.4e-3


def _isochromats(shape, coils: int, rng):
    index = np.indices(shape).reshape(3, -1).T - np.array(shape) // 2
    positions = 2e-3 * index.astype(float)
    n = len(positions)
    angles = 2.0 * np.pi * np.arange(coils) / coils
    centres = 0.12 * np.column_stack([np.cos(angles), np.sin(angles)])
    distance = np.linalg.norm(positions[:, None, :2] - centres[None], axis=2)
    receive = np.exp(-0.5 * (distance / 0.08) ** 2 + 1j * (angles + 25.0 * distance))
    properties = {
        "t1": rng.choice([0.8, 1.3, 4.0], n),
        "t2": rng.choice([0.04, 0.06, 0.08, 0.1, 2.0], n),
        "off_resonance": rng.uniform(-550.0, 380.0, n),
        "receive": receive,
    }
    return positions, properties


def _readout(samples: int, fov: float) -> np.ndarray:
    """Return the gradient along x a line is read under, its prephaser first: times in s, and Hz/m."""
    span = samples * DWELL
    readout = 1.0 / (fov * DWELL)
    pre = -readout * span / 2 / 0.3e-3
    times = [0.0, 0.05e-3, 0.35e-3, DELAY, DELAY + span, DELAY + 0.05e-3 + span]
    return np.array([times, [0.0, pre, pre, readout, readout, 0.0]])


def _repetition(samples: int, fov: float):
    """Return a repetition's blocks: a pulse, a line read under a held gradient after its prephaser, and spoiling."""
    gx = _readout(samples, fov)
    return [
        {
            "duration": 0.3e-3,
            "rf": pp.make_block_pulse(np.radians(12.0), duration=0.2e-3),
        },
        {
            "duration": gx[0, -1],
            "adc": pp.make_adc(samples, dwell=DWELL, delay=DELAY),
            "gradients": [gx, None, None],
        },
        {"duration": 0.6e-3},
    ]


def _area(gradient: np.ndarray, until: float) -> float:
    """Return the area of a piecewise-linear gradient up to ``until``, in 1/m."""
    times = np.append(gradient[0][gradient[0] < until], until)
    values = np.interp(times, gradient[0], gradient[1])
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(times)))


def _radial(samples: int, fov: float, spoke: np.ndarray):
    """Return the readouts, areas and net areas that turn each repetition's line about z by the golden angle times ``spoke``."""
    gx = _readout(samples, fov)
    angle = np.radians(111.246117975) * spoke
    turn = np.column_stack([np.cos(angle) - 1.0, np.sin(angle), np.zeros_like(angle)])
    readouts = (gx[1, 3] * turn)[:, None, :]
    areas = (_area(gx, DELAY) * turn)[:, None, :]
    nets = _area(gx, gx[0, -1]) * turn
    return readouts, areas, nets


#: A ZTE spoke's pulse block, and the turn onto the next spoke after its
#: window, in s.
HOLD, TURN = 60e-6, 0.1e-3


def _spiral(count: int) -> np.ndarray:
    """Return ``count`` directions and the next one's, on a spiral over the sphere."""
    n = np.arange(count + 1)
    z = 1.0 - 2.0 * (n + 0.5) / (count + 1)
    angle = n * np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack(
        [np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z]
    )


def _spoke(samples: int, fov: float, along: np.ndarray, after: np.ndarray):
    """Return a ZTE spoke's blocks: a hard pulse on the readout gradient along ``along``, held through its block, then the window under it and the turn onto ``after``."""
    readout = 1.0 / (fov * DWELL)
    read = 30e-6 + samples * DWELL
    return [
        {
            "duration": HOLD,
            "rf": pp.make_block_pulse(np.radians(3.0), duration=10e-6, delay=10e-6),
            "gradients": [np.array([[0.0, HOLD], [readout * a] * 2]) for a in along],
        },
        {
            "duration": read + TURN,
            "adc": pp.make_adc(samples, dwell=DWELL, delay=20e-6),
            "gradients": [
                np.array([[0.0, read, read + TURN], [readout * a] * 2 + [readout * b]])
                for a, b in zip(along, after, strict=True)
            ],
        },
    ]


def _turned_z(degrees: float) -> np.ndarray:
    """Return the rotation by ``degrees`` about z."""
    turn = np.radians(degrees)
    return np.array(
        [
            [np.cos(turn), -np.sin(turn), 0.0],
            [np.sin(turn), np.cos(turn), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _zte(samples: int, fov: float, count: int, shells: int):
    """Return the first ZTE spoke's blocks, and the pulse gradients, readouts, areas and nets that turn each of ``count`` spokes over ``shells`` shells onto its own; and the blocks between each shell and the next: its closing spoke, ramped to zero, and the ramp onto the next shell's first spoke."""
    readout = 1.0 / (fov * DWELL)
    read = 30e-6 + samples * DWELL
    spiral = _spiral(count // shells)
    directions = [spiral @ _turned_z(137.5 * shell).T for shell in range(shells + 1)]
    along = np.concatenate([d[:-1] for d in directions[:-1]])
    after = np.concatenate([d[1:] for d in directions[:-1]])
    between = [
        [
            *_spoke(samples, fov, directions[shell][-1], np.zeros(3)),
            {
                "duration": TURN,
                "gradients": [
                    np.array([[0.0, TURN], [0.0, readout * b]])
                    for b in directions[shell + 1][0]
                ],
            },
        ]
        for shell in range(shells - 1)
    ]
    change = readout * (along - along[0])
    turned = readout * (after - after[0])
    areas = ((20e-6 + DWELL / 2) * change)[:, None, :]
    nets = read * change + TURN * (change + turned) / 2
    blocks = _spoke(samples, fov, along[0], after[0])
    return blocks, change, change[:, None, :], areas, nets, between


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", help="torch device of the run")
    parser.add_argument("--coils", type=int, default=32)
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=(128, 128, 112),
        metavar=("X", "Y", "Z"),
        help="lattice points along x, y and z",
    )
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=528)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--threads", type=int, default=0, help="engine threads; 0 for every core"
    )
    parser.add_argument(
        "--profile", action="store_true", help="time each stage of the device's tiles"
    )
    shape = parser.add_mutually_exclusive_group()
    shape.add_argument(
        "--radial",
        action="store_true",
        help="read spokes turned about z by the golden angle, phase-encoded along z",
    )
    shape.add_argument(
        "--zte",
        action="store_true",
        help="play each spoke's pulse on its own readout gradient, as a ZTE scan does",
    )
    parser.add_argument(
        "--shells",
        type=int,
        default=1,
        help="with --zte, the shells the spokes fall into, the blocks between them played alone",
    )
    args = parser.parse_args(argv)
    if args.shells != 1 and not args.zte:
        parser.error("--shells is a ZTE scan's")
    args.repetitions -= args.repetitions % args.shells
    return args


def _schedule(args: argparse.Namespace, positions: np.ndarray, properties: dict):
    """Return the run's blocks, the keyword arguments of its repetitions, and the blocks between its shells; with ``--zte``, give ``properties`` a transmit field."""
    fov = 2e-3 * args.shape[0]
    k = np.arange(args.repetitions)
    areas = np.zeros((args.repetitions, 1, 3))
    given = {"phases": np.radians(117.0) * k * (k + 1) / 2, "areas": areas}
    if args.zte:
        blocks, held, readouts, areas, nets, between = _zte(
            args.samples, fov, args.repetitions, args.shells
        )
        properties["transmit"] = 1.0 + 0.2 * np.cos(
            np.pi * positions[:, 0] / fov
        ) * np.cos(np.pi * positions[:, 2] / fov)
        given.update(areas=areas, readouts=readouts, nets=nets, pulse_gradients=held)
        return blocks, given, between
    if args.radial:
        readouts, areas, nets = _radial(args.samples, fov, k // args.shape[2])
        areas[:, 0, 2] = (k % args.shape[2] - args.shape[2] // 2) / fov
        given.update(areas=areas, readouts=readouts, nets=nets)
    else:
        areas[:, 0, 1] = (k % args.shape[1] - args.shape[1] // 2) / fov
        areas[:, 0, 2] = (k // args.shape[1] % args.shape[2] - args.shape[2] // 2) / fov
    return _repetition(args.samples, fov), given, []


def _played(run, spins, shells: int, between: list) -> tuple[np.ndarray, float, float]:
    """Play the run after its first tile, shell by shell with the blocks between them played alone and the run resumed after them; return what it reads, the time its repetitions took and the time the blocks between shells and the resumptions took, in s."""
    count = len(run)
    played = []
    took = boundaries = 0.0
    for shell in range(shells):
        began = time.perf_counter()
        played.append(run.play((shell + 1) * count // shells - run.played))
        took += time.perf_counter() - began
        if shell < len(between):
            began = time.perf_counter()
            for block in between[shell]:
                signal = spins.play(**block)
                if signal.shape[1]:
                    played.append(signal[None])
            run.resume()
            boundaries += time.perf_counter() - began
    return np.concatenate(played), took, boundaries


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    rng = np.random.default_rng(0)
    positions, properties = _isochromats(tuple(args.shape), args.coils, rng)
    n = len(positions)
    blocks, given, between = _schedule(args, positions, properties)

    from pulserver.virtual._device import Device

    carrier = Device(args.device, profile=args.profile)
    signals = {}
    for name, device in (("engine", None), (args.device, carrier)):
        spins = Isochromats(
            positions, **properties, threads=args.threads, device=device
        )
        run = spins.repetitions(
            blocks, **given, tolerance=args.tolerance, split=not between
        )
        began = time.perf_counter()
        first = run.play(16)
        handed = time.perf_counter() - began
        if args.profile:
            carrier.stages.clear()
        rest, took, boundaries = _played(run, spins, args.shells, between)
        signals[name] = np.concatenate([first, rest])
        per = 1e9 * took / (n * (args.repetitions - 16))
        print(
            f"{name:>8}: {per:6.2f} ns per isochromat and repetition, "
            f"{1e3 * took / (args.repetitions - 16):.2f} ms per repetition; "
            f"{handed:.1f} s for the first tile"
            + (
                f"; {1e3 * boundaries / len(between):.1f} ms between shells"
                if between
                else ""
            )
        )
    tiles = carrier.stages.pop("tiles", 0) if args.profile else None
    if tiles == 0:
        print(f"  the {args.device} device left the run to the engine")
    elif tiles:
        stages = ", ".join(
            f"{stage} {1e3 * seconds / tiles:.2f} ms"
            for stage, seconds in carrier.stages.items()
        )
        print(f"  {args.device} per tile of 16, waiting between stages: {stages}")
    scale = np.abs(signals["engine"]).max()
    difference = np.abs(signals["engine"] - signals[args.device]).max() / scale
    print(
        f"{n} isochromats, {args.coils} coils, {args.samples} samples; the runs differ "
        f"by {difference:.1e} of the largest sample at a tolerance of {args.tolerance:g}"
    )


if __name__ == "__main__":
    main()
