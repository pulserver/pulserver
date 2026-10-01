"""Time an ADC window read by the engine and by a device: under a spiral on the isochromats' lattice, or off it, or under a held gradient.

The isochromats lie on a square lattice 1 mm apart, several to a point at
positions of their own through the slab, at off-resonances and T2s spread as
over a head at 3 T, and are received by coils of smooth sensitivities.
``--angle`` turns them off the lattice about z, as a moving subject's are, and
the device then sums the window sample by sample; ``--held`` reads a
Cartesian line under a held gradient instead, which the device sums sample by
sample too. Each window starts from the same magnetisation; the times are the
best of the repeats, after one window that plans the transforms and, on a CUDA
device, times the kernels' configurations; ``--profile`` waits for the device
between stages to time each::

    python scripts/benchmark_device.py --device cuda --coils 48 --profile
    python scripts/benchmark_device.py --device cuda --coils 48 --angle 10 --profile
    python scripts/benchmark_device.py --device cuda --coils 48 --held --samples 256
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pulserver.virtual import Isochromats


def _isochromats(points: int, copies: int, coils: int, angle: float, rng):
    index = np.indices((points, points)).reshape(2, -1).T - points // 2
    xy = 1e-3 * np.repeat(index, copies, axis=0).astype(float)
    c, s = np.cos(angle), np.sin(angle)
    xy = xy @ np.array([[c, -s], [s, c]]).T
    positions = np.column_stack([xy, rng.uniform(-2e-3, 2e-3, len(xy))])
    angles = 2.0 * np.pi * np.arange(coils) / coils
    centres = 0.12 * np.column_stack([np.cos(angles), np.sin(angles)])
    distance = np.linalg.norm(xy[:, None, :] - centres[None], axis=2)
    receive = np.exp(-0.5 * (distance / 0.08) ** 2 + 1j * (angles + 25.0 * distance))
    properties = {
        "t1": rng.choice([0.8, 1.3, 4.0], len(xy)),
        "t2": rng.choice([0.04, 0.06, 0.08, 0.1, 2.0], len(xy)),
        "off_resonance": rng.uniform(-550.0, 380.0, len(xy)),
        "receive": receive,
    }
    return positions, properties


def _spiral(duration: float, turns: float, reach: float):
    """Corners of the gradients, in Hz/m, of a spiral out to ``reach`` 1/m."""
    t = np.linspace(0.0, duration, 521)
    omega = 2.0 * np.pi * turns / duration
    g = reach / duration * np.exp(1j * omega * t) * (1.0 + 1j * omega * t)
    return [np.array([t, g.real]), np.array([t, g.imag]), None]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", help="torch device of the reads")
    parser.add_argument("--coils", type=int, default=48)
    parser.add_argument(
        "--points", type=int, default=220, help="lattice points along x and y"
    )
    parser.add_argument(
        "--copies", type=int, default=9, help="isochromats per lattice point"
    )
    parser.add_argument("--samples", type=int, default=2600)
    parser.add_argument(
        "--angle",
        type=float,
        default=0.0,
        help="degrees the isochromats are turned off their lattice about z",
    )
    parser.add_argument(
        "--held",
        action="store_true",
        help="read a line under a held gradient instead of the spiral",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--threads", type=int, default=0, help="engine threads; 0 for every core"
    )
    parser.add_argument(
        "--profile", action="store_true", help="time each stage of the device's windows"
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(0)
    positions, properties = _isochromats(
        args.points, args.copies, args.coils, np.radians(args.angle), rng
    )
    n = len(positions)
    if args.held:
        dwell = 4e-6
        duration = args.samples * dwell + 0.04e-3
        gradients = [
            np.array([[0.0, duration], [1.0 / (0.22 * dwell)] * 2]),
            None,
            None,
        ]
        adc = 0.02e-3 + dwell * np.arange(args.samples)
    else:
        duration = 5.2e-3
        gradients = _spiral(duration, 7.0, 0.5 / 1e-3)
        adc = np.linspace(0.02e-3, duration - 0.02e-3, args.samples)
    start = rng.normal(size=n) + 1j * rng.normal(size=n)
    magnetization = np.column_stack([start.real, start.imag, np.full(n, 0.2)])

    from pulserver.virtual._device import Device

    reader = Device(args.device, profile=args.profile)
    signals = {}
    for name, device in (("engine", None), (args.device, reader)):
        spins = Isochromats(
            positions, **properties, threads=args.threads, device=device
        )
        times = []
        for _ in range(args.repeats + 1):
            spins.magnetization = magnetization
            began = time.perf_counter()
            signals[name] = spins.play(
                duration, gradients=gradients, adc=adc, tolerance=args.tolerance
            )
            times.append(time.perf_counter() - began)
        if spins.lattice_windows:
            read = "on the lattice"
        elif device is None:
            read = "by its transform or sample by sample"
        else:
            read = "sample by sample"
        print(
            f"{name:>8}: {1e3 * min(times[1:]):8.1f} ms per window ({read}), "
            f"{1e3 * times[0]:.0f} ms the first"
        )
    if args.profile:
        windows = reader.stages.pop("windows")
        stages = ", ".join(
            f"{stage} {1e3 * seconds / windows:.2f} ms"
            for stage, seconds in reader.stages.items()
        )
        print(f"  {args.device} per window, waiting between stages: {stages}")
    terms = (np.abs(start) @ np.abs(properties["receive"])).max()
    difference = np.abs(signals["engine"] - signals[args.device]).max() / terms
    print(
        f"{n} isochromats, {args.coils} coils, {args.samples} samples; the reads differ "
        f"by {difference:.1e} of the terms at a tolerance of {args.tolerance:g}"
    )


if __name__ == "__main__":
    main()
