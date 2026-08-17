"""Design cost of a protocol-scale MPRAGE, Cartesian against stack-of-spirals.

Documentation-only tooling, like everything else in this directory.

Three cases at the same encoding size -- 512 partitions, 1024 views per
inversion train -- so the only thing that varies is how the in-plane shot is
held:

``cartesian``
    One phase-encode line per view, the line readout scaled per shot.
``spiral_rotated``
    One spiral arm turned per shot by a ``ROTATIONS`` extension, so the
    waveform is stored once however many arms the scan plays.
``spiral_explicit``
    The same arms written out as their own waveforms, which is the path a
    reader that will not compose a rotation has to take.

``TransformFOV`` is reported on its own line rather than folded into design.
Every plugin applies it at the end of ``main``, over a sequence that is by then
millions of blocks, so what it costs is a property of the scan size and not of
the family -- and it is invisible in a single end-to-end number.

Run it as::

    python _docs/_bench/bench_mprage.py                 # every case
    python _docs/_bench/bench_mprage.py --only=cartesian
    python _docs/_bench/bench_mprage.py --scale=0.25    # a quarter-size sweep
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

#: The encoding size every case is measured at: partitions, views per train.
N_Z = 512
VIEWS_PER_TRAIN = 1024

#: In-plane matrix per family, and the interleave count the spiral is designed
#: for. The spiral is held at 128 because its rewinder cannot be rotated above
#: that: the bridge undoes a two-axis k-vector, so both axes run near the
#: gradient ceiling and the vector magnitude is about sqrt(2) times it, which a
#: rotation can land on a single axis. See ``ROTATION HEADROOM`` below.
N_X_CARTESIAN = 512
N_X_SPIRAL = 128
DESIGN_INTERLEAVES = 48

#: A 1024-view train is 15 s (Cartesian) to 29 s (spiral) long, so TI and the
#: outer TR are set to hold it. These are scale cases -- what the design path
#: costs per block at protocol size -- not clinically prescribed protocols.
TI = 8.0
TR_OUTER_CARTESIAN = 20.0
TR_OUTER_SPIRAL = 40.0

#: A real offset, so the transform has work to do rather than a no-op to skip.
FOV_OFFSET = (12e-3, -8e-3, 5e-3)


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def timed_transform():
    """Patch ``TransformFOV.apply_to_sequence`` to accumulate its own time.

    Returns a one-element list the elapsed seconds land in. The plugin is run
    unmodified, so the transform is measured exactly where it actually happens:
    at the end of ``main``, over the finished scan.
    """
    import pulserver.pypulseq as pp

    elapsed = [0.0]
    original = pp.TransformFOV.apply_to_sequence

    def instrumented(self, seq, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original(self, seq, *args, **kwargs)
        finally:
            elapsed[0] += time.perf_counter() - start

    pp.TransformFOV.apply_to_sequence = instrumented
    return elapsed


def cartesian(n_z: int, views: int):
    """Cartesian MPRAGE: one phase-encode line per view."""
    from pulserver.app import mprage3D_sequence

    return mprage3D_sequence(
        n_x=N_X_CARTESIAN,
        n_y=views,
        n_z=n_z,
        slab_thickness=0.192,
        views_per_segment=views,
        fov_offset=FOV_OFFSET,
        ti=TI,
        tr_outer=TR_OUTER_CARTESIAN,
        n_acs=0,
        n_acs_z=0,
        elliptical=False,
    )


def spiral(n_z: int, views: int, *, use_rotation_ext: bool):
    """Stack-of-spirals MPRAGE: one arm per view, turned or written out."""
    from pulserver.app import mprage_stack_of_spirals3D_sequence

    return mprage_stack_of_spirals3D_sequence(
        n_x=N_X_SPIRAL,
        n_z=n_z,
        slab_thickness=0.192,
        n_arms=views,
        etl=views,
        design_interleaves=DESIGN_INTERLEAVES,
        fov_offset=FOV_OFFSET,
        ti=TI,
        tr_outer=TR_OUTER_SPIRAL,
        use_rotation_ext=use_rotation_ext,
    )


CASES = {
    "cartesian": lambda z, v: cartesian(z, v),
    "spiral_rotated": lambda z, v: spiral(z, v, use_rotation_ext=True),
    "spiral_explicit": lambda z, v: spiral(z, v, use_rotation_ext=False),
}


def run(name: str, n_z: int, views: int) -> None:
    """Build one case and print its line."""
    transform = timed_transform()
    start = time.perf_counter()
    seq = CASES[name](n_z, views)
    total = time.perf_counter() - start

    design = total - transform[0]
    blocks = seq.num_blocks
    print(
        f"{name:<16s} {n_z}x{views:<6d} blocks {blocks:>9d}"
        f"  design {design:7.2f}s ({design / blocks * 1e6:5.2f} us/block)"
        f"  TransformFOV {transform[0]:6.2f}s"
        f"  shapes {seq._native.num_shapes():>6d}"
        f"  peak RSS {peak_rss_mb():7.0f} MB"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(CASES), help="run one case")
    parser.add_argument("--scale", type=float, default=1.0, help="shrink the encoding size")
    args = parser.parse_args()

    n_z = max(2, int(N_Z * args.scale))
    views = max(2, int(VIEWS_PER_TRAIN * args.scale))
    for name in [args.only] if args.only else sorted(CASES):
        run(name, n_z, views)
    return 0


if __name__ == "__main__":
    sys.exit(main())
