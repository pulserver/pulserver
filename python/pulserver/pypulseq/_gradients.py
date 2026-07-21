"""Gradient factories and private helpers used by Pulserver modules."""

from __future__ import annotations

import numpy as np
import pypulseq as pp

DEFAULT_SPOIL_CYCLES = 4.0
SPOIL_FACTOR_X = 4.0
SPOIL_FACTOR_Z = 2.0
CRUSHER_CYCLES_Z = 4.0


def make_spoiler(
    system: pp.Opts,
    voxel_size: float,
    *,
    dephasing_cycles: float = DEFAULT_SPOIL_CYCLES,
):
    """Build a non-selective 3-axis crusher/spoiler gradient set.

    ``voxel_size`` is the spoiling reference length (typically the
    in-plane resolution FOV/N) — used e.g. after a non-selective inversion
    pulse, which has no dedicated slice axis to crush on.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    voxel_size : float
        Reference length (m) across which dephasing accrues.
    dephasing_cycles : float, optional
        Dephasing cycles per ``voxel_size``.

    Returns
    -------
    gx, gy, gz : SimpleNamespace
        One trapezoid per axis, all with area
        ``dephasing_cycles / voxel_size``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.pypulseq import make_spoiler
    >>> gx, gy, gz = make_spoiler(pp.Opts(), voxel_size=3e-3)

    .. plot::
       :include-source: false

       import numpy as np
       import pypulseq as pp
       from pulserver.pypulseq import make_spoiler
       import matplotlib.pyplot as plt
       gx, gy, gz = make_spoiler(pp.Opts(), voxel_size=3e-3)
       t = np.array([0, gx.rise_time, gx.rise_time + gx.flat_time,
                     gx.rise_time + gx.flat_time + gx.fall_time]) * 1e3
       plt.plot(t, [0, gx.amplitude, gx.amplitude, 0])
       plt.xlabel("t [ms]"); plt.ylabel("G [Hz/m]")
    """
    return tuple(
        make_crusher(
            system,
            axis,
            dephasing_cycles=dephasing_cycles,
            voxel_size=voxel_size,
        )
        for axis in "xyz"
    )


def make_phase_encoding(
    system: pp.Opts,
    axis: str,
    fov: float,
    matrix: int,
    *,
    acceleration: float = 1.0,
    partial_fourier: float = 1.0,
) -> tuple:
    """Build the phase-encode template trapezoid and per-view area array.

    The returned ``areas`` follows the absolute-index convention used across
    the zoo plugins: entry ``i`` is the area of view ``i`` regardless of
    acceleration, so undersampling patterns (from Pulserver's sampling
    helpers)
    index into it directly. ``accel``/``pf`` only bound the template to the
    worst-case *sampled* view.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    axis : str
        Gradient channel (``"x"``, ``"y"``, or ``"z"``).
    fov : float
        Field of view along ``axis`` (m).
    matrix : int
        Number of encoded views.
    acceleration : float, optional
        Parallel-imaging acceleration along this axis (documentation of
        intent; view 0 is always sampled so it does not shrink the template).
    partial_fourier : float, optional
        Partial-Fourier fraction in ``(0, 1]``; the first
        ``round((1 - pf) * n)`` (most-negative-k) views are considered
        unsampled when bounding the template.

    Returns
    -------
    template : SimpleNamespace
        Trapezoid with the worst-case sampled area, to be ``pp.scale_grad``-ed
        per view.
    areas : numpy.ndarray
        Per-view areas, length ``n``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.pypulseq import make_phase_encoding
    >>> template, areas = make_phase_encoding(pp.Opts(), "y", 0.22, 64)
    >>> len(areas)
    64

    .. plot::
       :include-source: false

       import pypulseq as pp
       from pulserver.pypulseq import make_phase_encoding
       import matplotlib.pyplot as plt
       template, areas = make_phase_encoding(pp.Opts(), "y", 0.22, 64)
       plt.stem(range(64), areas)
       plt.xlabel("view"); plt.ylabel("area [1/m]")
    """
    if not (0.0 < partial_fourier <= 1.0):
        raise ValueError(f"partial_fourier must be in (0, 1], got {partial_fourier}")
    if acceleration < 1.0:
        raise ValueError(f"acceleration must be >= 1, got {acceleration}")
    delta_k = 1.0 / fov
    areas = (np.arange(matrix) - 0.5 * matrix) * delta_k
    first_sampled = int(round((1.0 - partial_fourier) * matrix))
    max_area = float(np.max(np.abs(areas[first_sampled:]))) if matrix > 0 else 0.0
    template = pp.make_trapezoid(channel=axis, area=max_area, system=system)
    return template, areas


def make_crusher(
    system: pp.Opts,
    axis: str,
    *,
    dephasing_cycles: float | None = None,
    area: float | None = None,
    voxel_size: float | None = None,
):
    """Build a single-axis crusher trapezoid.

    Specify the gradient either by ``dephasing_cycles`` + ``voxel_size``
    (area = ``dephasing_cycles / voxel_size``) or by an explicit ``area``.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    axis : str
        Gradient channel.
    dephasing_cycles : float or None, optional
        Dephasing cycles across ``voxel_size``.
    area : float or None, optional
        Explicit gradient area (1/m).
    voxel_size : float or None, optional
        Reference length for ``dephasing_cycles``.

    Returns
    -------
    SimpleNamespace
        The crusher trapezoid.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.pypulseq import make_crusher
    >>> gz = make_crusher(pp.Opts(), "z", dephasing_cycles=4.0, voxel_size=5e-3)
    >>> gz_explicit = make_crusher(pp.Opts(), "z", area=800.0)

    .. plot::
       :include-source: false

       import pypulseq as pp
       from pulserver.pypulseq import make_crusher
       import matplotlib.pyplot as plt
       g = make_crusher(pp.Opts(), "z", dephasing_cycles=4.0, voxel_size=5e-3)
       t = [0, g.rise_time, g.rise_time + g.flat_time,
            g.rise_time + g.flat_time + g.fall_time]
       plt.plot([x * 1e3 for x in t], [0, g.amplitude, g.amplitude, 0])
       plt.xlabel("t [ms]"); plt.ylabel("G [Hz/m]")
    """
    by_cycles = dephasing_cycles is not None
    by_area = area is not None
    if by_cycles == by_area:
        raise ValueError("specify exactly one of dephasing_cycles(+voxel_size) or area")
    if by_cycles:
        if voxel_size is None:
            raise ValueError("voxel_size is required when specifying dephasing_cycles")
        area = dephasing_cycles / voxel_size
    return pp.make_trapezoid(channel=axis, area=area, system=system)


def make_phase_blip(
    system: pp.Opts,
    axis: str,
    fov: float,
    *,
    steps: float = 1.0,
    duration: float | None = None,
):
    """Create a phase blip spanning ``steps`` Cartesian k-space cells.

    The short gradient played between echoes of a train to advance the phase
    encode. Stated in *cells* rather than area: one cell is ``1 / fov``, so a
    blip of ``steps=1`` moves one line and ``steps=R`` implements an
    acceleration of ``R`` without any further arithmetic. Negative ``steps``
    blip backwards.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    axis : str
        Gradient channel (``"x"``, ``"y"``, or ``"z"``).
    fov : float
        Field of view along ``axis`` (m); one k-space cell is ``1 / fov``.
    steps : float, optional
        Cells to traverse; non-zero, may be negative.
    duration : float or None, optional
        Force a blip duration (s); ``None`` uses the shortest feasible.

    Returns
    -------
    SimpleNamespace
        Blip trapezoid with area ``steps / fov``.

    Examples
    --------
    >>> import pypulseq as pp_up
    >>> from pulserver.pypulseq import make_phase_blip
    >>> blip = make_phase_blip(pp_up.Opts(), "y", 0.24, steps=2)
    >>> round(blip.area, 3)
    8.333

    Blip by the increments a sampling plan prescribes::

        for step in plan.increments(shot)[:, 0]:
            seq.add_block(make_phase_blip(system, "y", fov, steps=step))

    See Also
    --------
    make_phase_encoding : the full-area encode used once per shot.
    """
    if fov <= 0:
        raise ValueError("fov must be > 0")
    if steps == 0:
        raise ValueError("steps must be non-zero")
    kwargs = {"channel": axis, "area": float(steps) / float(fov), "system": system}
    if duration is not None:
        kwargs["duration"] = duration
    return pp.make_trapezoid(**kwargs)


def partition_geometry(npar: int, slice_spacing_m: float):
    """Return ``(par_areas, max_par_area)`` for ``npar`` 3D partitions.

    ``par_areas`` mirrors the ``phase_areas`` convention used for the Y
    phase encode; ``max_par_area`` is 0.0 when ``npar <= 1`` (no encoding).

    Parameters
    ----------
    npar : int
        Number of partitions.
    slice_spacing_m : float
        Partition spacing (m); slab extent is ``npar * slice_spacing_m``.

    Returns
    -------
    par_areas : numpy.ndarray
        Per-partition encode areas (1/m).
    max_par_area : float
        Largest absolute area.

    Examples
    --------
    >>> from pulserver.pypulseq._gradients import partition_geometry
    >>> areas, max_area = partition_geometry(8, 1e-3)
    >>> len(areas)
    8
    """
    if npar <= 1:
        return np.zeros(max(npar, 1)), 0.0
    slab_extent_m = slice_spacing_m * npar
    delta_k_par = 1.0 / slab_extent_m
    par_areas = (np.arange(npar) - 0.5 * npar) * delta_k_par
    max_par_area = float(np.max(np.abs(par_areas)))
    return par_areas, max_par_area


def z_worst_case_trapezoids(gz_reph, gz_spoil, max_par_area: float, opts: pp.Opts):
    """Build upper-bound-duration combined Z trapezoids for feasibility checks.

    Used only for TR/TE feasibility estimation — see
    :func:`combined_z_gradients` for the actual per-partition gradients used
    when building the sequence.

    Parameters
    ----------
    gz_reph : SimpleNamespace
        Slab-rephase trapezoid.
    gz_spoil : SimpleNamespace
        Z spoiler trapezoid.
    max_par_area : float
        Largest partition-encode area (1/m).
    opts : pypulseq.Opts
        System limits.

    Returns
    -------
    gz_pre_worst, gz_post_worst : SimpleNamespace
        Worst-case pre-/post-readout Z trapezoids.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.pypulseq._gradients import z_worst_case_trapezoids
    >>> opts = pp.Opts()
    >>> gz_reph = pp.make_trapezoid(channel="z", area=-200.0, system=opts)
    >>> gz_spoil = pp.make_trapezoid(channel="z", area=400.0, system=opts)
    >>> pre, post = z_worst_case_trapezoids(gz_reph, gz_spoil, 100.0, opts)
    """
    gz_pre_worst = pp.make_trapezoid(channel="z", area=abs(gz_reph.area) + max_par_area, system=opts)
    gz_post_worst = pp.make_trapezoid(channel="z", area=max_par_area + abs(gz_spoil.area), system=opts)
    return gz_pre_worst, gz_post_worst


def combined_z_gradients(z_scale: float, gz_pe_template, gz_reph, gz_spoil, opts: pp.Opts):
    """Fold rephase+encode (pre) and un-encode+spoil (post) into single Z trapezoids.

    A pypulseq block can only carry one gradient per channel, so the fixed
    RF slab-rephase area and the variable per-partition encode area must be
    combined into one Z-channel event per block.

    Parameters
    ----------
    z_scale : float
        Partition scaling in ``[-1, 1]`` applied to ``gz_pe_template``.
    gz_pe_template : SimpleNamespace
        Max-area partition-encode trapezoid.
    gz_reph : SimpleNamespace
        Slab-rephase trapezoid.
    gz_spoil : SimpleNamespace
        Z spoiler trapezoid.
    opts : pypulseq.Opts
        System limits.

    Returns
    -------
    gz_pre_combined, gz_post_combined : SimpleNamespace
        Combined Z trapezoids for the pre- and post-readout blocks.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.pypulseq._gradients import combined_z_gradients
    >>> opts = pp.Opts()
    >>> gz_reph = pp.make_trapezoid(channel="z", area=-200.0, system=opts)
    >>> gz_spoil = pp.make_trapezoid(channel="z", area=400.0, system=opts)
    >>> gz_pe = pp.make_trapezoid(channel="z", area=100.0, system=opts)
    >>> pre, post = combined_z_gradients(0.5, gz_pe, gz_reph, gz_spoil, opts)
    """
    gz_pe_scaled = pp.scale_grad(gz_pe_template, z_scale)
    gz_pre_combined = pp.make_trapezoid(channel="z", area=gz_reph.area + gz_pe_scaled.area, system=opts)
    gz_post_combined = pp.make_trapezoid(channel="z", area=-gz_pe_scaled.area + gz_spoil.area, system=opts)
    return gz_pre_combined, gz_post_combined
