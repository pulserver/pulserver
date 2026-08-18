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
    >>> from pulserver.design import make_spoiler
    >>> gx, gy, gz = make_spoiler(pp.Opts(), voxel_size=3e-3)

    .. plot::
       :include-source: false

       import numpy as np
       import pypulseq as pp
       from pulserver.design import make_spoiler
       import matplotlib.pyplot as plt
       gx, gy, gz = make_spoiler(pp.Opts(), voxel_size=3e-3)
       t = np.array([0, gx.rise_time, gx.rise_time + gx.flat_time,
                     gx.rise_time + gx.flat_time + gx.fall_time]) * 1e3
       plt.plot(t, [0, gx.amplitude, gx.amplitude, 0])
       plt.xlabel("t [ms]"); plt.ylabel("G [Hz/m]")
    """
    return tuple(
        _crusher_trapezoid(
            system,
            axis,
            dephasing_cycles=dephasing_cycles,
            voxel_size=voxel_size,
        )
        for axis in "xyz"
    )


def _crusher_trapezoid(
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
    >>> from pulserver.design import make_crusher
    >>> gz = make_crusher(pp.Opts(), "z", dephasing_cycles=4.0, voxel_size=5e-3)
    >>> gz_explicit = make_crusher(pp.Opts(), "z", area=800.0)

    .. plot::
       :include-source: false

       import pypulseq as pp
       from pulserver.design import make_crusher
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
    >>> from pulserver.design._gradients import partition_geometry
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
    >>> from pulserver.design._gradients import z_worst_case_trapezoids
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
    >>> from pulserver.design._gradients import combined_z_gradients
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
