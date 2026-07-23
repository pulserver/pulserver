"""Spectral-spatial and arbitrary-trajectory spatial RF pulse design."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pypulseq as pp

from ...pypulseq._opts import default_system
from ._base import RfModule
from ._slr import design_slr


class SpatialPulse(RfModule):
    """RF event, simultaneous encoding gradients, and their rephasers."""

    def __init__(self, system, rf, gradients, rephasers, kspace, self_refocused) -> None:
        self.rf = rf
        self.gradients = tuple(gradients)
        self.rephasers = tuple(rephasers)
        self.kspace = np.asarray(kspace)
        self.self_refocused = bool(self_refocused)
        blocks = [(rf, *self.gradients)]
        if self.rephasers:
            blocks.append(self.rephasers)
        super().__init__(system, blocks)


def _as_tuple(value, ndim: int, name: str, cast=float) -> tuple:
    result = (cast(value),) * ndim if np.isscalar(value) else tuple(cast(item) for item in value)
    if len(result) != ndim:
        raise ValueError(f"{name} must have {ndim} entries")
    return result


def _target_and_coordinates(
    matrix: tuple[int, ...],
    fov: tuple[float, ...],
    selective_size: tuple[float, ...] | None,
    target: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    axes = [(np.arange(n) - (n - 1) / 2.0) * (extent / n) for n, extent in zip(matrix, fov, strict=True)]
    mesh = np.meshgrid(*axes, indexing="ij")
    coordinates = np.stack([axis.ravel() for axis in mesh], axis=1)
    if target is not None:
        target = np.asarray(target, dtype=np.complex128)
        if target.shape != matrix:
            raise ValueError(f"target shape {target.shape} does not match matrix {matrix}")
        return target.ravel(), coordinates

    if selective_size is None:
        selective_size = tuple(0.5 * extent for extent in fov)
    radius_squared = np.zeros(coordinates.shape[0])
    for dim, diameter in enumerate(selective_size):
        if not 0 < diameter <= fov[dim]:
            raise ValueError("each selective_size entry must lie in (0, fov]")
        radius_squared += (coordinates[:, dim] / (diameter / 2.0)) ** 2
    return (radius_squared <= 1.0).astype(np.complex128), coordinates


def _small_tip_weights(target: np.ndarray, coordinates: np.ndarray, kspace: np.ndarray) -> np.ndarray:
    """Evaluate the target's Fourier transform along an excitation trajectory."""
    weights = np.empty(kspace.shape[0], dtype=np.complex128)
    # Bound the temporary matrix to about 64 MiB even for long trajectories.
    chunk = max(1, int(4_000_000 / max(1, coordinates.shape[0])))
    for start in range(0, kspace.shape[0], chunk):
        stop = min(start + chunk, kspace.shape[0])
        phase = coordinates @ kspace[start:stop].T
        weights[start:stop] = target @ np.exp(-2j * np.pi * phase) / target.size
    if np.allclose(weights, 0.0):
        raise ValueError("the requested target has zero response on the supplied trajectory")
    return weights


def _gradient_events_and_rephasers(
    gradient_t_per_m: np.ndarray,
    axes: tuple[str, ...],
    system: pp.Opts,
) -> tuple[tuple, tuple]:
    gradient_hz_per_m = gradient_t_per_m * system.gamma
    gradients = []
    rephasers = []
    for dim, axis in enumerate(axes):
        waveform = np.ascontiguousarray(gradient_hz_per_m[:, dim])
        if np.allclose(waveform, 0.0):
            continue
        event = pp.make_arbitrary_grad(
            channel=axis,
            waveform=waveform,
            first=0.0,
            last=0.0,
            system=system,
        )
        gradients.append(event)
        if not np.isclose(event.area, 0.0, atol=1e-9):
            rephasers.append(pp.make_trapezoid(channel=axis, area=-event.area, system=system))
    return tuple(gradients), tuple(rephasers)


def _assemble_spatial_pulse(
    flip_angle: float,
    gradient: np.ndarray,
    fov: tuple[float, ...],
    matrix: tuple[int, ...],
    selective_size: tuple[float, ...] | None,
    target: np.ndarray | None,
    axes: tuple[str, ...],
    system: pp.Opts,
    use: str,
    freq_offset: float,
    phase_offset: float,
    active_samples: np.ndarray | None = None,
) -> SpatialPulse:
    """Assemble an RF pulse after trajectory arguments have been validated."""
    dwell = system.grad_raster_time
    gradient_hz_per_m = gradient * system.gamma
    # Excitation k-space at an RF sample is the gradient moment remaining
    # after that sample. This convention directly includes the final phase.
    kspace = -np.cumsum(gradient_hz_per_m[::-1], axis=0)[::-1] * dwell
    desired, coordinates = _target_and_coordinates(matrix, fov, selective_size, target)
    weights = _small_tip_weights(desired, coordinates, kspace)
    if active_samples is not None:
        active_samples = np.asarray(active_samples, dtype=bool)
        if active_samples.shape != (len(weights),):
            raise ValueError("active_samples must have one entry per gradient sample")
        weights[~active_samples] = 0.0
    rf = pp.make_arbitrary_rf(
        signal=weights,
        flip_angle=flip_angle,
        dwell=dwell,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        system=system,
        use=use,
    )
    gradients, rephasers = _gradient_events_and_rephasers(gradient, axes, system)
    return SpatialPulse(
        system=system,
        rf=rf,
        gradients=gradients,
        rephasers=rephasers,
        kspace=kspace,
        self_refocused=len(rephasers) == 0,
    )


def _sample_trapezoid(event, dwell: float) -> np.ndarray:
    """Sample a trapezoid at RF-raster midpoints."""
    duration = event.rise_time + event.flat_time + event.fall_time
    time = (np.arange(int(round(duration / dwell))) + 0.5) * dwell
    knots = np.array([0.0, event.rise_time, event.rise_time + event.flat_time, duration])
    values = np.array([0.0, event.amplitude, event.amplitude, 0.0])
    return np.interp(time, knots, values)


def _verse_to_trapezoid(signal: np.ndarray, event, dwell: float) -> np.ndarray:
    """VERSE an RF envelope from a flat gradient onto a trapezoid."""
    gradient = _sample_trapezoid(event, dwell)
    magnitude = np.abs(gradient)
    total = magnitude.sum()
    if total <= 0:
        raise ValueError("VERSE gradient has zero area")
    excitation_position = (np.cumsum(magnitude) - 0.5 * magnitude) / total
    source_position = (np.arange(len(signal)) + 0.5) / len(signal)
    source = np.asarray(signal, dtype=complex)
    interpolated = np.interp(excitation_position, source_position, source.real) + 1j * np.interp(
        excitation_position, source_position, source.imag
    )
    return interpolated * magnitude / np.max(magnitude)


def _alternating_extended_trapezoid(event, count: int, system: pp.Opts):
    """Concatenate alternating trapezoids without rasterizing their ramps."""
    lobe_duration = event.rise_time + event.flat_time + event.fall_time
    times = []
    amplitudes = []
    for index in range(count):
        start = index * lobe_duration
        sign = 1.0 if index % 2 == 0 else -1.0
        knots = (start, start + event.rise_time, start + event.rise_time + event.flat_time, start + lobe_duration)
        values = (0.0, sign * event.amplitude, sign * event.amplitude, 0.0)
        for knot, value in zip(knots, values, strict=True):
            if times and np.isclose(knot, times[-1], atol=1e-15):
                amplitudes[-1] = value
            else:
                times.append(knot)
                amplitudes.append(value)
    return pp.make_extended_trapezoid(
        channel="z",
        times=np.asarray(times),
        amplitudes=np.asarray(amplitudes),
        system=system,
    )


def make_spatially_selective_pulse(
    flip_angle: float,
    gradient: np.ndarray,
    fov: float | Sequence[float],
    matrix: int | Sequence[int],
    *,
    selective_size: float | Sequence[float] | None = None,
    target: np.ndarray | None = None,
    axes: Sequence[str] | None = None,
    system: pp.Opts | None = None,
    use: str = "excitation",
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
) -> SpatialPulse:
    """Design a small-tip 2D/3D pulse on a gradient trajectory you supply.

    Under the small-tip approximation the excitation profile is the Fourier
    transform of the RF envelope sampled along excitation k-space, so the
    envelope is obtained by evaluating the *desired* profile's transform along
    the trajectory the gradients trace. That makes selectivity in two or three
    dimensions — a pencil beam, an inner-volume box, a curved slab — a design
    choice rather than a hardware one.

    The trajectory is deliberately a parameter, not something this function
    designs: EPI paths come from pypulseq, spiral paths from
    :func:`traj2grad`, and anything else from the caller.

    Excitation k-space is the moment *remaining* after each sample, so a
    trajectory ending at the k-space origin is self-refocused
    (``.self_refocused``) and no rephasers are emitted.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    gradient : numpy.ndarray
        Trajectory in T/m, shape ``(samples, 2)`` or ``(samples, 3)``, on the
        gradient raster. Must start and end at zero amplitude.
    fov : float or sequence of float
        Excitation field of view (m) per gradient axis.
    matrix : int or sequence of int
        Profile grid size per axis (>= 2).
    selective_size : float or sequence of float, optional
        Diameter (m) of the default ellipse/ellipsoid target.
    target : numpy.ndarray, optional
        Explicit complex desired profile on the ``matrix`` grid; overrides
        ``selective_size``.
    axes : sequence of str, optional
        Gradient channels for the trajectory columns (default ``x, y[, z]``).
    system : pypulseq.Opts, optional
        System limits.
    use : str, optional
        Pulseq ``use`` tag.
    freq_offset : float, optional
        RF frequency offset (Hz).
    phase_offset : float, optional
        RF phase offset (rad).

    Returns
    -------
    SpatialPulse
        Module with ``.rf``, ``.gradients``, ``.rephasers``, the excitation
        ``.kspace``, and the ``.self_refocused`` flag.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> pulse = pp.make_spiral_selective_pulse(
    ...     np.deg2rad(30), fov=0.256, matrix=32, selective_size=0.04
    ... )
    >>> pulse.kspace.shape[1]
    2

    The excited region in the plane the gradient traverses:

    The RF envelope, the excitation k-space path it is played along, and the resulting in-plane profile:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(pp.make_spiral_selective_pulse(np.deg2rad(30), 0.24, 24,
                                                selective_size=0.05, system=system),
                 "xy", title="small-tip 2D selective, 30 deg", extent=0.12)

    See Also
    --------
    make_2d_selective_pulse : convenience wrapper that can design the spiral.
    make_spiral_selective_pulse : spiral trajectory from ``arbgrad``.
    make_3d_selective_pulse : the 3D entry point.
    """
    system = default_system(system)
    gradient = np.asarray(gradient, dtype=float)
    if gradient.ndim != 2 or gradient.shape[1] not in (2, 3) or gradient.shape[0] < 2:
        raise ValueError("gradient must have shape (samples, 2) or (samples, 3)")
    ndim = gradient.shape[1]
    fov = _as_tuple(fov, ndim, "fov")
    matrix = _as_tuple(matrix, ndim, "matrix", int)
    size = None if selective_size is None else _as_tuple(selective_size, ndim, "selective_size")
    axes = tuple(("x", "y", "z")[:ndim] if axes is None else axes)
    if len(axes) != ndim or len(set(axes)) != ndim or any(axis not in ("x", "y", "z") for axis in axes):
        raise ValueError("axes must contain two or three distinct gradient channels")
    if any(n < 2 for n in matrix) or any(extent <= 0 for extent in fov):
        raise ValueError("matrix entries must be >= 2 and fov entries must be > 0")
    if not np.allclose(gradient[[0, -1]], 0.0, atol=1e-9):
        raise ValueError("gradient must ramp from and return to zero")

    return _assemble_spatial_pulse(
        flip_angle,
        gradient,
        fov,
        matrix,
        size,
        target,
        axes,
        system,
        use,
        freq_offset,
        phase_offset,
    )


def make_spiral_selective_pulse(
    flip_angle: float,
    fov: float,
    matrix: int,
    *,
    selective_size: float | Sequence[float] | None = None,
    target: np.ndarray | None = None,
    system: pp.Opts | None = None,
    **kwargs,
) -> SpatialPulse:
    """Create a 2D selective pulse on an ``arbgrad``-designed spiral.

    The usual 2D-selective choice: a spiral covers excitation k-space in a
    single, gradient-efficient shot, and ends at the origin so the pulse is
    self-refocused. The trajectory is designed at the system's slew and
    gradient limits by the vendored spiral designer; the RF
    envelope then comes from :func:`make_spatially_selective_pulse`.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    fov : float
        Excitation field of view (m); square.
    matrix : int
        Excitation grid size; square.
    selective_size : float or sequence of float, optional
        Diameter (m) of the excited disc.
    target : numpy.ndarray, optional
        Explicit complex desired profile, overriding ``selective_size``.
    system : pypulseq.Opts, optional
        System limits.
    **kwargs
        Forwarded to :func:`make_spatially_selective_pulse`.

    Returns
    -------
    SpatialPulse
        Module with the RF event and both in-plane gradients.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> pulse = pp.make_spiral_selective_pulse(
    ...     np.deg2rad(30), fov=0.256, matrix=32, selective_size=0.04
    ... )
    >>> len(pulse.gradients)
    2

    A spiral excitation k-space path localises in both in-plane axes at once:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(pp.make_spiral_selective_pulse(np.deg2rad(30), 0.24, 24,
                                                selective_size=0.05, system=system),
                 "xy", title="spiral selective, 30 deg, 50 mm spot", extent=0.12)

    See Also
    --------
    make_spatially_selective_pulse : the underlying design.
    """
    # Keep the compiled arbgrad extension optional for all other RF designs.
    from .. import _arbgrad as arbgrad

    system = default_system(system)
    native = arbgrad.spiral(
        fov=fov,
        n_pix=matrix,
        slew_limit=system.max_slew * fov / matrix,
        grad_limit=system.max_grad * fov / matrix,
        dt=system.grad_raster_time,
    )
    # MRArbGrad returns one centre-out interleaf plus the number required for
    # Nyquist coverage. Rewind every rotated interleaf to the origin before
    # starting the next one; RF is active only on the centre-out traversal.
    # Concatenating bare interleaves would instead walk around the cumulative
    # outer-k-space endpoints and produces a striped, displaced profile.
    interleaves = []
    active_samples = []
    for angle in arbgrad.shot_angles(native.n_shots, mode="uniform"):
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        interleaf = native.gradient[:, :2] @ rotation.T
        interleaves.extend((interleaf, -interleaf[::-1]))
        active_samples.extend((np.ones(len(interleaf), dtype=bool), np.zeros(len(interleaf), dtype=bool)))
    native_gradient = np.concatenate(interleaves, axis=0)
    active_samples = np.concatenate(active_samples)
    gradient = arbgrad.to_gradient_tesla_per_meter(native_gradient, fov, matrix, system.gamma)
    size = None if selective_size is None else _as_tuple(selective_size, 2, "selective_size")
    use = kwargs.pop("use", "excitation")
    freq_offset = kwargs.pop("freq_offset", 0.0)
    phase_offset = kwargs.pop("phase_offset", 0.0)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    return _assemble_spatial_pulse(
        flip_angle,
        gradient[:, :2],
        (fov, fov),
        (matrix, matrix),
        size,
        target,
        ("x", "y"),
        system,
        use,
        freq_offset,
        phase_offset,
        active_samples,
    )


def make_plane_selective_pulse(
    flip_angle: float,
    fov: float,
    matrix: int,
    *,
    selective_size: float | Sequence[float] | None = None,
    target: np.ndarray | None = None,
    system: pp.Opts | None = None,
    **kwargs,
) -> SpatialPulse:
    """Create a plane-selective pulse on an internally generated spiral.

    This is the supported multidimensional RF entry point. The spiral covers
    two-dimensional excitation k-space and the RF envelope selects the target
    region within that plane.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    fov : float
        Square excitation field of view (m).
    matrix : int
        Square excitation matrix.
    selective_size : float or sequence of float, optional
        Diameter (m) of the excited disc.
    target : numpy.ndarray, optional
        Explicit complex desired profile, overriding ``selective_size``.
    system : pypulseq.Opts, optional
        System limits.
    **kwargs
        Passed to the small-tip spatial design.

    Returns
    -------
    SpatialPulse
        RF event and both in-plane spiral gradients.

    Examples
    --------
    Append the RF/gradient blocks to a sequence::

        import numpy as np
        import pulserver.design as design
        import pulserver.pypulseq as pp

        pulse = design.make_plane_selective_pulse(
            np.deg2rad(30), 0.24, 24, selective_size=0.05)
        seq = pp.Sequence(pulse.system)
        for block in pulse:
            seq.add_block(*block)

    The complete set of spiral interleaves is concatenated into one module;
    the magnetization map therefore reflects the full plane-selective design:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m",
                        max_slew=150, slew_unit="T/m/s")
       pulse = design.make_plane_selective_pulse(
           np.deg2rad(30), 0.24, 24, selective_size=0.05, system=system)
       rf_figure(pulse, "xy", title="plane selective, 30 deg, 50 mm spot",
                 extent=0.12)
    """
    return make_spiral_selective_pulse(
        flip_angle,
        fov,
        matrix,
        selective_size=selective_size,
        target=target,
        system=system,
        **kwargs,
    )


def make_2d_selective_pulse(
    flip_angle: float,
    fov: float | Sequence[float],
    matrix: int | Sequence[int],
    *,
    gradient: np.ndarray | None = None,
    system: pp.Opts | None = None,
    **kwargs,
) -> SpatialPulse:
    """Create a 2D selective pulse, designing a spiral if none is supplied.

    The general 2D entry point. Pass ``gradient`` to excite along your own
    trajectory (an EPI raster, a custom path); omit it and an ``arbgrad``
    spiral is designed for you — which requires scalar, square ``fov`` and
    ``matrix``.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    fov : float or sequence of float
        Excitation field of view (m); scalar on the automatic spiral path.
    matrix : int or sequence of int
        Excitation grid size; scalar on the automatic spiral path.
    gradient : numpy.ndarray, optional
        Trajectory in T/m, shape ``(samples, 2)``. Designs a spiral when
        omitted.
    system : pypulseq.Opts, optional
        System limits.
    **kwargs
        Forwarded to the selected design (``selective_size``, ``target``,
        ``axes``, ...).

    Returns
    -------
    SpatialPulse
        Module with the RF event and its two in-plane gradients.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> pulse = pp.make_2d_selective_pulse(
    ...     np.deg2rad(30), fov=0.256, matrix=32, selective_size=0.04
    ... )
    >>> pulse.kspace.shape[1]
    2

    The EPI excitation path gives a rectangular in-plane profile:

    With ``gradient`` supplied the pulse follows the caller's excitation path instead of the designed spiral — here an EPI raster, whose blips resolve y while the fast axis stays unselective:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       amplitude, ramp, flat, n_lines = 5e-3, 8, 48, 9
       lobe = np.concatenate([np.linspace(0, 1, ramp), np.ones(flat), np.linspace(1, 0, ramp)])
       blip = np.concatenate([np.linspace(0, 1, 4), np.linspace(1, 0, 4)]) * amplitude * 0.6
       gx, gy = [], []
       for line in range(n_lines):
           gx.append(lobe * amplitude * (1 if line % 2 == 0 else -1))
           gy.append(np.zeros(len(lobe)))
           if line < n_lines - 1:
               gx.append(np.zeros(len(blip)))
               gy.append(blip)
       raster = np.stack([np.concatenate(gx), np.concatenate(gy)], axis=-1)
       rf_figure(pp.make_2d_selective_pulse(np.deg2rad(30), (0.24, 0.24), (24, 24),
                                            gradient=raster, selective_size=0.05,
                                            system=system),
                 "xy", title="2D selective on an EPI raster, 30 deg", extent=0.12)

    See Also
    --------
    make_spiral_selective_pulse, make_spatially_selective_pulse
    """
    if gradient is None:
        if not np.isscalar(fov) or not np.isscalar(matrix):
            raise ValueError("the automatic spiral path requires scalar fov and matrix")
        return make_spiral_selective_pulse(
            flip_angle,
            float(fov),
            int(matrix),
            system=system,
            **kwargs,
        )
    return make_spatially_selective_pulse(
        flip_angle,
        gradient,
        fov,
        matrix,
        system=system,
        **kwargs,
    )


def make_3d_selective_pulse(
    flip_angle: float,
    gradient: np.ndarray,
    fov: Sequence[float],
    matrix: Sequence[int],
    *,
    system: pp.Opts | None = None,
    **kwargs,
) -> SpatialPulse:
    """Create a 3D selective pulse on a trajectory you supply.

    Excites a genuinely three-dimensional region — an inner-volume box, an
    ellipsoid — by playing RF along a 3D excitation-k-space path (stack-of-EPI,
    stack-of-spiral, kooshball). No trajectory is designed here: 3D excitation
    paths are long and application-specific, so the caller supplies one.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    gradient : numpy.ndarray
        Trajectory in T/m, shape ``(samples, 3)``, starting and ending at zero.
    fov : sequence of float
        Excitation field of view (m) along each axis.
    matrix : sequence of int
        Excitation grid size along each axis.
    system : pypulseq.Opts, optional
        System limits.
    **kwargs
        Forwarded to :func:`make_spatially_selective_pulse`.

    Returns
    -------
    SpatialPulse
        Module with the RF event and its three gradients.

    Examples
    --------
    Excite a 40 mm ellipsoid on a caller-supplied trajectory::

        pulse = pp.make_3d_selective_pulse(
            np.deg2rad(20), gradient, fov=(0.256,) * 3, matrix=(24,) * 3,
            selective_size=0.04,
        )

    See Also
    --------
    make_spatially_selective_pulse : the underlying design and its k-space
        convention.
    """
    gradient = np.asarray(gradient)
    if gradient.ndim != 2 or gradient.shape[1] != 3:
        raise ValueError("gradient must have shape (samples, 3)")
    return make_spatially_selective_pulse(
        flip_angle,
        gradient,
        fov,
        matrix,
        system=system,
        **kwargs,
    )


def make_spsp_pulse(
    flip_angle: float,
    slice_thickness: float,
    spectral_bandwidth: float,
    *,
    freq_offset: float = 0.0,
    spatial_time_bandwidth_product: float = 4.0,
    spectral_time_bandwidth_product: float = 3.0,
    n_subpulses: int = 10,
    system: pp.Opts | None = None,
    use: str = "excitation",
) -> SpatialPulse:
    """Create a spectral-spatial (SPSP) pulse on an alternating gradient.

    Selects a slice *and* a spectral band in one pulse: a train of short
    slice-selective subpulses is played on an alternating slice gradient, and
    the subpulse *envelope* — sampled at the subpulse repetition rate — sets
    the spectral profile. Water-selective excitation therefore needs no
    separate fat-saturation module and costs no extra TR time.

    Both envelopes are SLR designs rather than truncated sincs; the
    construction otherwise follows Peder Larson's compact SPSP example.

    ``n_subpulses`` is rounded up to an even count so the alternating gradient
    ends balanced. A short slice and a wide spectral band together force very
    fast gradient lobes — if the design would exceed ``system.max_slew`` a
    ``ValueError`` is raised, and the fix is a thicker slice, a narrower
    spectral band, or fewer subpulses.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    slice_thickness : float
        Slice thickness (m).
    spectral_bandwidth : float
        Spectral passband (Hz).
    freq_offset : float, optional
        Centre of the spectral passband (Hz), e.g. the fat offset to suppress.
    spatial_time_bandwidth_product : float, optional
        Time-bandwidth product of each spatial subpulse.
    spectral_time_bandwidth_product : float, optional
        Time-bandwidth product of the spectral envelope; sets the total
        duration as ``spectral_time_bandwidth_product / spectral_bandwidth``.
    n_subpulses : int, optional
        Number of subpulses (>= 4, rounded up to even).
    system : pypulseq.Opts, optional
        System limits.
    use : str, optional
        Pulseq ``use`` tag.

    Returns
    -------
    SpatialPulse
        Module with the RF event, the alternating slice gradient, and its
        rephaser.

    Raises
    ------
    ValueError
        If the requested selectivity exceeds the slew limit, or the spectral
        bandwidth is too wide for the requested subpulse count.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> pulse = design.make_spsp_pulse(
    ...     np.deg2rad(30), slice_thickness=10e-3,
    ...     spectral_bandwidth=300.0, n_subpulses=12, system=system,
    ... )
    >>> len(pulse.gradients), len(pulse.rephasers)
    (1, 1)

    Append the single RF waveform, sparse alternating gradient, and rephaser::

        seq = pp.Sequence(pulse.system)
        for block in pulse:
            seq.add_block(*block)

    Selective in space *and* in frequency — the whole point of the design. Fat, several hundred Hz away, falls in a null:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(design.make_spsp_pulse(np.deg2rad(90), 1e-2, 250.0, system=system),
                 "zf", title="spectral-spatial 90 deg, 10 mm, 250 Hz",
                 extent=0.025, span=700)

    See Also
    --------
    make_fat_saturation_pulse : the separate-module alternative.
    make_frequency_selective_pulse : purely spectral selectivity.
    """
    system = default_system(system)
    if slice_thickness <= 0 or spectral_bandwidth <= 0:
        raise ValueError("slice_thickness and spectral_bandwidth must be > 0")
    if n_subpulses < 4:
        raise ValueError("n_subpulses must be >= 4")
    if n_subpulses % 2:
        n_subpulses += 1

    grad_raster = system.grad_raster_time
    rf_raster = system.rf_raster_time
    total_duration = spectral_time_bandwidth_product / spectral_bandwidth
    lobe_duration = round(total_duration / (n_subpulses * grad_raster)) * grad_raster
    if lobe_duration < 8 * grad_raster:
        raise ValueError("spectral bandwidth is too large for the requested subpulse count")

    # Reserve 20% of each lobe for the two ramps, on the gradient raster.
    ramp_time = max(grad_raster, round(0.1 * lobe_duration / grad_raster) * grad_raster)
    flat_time = lobe_duration - 2.0 * ramp_time
    if flat_time < 8 * rf_raster:
        raise ValueError("SPSP sublobes leave fewer than 8 RF samples")
    amplitude_hz_per_m = spatial_time_bandwidth_product / (flat_time * slice_thickness)
    if amplitude_hz_per_m > system.max_grad:
        raise ValueError("SPSP slice-selection amplitude exceeds system.max_grad")
    slew = amplitude_hz_per_m / ramp_time
    if slew > system.max_slew * (1.0 + 1e-9):
        raise ValueError("SPSP slice-selection ramps exceed system.max_slew")

    # Design the spatial SLR pulse on the flat top, build one trapezoid, then
    # VERSE the RF onto its ramps. This keeps time-bandwidth product intact
    # while allowing the RF and gradient to overlap for the entire lobe.
    from ._excitation import make_slr_pulse

    spatial_module = make_slr_pulse(
        1.0,
        duration=flat_time,
        dwell=rf_raster,
        time_bw_product=spatial_time_bandwidth_product,
        pulse_type="st",
        system=system,
        use=use,
    )
    positive_lobe = pp.make_trapezoid(
        channel="z",
        amplitude=amplitude_hz_per_m,
        flat_time=flat_time,
        rise_time=ramp_time,
        fall_time=ramp_time,
        system=system,
    )
    spatial = _verse_to_trapezoid(spatial_module.rf.signal, positive_lobe, rf_raster)
    spectral = design_slr(n_subpulses, spectral_time_bandwidth_product, pulse_type="st", filter_type="ls")

    rf_shape = np.empty(n_subpulses * spatial.size, dtype=np.complex128)
    for index in range(n_subpulses):
        start = index * spatial.size
        stop = start + spatial.size
        subpulse = spatial if index % 2 == 0 else spatial[::-1]
        rf_shape[start:stop] = spectral[index] * subpulse

    rf = pp.make_arbitrary_rf(
        signal=rf_shape,
        flip_angle=flip_angle,
        dwell=rf_raster,
        freq_offset=freq_offset,
        system=system,
        use=use,
    )
    gz = _alternating_extended_trapezoid(positive_lobe, n_subpulses, system)
    # Rephase from the center of the final spatial subpulse. The alternating
    # full-lobe areas cancel for even n_subpulses, but this half-lobe remains.
    last_sign = -1.0
    rephase_area = -last_sign * 0.5 * positive_lobe.area
    gz_reph = pp.make_trapezoid(channel="z", area=rephase_area, system=system)
    dense_gradient = np.concatenate(
        [
            _sample_trapezoid(positive_lobe, rf_raster) * (1.0 if index % 2 == 0 else -1.0)
            for index in range(n_subpulses)
        ]
    )
    kspace = -np.cumsum(dense_gradient[::-1])[::-1, None] * rf_raster
    return SpatialPulse(
        system=system,
        rf=rf,
        gradients=(gz,),
        rephasers=(gz_reph,),
        kspace=kspace,
        self_refocused=False,
    )
