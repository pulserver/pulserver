"""Class-based non-Cartesian readouts.

The trajectory objects (:class:`Radial`, :class:`Spiral`, and
:class:`Rosette`) describe one canonical base interleave.  The train objects
then place that interleave in 2D, promote it to a Cartesian-encoded stack or
projection acquisition, or play a native 3D trajectory.  Shot-to-shot
orientation is a Pulseq rotation extension and is deliberately separate from
trajectory generation.
"""

from __future__ import annotations

__all__ = [
    "NonCartesianGradient",
    "Arbitrary",
    "Radial",
    "Spiral",
    "Rosette",
    "NonCartesian2D",
    "StackOfTrajectories",
    "Projection",
    "NonCartesian3D",
    "NonCartesianStack3D",
    "NonCartesianProjection",
    "radial_trajectory",
    "spiral_trajectory",
    "rosette_trajectory",
]

import copy
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pypulseq as pp

from .._system import DEFAULT_BANDWIDTH_HZ_PX, apply_system_derates, ceil_to_raster
from .._traj2grad import traj2grad
from ._base import Readout, normalize_rotation

_AXES = ("x", "y", "z")
_SPIRAL_VARIANTS = ("outward", "inward", "in_out")
_SPIRAL_DENSITIES = ("constant", "variable", "dual")


@dataclass(frozen=True)
class _NonCartesianState:
    rotation: object | None
    rotation_idx: int
    phase_idx: int | None
    adc_phase_rad: float


def _as_scalar(value, name, cast=float):
    array = np.asarray(value)
    if array.ndim == 0:
        return cast(array)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    first = cast(array.flat[0])
    if not np.all(array == array.flat[0]):
        raise ValueError(f"{name} must be isotropic for a rotationally symmetric trajectory")
    return first


def _validate_common(fov, matrix, oversamp, bandwidth_hz_px):
    fov_m = _as_scalar(fov, "fov")
    n = _as_scalar(matrix, "matrix", int)
    if fov_m <= 0 or n < 2:
        raise ValueError("fov must be positive and matrix must be >= 2")
    if oversamp < 1:
        raise ValueError("oversamp must be >= 1")
    if bandwidth_hz_px <= 0:
        raise ValueError("bandwidth_hz_px must be positive")
    return fov_m, n


def _cumtrapz(values, x):
    out = np.zeros_like(values, dtype=float)
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(x))
    return out


def radial_trajectory(fov, matrix, *, num_points=None):
    """Return a canonical full radial spoke in cycles/m."""
    fov_m, n = _validate_common(fov, matrix, 1.0, 1.0)
    count = int(num_points or n)
    if count < 2:
        raise ValueError("num_points must be >= 2")
    kmax = n / (2.0 * fov_m)
    return np.column_stack((np.linspace(-kmax, kmax, count), np.zeros(count)))


def spiral_trajectory(
    fov,
    matrix,
    design_interleaves,
    *,
    density="constant",
    inner_design_interleaves=None,
    outer_design_interleaves=None,
    variable_density_power=2.0,
    transition_radius=0.5,
    transition_speed=12.0,
    num_points=1024,
):
    """Generate one NumPy spiral-out interleave in cycles/m.

    ``design_interleaves`` sets the nominal constant-density pitch; it does
    not prescribe how many rotations the caller acquires.  The optional
    ``inner_design_interleaves`` and ``outer_design_interleaves`` describe the
    local pitch for variable- and dual-density designs.  Constant density uses
    one value, variable density changes smoothly as
    ``radius ** variable_density_power``, and dual density uses two plateaus
    joined by a logistic transition.
    """
    fov_m, n = _validate_common(fov, matrix, 1.0, 1.0)
    design_interleaves = int(design_interleaves)
    num_points = int(num_points)
    if design_interleaves < 1 or num_points < 4:
        raise ValueError("design_interleaves must be >= 1 and num_points must be >= 4")
    if density not in _SPIRAL_DENSITIES:
        raise ValueError(f"density must be one of {_SPIRAL_DENSITIES}, got {density!r}")

    inner = float(design_interleaves if inner_design_interleaves is None else inner_design_interleaves)
    if outer_design_interleaves is None:
        if density == "variable":
            outer = 2.0 * inner
        elif density == "dual":
            raise ValueError("outer_design_interleaves is required for dual-density spirals")
        else:
            outer = inner
    else:
        outer = float(outer_design_interleaves)
    if inner <= 0 or outer <= 0:
        raise ValueError("inner_design_interleaves and outer_design_interleaves must be positive")

    radius = np.linspace(0.0, 1.0, num_points)
    if density == "constant":
        local_interleaves = np.full_like(radius, inner)
    elif density == "variable":
        if variable_density_power <= 0:
            raise ValueError("variable_density_power must be positive")
        local_interleaves = inner + (outer - inner) * radius ** float(variable_density_power)
    else:
        if not 0.0 < transition_radius < 1.0 or transition_speed <= 0:
            raise ValueError("transition_radius must be in (0, 1) and transition_speed must be positive")
        blend = 1.0 / (1.0 + np.exp(-float(transition_speed) * (radius - float(transition_radius))))
        blend = (blend - blend[0]) / (blend[-1] - blend[0])
        local_interleaves = inner + (outer - inner) * blend

    # For a square matrix, dphi/dr = pi*N/n_interleaves gives N/2 turns
    # for a single-shot constant-density spiral and the corresponding local
    # pitch for multi-shot / variable-density paths.
    phi = _cumtrapz(np.pi * n / local_interleaves, radius)
    kmax = n / (2.0 * fov_m)
    rho = kmax * radius
    return np.column_stack((rho * np.cos(phi), rho * np.sin(phi)))


def rosette_trajectory(
    fov,
    matrix,
    *,
    petals=5,
    angular_frequency_ratio=3.0 / 5.0,
    num_points=2049,
):
    """Generate one multi-petal rosette base interleave in cycles/m.

    The path is

    ``rho(u) = kmax * sin(pi * petals * u)`` and
    ``theta(u) = pi * petals * angular_frequency_ratio * u``.

    Consequently, ``petals`` is the number of center-to-center radial lobes
    played within this one interleave.  Increasing it adds more k-space
    center crossings and lengthens the gradient waveform.  The angular
    frequency ratio is ``omega_angular / omega_radial``: zero degenerates to
    a repeatedly traversed line, values below one produce relatively open
    petals, one produces the constant-speed circular limiting case, and
    values above one wind more tightly while each radial lobe is played.
    Neither parameter describes shot-to-shot rotations; callers rotate the
    complete returned interleave independently.

    ``num_points`` only controls the numerical polyline used to describe the
    ideal path.  It does not set the number of acquired ADC samples.
    """
    fov_m, n = _validate_common(fov, matrix, 1.0, 1.0)
    petals, num_points = int(petals), int(num_points)
    angular_frequency_ratio = float(angular_frequency_ratio)
    if petals < 1 or not math.isfinite(angular_frequency_ratio) or angular_frequency_ratio <= 0 or num_points < 5:
        raise ValueError("petals and angular_frequency_ratio must be positive and num_points must be >= 5")
    u = np.linspace(0.0, 1.0, num_points)
    rho = (n / (2.0 * fov_m)) * np.sin(np.pi * petals * u)
    theta = np.pi * petals * angular_frequency_ratio * u
    return np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))


def _stretch_gradient(gradient, target_duration, raster):
    """Time-stretch ``gradient`` to at least ``target_duration``, preserving area."""
    gradient = np.asarray(gradient, dtype=float)
    old_n = gradient.shape[0]
    new_n = max(old_n, int(math.ceil(target_duration / raster - 1e-12)))
    if new_n == old_n:
        return gradient
    k_edges = np.vstack((np.zeros((1, gradient.shape[1])), np.cumsum(gradient, axis=0) * raster))
    old_u = np.linspace(0.0, 1.0, old_n + 1)
    new_u = np.linspace(0.0, 1.0, new_n + 1)
    interp = np.column_stack([np.interp(new_u, old_u, k_edges[:, axis]) for axis in range(gradient.shape[1])])
    return np.diff(interp, axis=0) / raster


def _make_grad_events(system, gradient, axes, *, first=None, last=None):
    events = []
    for axis_index, channel in enumerate(axes):
        waveform = np.ascontiguousarray(gradient[:, axis_index])
        events.append(
            pp.make_arbitrary_grad(
                channel=channel,
                waveform=waveform,
                first=None if first is None else float(first[axis_index]),
                last=None if last is None else float(last[axis_index]),
                system=system,
            )
        )
    return tuple(events)


def _make_adc(system, n_samples, read_duration):
    """Make an ADC that fits inside ``read_duration`` on the ADC raster."""
    n_samples = int(n_samples)
    dwell = math.floor(read_duration / n_samples / system.adc_raster_time + 1e-10) * system.adc_raster_time
    if dwell < system.adc_raster_time:
        raise ValueError("readout is too short for the requested ADC oversampling")
    return pp.make_adc(num_samples=n_samples, dwell=dwell, system=system)


def _sample_gradient_trajectory(gradient, raster, adc):
    """Integrate a rasterized gradient at the ADC sample-center times."""
    gradient = np.asarray(gradient, dtype=float)
    k_edges = np.vstack((np.zeros((1, gradient.shape[1])), np.cumsum(gradient, axis=0) * raster))
    edge_times = np.arange(gradient.shape[0] + 1, dtype=float) * raster
    sample_times = float(adc.delay) + (np.arange(int(adc.num_samples), dtype=float) + 0.5) * float(adc.dwell)
    return np.column_stack([np.interp(sample_times, edge_times, k_edges[:, axis]) for axis in range(gradient.shape[1])])


def _moment_bridges(system, area, grad_start, grad_end, axes):
    events = []
    for i, channel in enumerate(axes):
        if abs(area[i]) < 1e-12 and abs(grad_start[i]) < 1e-12 and abs(grad_end[i]) < 1e-12:
            continue
        grad, _, _ = pp.make_extended_trapezoid_area(
            area=float(area[i]),
            channel=channel,
            grad_start=float(grad_start[i]),
            grad_end=float(grad_end[i]),
            system=system,
        )
        events.append(grad)
    return tuple(events)


class NonCartesianGradient:
    """One canonical non-Cartesian base interleave, independent of its acquisition schedule."""

    def __init__(
        self,
        *,
        system,
        gradients,
        adc,
        trajectory,
        design_interleaves=None,
        recommended_rotations=None,
        prewinders=(),
        rewinders=(),
        variant=None,
    ):
        self.system = system
        self.gradients = tuple(gradients)
        self.adc = adc
        self.trajectory = np.asarray(trajectory, dtype=float)
        self.design_interleaves = None if design_interleaves is None else int(design_interleaves)
        self.recommended_rotations = None if recommended_rotations is None else int(recommended_rotations)
        self.prewinders = tuple(prewinders)
        self.rewinders = tuple(rewinders)
        self.variant = variant
        self.n_samples = int(adc.num_samples)
        self.adc_dwell_s = float(adc.dwell)
        self.bandwidth_hz_px = 1.0 / self.adc_dwell_s
        self.read_duration = pp.calc_duration(*self.gradients, adc)
        self.duration = (
            (max((pp.calc_duration(g) for g in self.prewinders), default=0.0))
            + self.read_duration
            + (max((pp.calc_duration(g) for g in self.rewinders), default=0.0))
        )

    @property
    def has_prewinder(self):
        return bool(self.prewinders)

    @property
    def has_rewinder(self):
        return bool(self.rewinders)

    @property
    def gx(self):
        return next((g for g in self.gradients if g.channel == "x"), None)

    @property
    def gy(self):
        return next((g for g in self.gradients if g.channel == "y"), None)

    @property
    def gz(self):
        return next((g for g in self.gradients if g.channel == "z"), None)


class Arbitrary(NonCartesianGradient):
    """Build a canonical 2D or 3D base interleave from a user-provided NumPy path."""

    def __init__(
        self,
        system,
        trajectory,
        *,
        matrix,
        bandwidth_hz_px=DEFAULT_BANDWIDTH_HZ_PX,
        oversamp=1.0,
        axes=None,
        solver_oversampling=8,
        derate=True,
    ):
        path = np.asarray(trajectory, dtype=float)
        if path.ndim != 2 or path.shape[1] not in (2, 3) or path.shape[0] < 4:
            raise ValueError("trajectory must have shape (n, 2) or (n, 3), with n >= 4")
        n = _as_scalar(matrix, "matrix", int)
        if n < 2 or oversamp < 1 or bandwidth_hz_px <= 0:
            raise ValueError("matrix must be >= 2, oversamp >= 1, and bandwidth_hz_px positive")
        axes = tuple(_AXES[: path.shape[1]] if axes is None else axes)
        if len(axes) != path.shape[1] or len(set(axes)) != len(axes) or any(axis not in _AXES for axis in axes):
            raise ValueError("axes must contain one distinct gradient channel per trajectory dimension")
        if derate:
            apply_system_derates(system)

        k_start = path[0]
        has_pre = not np.allclose(k_start, 0.0, atol=1e-12)
        has_rew = not np.allclose(path[-1], 0.0, atol=1e-12)
        n_adc = max(2, int(round(n * oversamp)))
        gradient = traj2grad(
            path,
            system,
            oversampling=solver_oversampling,
            start_at_zero=not has_pre,
            end_at_zero=not has_rew,
        )[:, : path.shape[1]]
        gradient = _stretch_gradient(gradient, n_adc / bandwidth_hz_px, system.grad_raster_time)
        first, last = gradient[0], gradient[-1]
        gradients = _make_grad_events(system, gradient, axes, first=first, last=last)
        prewinders = _moment_bridges(system, k_start, np.zeros(path.shape[1]), first, axes) if has_pre else ()
        actual_end = k_start + np.sum(gradient, axis=0) * system.grad_raster_time
        rewinders = _moment_bridges(system, -actual_end, last, np.zeros(path.shape[1]), axes) if has_rew else ()
        adc = _make_adc(system, n_adc, gradient.shape[0] * system.grad_raster_time)
        super().__init__(
            system=system,
            gradients=gradients,
            adc=adc,
            trajectory=path,
            prewinders=prewinders,
            rewinders=rewinders,
            variant="arbitrary",
        )


class Radial(NonCartesianGradient):
    """Full-spoke radial readout with bridged prewinder and rewinder."""

    def __init__(
        self,
        system,
        fov,
        matrix,
        *,
        bandwidth_hz_px=DEFAULT_BANDWIDTH_HZ_PX,
        oversamp=1.0,
        ro_axis="x",
        derate=True,
    ):
        fov_m, n = _validate_common(fov, matrix, oversamp, bandwidth_hz_px)
        if derate:
            apply_system_derates(system)
        raster = system.grad_raster_time
        kmax = n / (2.0 * fov_m)
        n_adc = max(2, int(round(n * oversamp)))
        target_duration = n_adc / float(bandwidth_hz_px)
        read_duration = ceil_to_raster(max(target_duration, 2.0 * kmax / system.max_grad), raster)
        amplitude = 2.0 * kmax / read_duration
        grad = pp.make_extended_trapezoid(
            channel=ro_axis,
            times=np.array([0.0, read_duration]),
            amplitudes=np.array([amplitude, amplitude]),
            system=system,
        )
        pre, _, _ = pp.make_extended_trapezoid_area(
            area=-kmax, channel=ro_axis, grad_start=0.0, grad_end=amplitude, system=system
        )
        rew, _, _ = pp.make_extended_trapezoid_area(
            area=-kmax, channel=ro_axis, grad_start=amplitude, grad_end=0.0, system=system
        )
        adc = _make_adc(system, n_adc, read_duration)
        trajectory = radial_trajectory(fov_m, n, num_points=n_adc)
        super().__init__(
            system=system,
            gradients=(grad,),
            adc=adc,
            trajectory=trajectory,
            recommended_rotations=int(math.ceil(np.pi * n / 2.0)),
            prewinders=(pre,),
            rewinders=(rew,),
            variant="full",
        )


class Spiral(NonCartesianGradient):
    """Build one constant-, variable-, or dual-density spiral interleave.

    ``design_interleaves`` is the nominal number of rotated interleaves used
    to set the spiral pitch.  Increasing it makes this base interleave more
    open, with fewer turns between the center and ``kmax``; decreasing it
    makes the base interleave wind more tightly.  It is a gradient-design
    parameter, not the number of rotations that the caller must execute.

    ``density`` controls how that nominal count varies with radius.
    ``constant`` keeps one pitch, ``variable`` changes it smoothly according
    to ``variable_density_power``, and ``dual`` joins the inner and outer
    pitches around ``transition_radius``.  ``variant`` chooses whether the
    base gradient runs center-to-edge, edge-to-center, or edge-to-center-to-
    edge.  The caller remains responsible for applying complete-interleave
    rotations and may acquire a number different from ``design_interleaves``.
    """

    def __init__(
        self,
        system,
        fov,
        matrix,
        design_interleaves,
        *,
        variant="outward",
        density="constant",
        inner_design_interleaves=None,
        outer_design_interleaves=None,
        variable_density_power=2.0,
        transition_radius=0.5,
        transition_speed=12.0,
        num_points=1024,
        bandwidth_hz_px=DEFAULT_BANDWIDTH_HZ_PX,
        oversamp=1.0,
        axes=("x", "y"),
        solver_oversampling=8,
        derate=True,
    ):
        fov_m, n = _validate_common(fov, matrix, oversamp, bandwidth_hz_px)
        design_interleaves = int(design_interleaves)
        if design_interleaves < 1:
            raise ValueError("design_interleaves must be >= 1")
        if variant not in _SPIRAL_VARIANTS:
            raise ValueError(f"variant must be one of {_SPIRAL_VARIANTS}, got {variant!r}")
        axes = tuple(axes)
        if len(axes) != 2 or len(set(axes)) != 2 or any(axis not in _AXES for axis in axes):
            raise ValueError("axes must contain two distinct gradient channels")
        if derate:
            apply_system_derates(system)

        factor = 2 if variant == "in_out" else 1
        path_out = spiral_trajectory(
            fov_m,
            n,
            factor * design_interleaves,
            density=density,
            inner_design_interleaves=(None if inner_design_interleaves is None else factor * inner_design_interleaves),
            outer_design_interleaves=(None if outer_design_interleaves is None else factor * outer_design_interleaves),
            variable_density_power=variable_density_power,
            transition_radius=transition_radius,
            transition_speed=transition_speed,
            num_points=num_points,
        )
        grad_out = traj2grad(
            path_out,
            system,
            oversampling=solver_oversampling,
            start_at_zero=True,
            end_at_zero=False,
        )[:, :2]
        n_adc = max(2, int(round(n * oversamp * (2 if variant == "in_out" else 1))))
        grad_out = _stretch_gradient(grad_out, n_adc / bandwidth_hz_px, system.grad_raster_time)
        out_area = np.sum(grad_out, axis=0) * system.grad_raster_time

        if variant == "outward":
            path = path_out
            gradient = grad_out
            pre_area, rew_area = None, -out_area
        elif variant == "inward":
            path = path_out[::-1]
            gradient = -grad_out[::-1]
            pre_area, rew_area = out_area, None
        else:
            path = np.concatenate((-path_out[:0:-1], path_out), axis=0)
            gradient = np.concatenate((grad_out[::-1], grad_out), axis=0)
            pre_area, rew_area = -out_area, -out_area

        first = gradient[0]
        last = gradient[-1]
        gradients = _make_grad_events(system, gradient, axes, first=first, last=last)
        prewinders = _moment_bridges(system, pre_area, np.zeros(2), first, axes) if pre_area is not None else ()
        rewinders = _moment_bridges(system, rew_area, last, np.zeros(2), axes) if rew_area is not None else ()
        read_duration = gradient.shape[0] * system.grad_raster_time
        adc = _make_adc(system, n_adc, read_duration)
        super().__init__(
            system=system,
            gradients=gradients,
            adc=adc,
            trajectory=path,
            design_interleaves=design_interleaves,
            prewinders=prewinders,
            rewinders=rewinders,
            variant=variant,
        )
        self.density = density


class Rosette(NonCartesianGradient):
    """Build one multi-petal rosette base interleave.

    Parameters
    ----------
    system : pypulseq.Opts
        Gradient, slew, and raster limits used to time-parameterize the path.
        Tighter limits lengthen every petal but do not change its k-space
        extent.
    fov : float or array-like
        Isotropic field of view in metres.  Together with ``matrix`` this
        sets ``kmax = matrix / (2 * fov)`` and the largest permitted ADC
        k-space step, ``1 / fov``.  At fixed matrix, a smaller FOV pushes the
        petals farther out and increases gradient demand.
    matrix : int or array-like
        Isotropic reconstruction matrix.  Resolution is ``fov / matrix``;
        increasing the matrix pushes every petal farther into k-space.
    petals : int, optional
        Number of center-to-center radial lobes within this base interleave.
        More petals add k-space center crossings, readout duration, and ADC
        samples.  They are not separately rotated shots.
    angular_frequency_ratio : float, optional
        Ratio ``omega_angular / omega_radial`` within the base interleave.
        Values below one form relatively open petals, one is the circular
        limiting case, and values above one wind more tightly.  This changes
        gradient direction and slew demand; it is unrelated to rotations
        applied to the complete interleave by the caller.
    echo_spacing_s : float, optional
        Requested average center-to-center petal duration.  ``None`` uses the
        minimum duration allowed by the gradient system.  A longer value
        uniformly stretches the waveform and reduces gradient amplitude and
        slew.  A value below the hardware minimum raises ``ValueError``.
        Start/end slew ramps can make the first and last individual crossing
        intervals differ slightly from this average.
    bandwidth_hz_px : float, optional
        Requested receiver bandwidth, with the same ``1 / ADC dwell``
        convention as the other readout classes.  The realized bandwidth is
        raised when necessary to keep adjacent samples close enough for the
        requested FOV.
    oversamp : float, optional
        ADC sampling oversampling factor.  It reduces the permitted k-space
        step without changing ``kmax`` or the gradient shape.
    axes : tuple[str, str], optional
        Gradient channels receiving the two components of the base path.
        This changes physical channel assignment, not k-space geometry.
    solver_oversampling : int, optional
        Internal MRArbGrad accuracy setting.  It affects numerical timing
        fidelity, not the intended trajectory or ADC oversampling.
    derate : bool, optional
        Apply the package gradient/slew derating before design.

    Notes
    -----
    This object owns only one base interleave.  The caller chooses how many
    rotated copies to acquire and supplies their rotation increments.
    """

    def __init__(
        self,
        system,
        fov,
        matrix,
        *,
        petals=5,
        angular_frequency_ratio=3.0 / 5.0,
        echo_spacing_s=None,
        bandwidth_hz_px=DEFAULT_BANDWIDTH_HZ_PX,
        oversamp=1.0,
        axes=("x", "y"),
        solver_oversampling=8,
        derate=True,
    ):
        fov_m, n = _validate_common(fov, matrix, oversamp, bandwidth_hz_px)
        petals = int(petals)
        angular_frequency_ratio = float(angular_frequency_ratio)
        if petals < 1 or not math.isfinite(angular_frequency_ratio) or angular_frequency_ratio <= 0:
            raise ValueError("petals and angular_frequency_ratio must be positive")
        if echo_spacing_s is not None:
            echo_spacing_s = float(echo_spacing_s)
            if not math.isfinite(echo_spacing_s) or echo_spacing_s <= 0:
                raise ValueError("echo_spacing_s must be positive")
        axes = tuple(axes)
        if len(axes) != 2 or len(set(axes)) != 2 or any(axis not in _AXES for axis in axes):
            raise ValueError("axes must contain two distinct gradient channels")
        if derate:
            apply_system_derates(system)
        path = rosette_trajectory(
            fov_m,
            n,
            petals=petals,
            angular_frequency_ratio=angular_frequency_ratio,
            num_points=2049,
        )
        gradient = traj2grad(
            path,
            system,
            oversampling=solver_oversampling,
            start_at_zero=True,
            end_at_zero=True,
        )[:, :2]
        raster = float(system.grad_raster_time)
        min_read_duration = gradient.shape[0] * raster
        min_echo_spacing = min_read_duration / petals
        if echo_spacing_s is not None:
            requested_duration = petals * echo_spacing_s
            if requested_duration < min_read_duration - 1e-12:
                raise ValueError(
                    f"echo_spacing_s={echo_spacing_s:.9g} is shorter than the minimum feasible {min_echo_spacing:.9g} s"
                )
            gradient = _stretch_gradient(gradient, requested_duration, raster)

        # The sampled time-optimal solve can retain a small numerical zeroth
        # moment even though the ideal rosette ends at k=0.  Remove that
        # constant residual without changing slew or requiring a rewinder.
        gradient = gradient - np.mean(gradient, axis=0, keepdims=True)

        read_duration = gradient.shape[0] * raster
        max_k_speed = float(np.max(np.linalg.norm(gradient, axis=1)))
        dwell_from_fov = 1.0 / (float(oversamp) * fov_m * max_k_speed)
        dwell_from_bandwidth = 1.0 / float(bandwidth_hz_px)
        max_dwell = min(dwell_from_fov, dwell_from_bandwidth)
        # Choose the dwell first, on the ADC raster, then fit an integral
        # number of samples into every petal.  Choosing the sample count first
        # and letting ``_make_adc`` round the dwell down can leave an entire
        # tail of the final petal unacquired when the gradient and ADC rasters
        # differ.  The displayed/acquired path must cover the complete petal
        # set requested by the user, not merely the time-optimal gradient.
        dwell = math.floor(max_dwell / system.adc_raster_time + 1e-12) * system.adc_raster_time
        dwell = max(float(system.adc_raster_time), dwell)
        samples_per_petal = max(1, int(math.floor(read_duration / petals / dwell + 1e-12)))
        n_adc = petals * samples_per_petal
        adc = pp.make_adc(num_samples=n_adc, dwell=dwell, system=system)
        acquired_trajectory = _sample_gradient_trajectory(gradient, raster, adc)
        gradients = _make_grad_events(system, gradient, axes, first=np.zeros(2), last=np.zeros(2))
        super().__init__(
            system=system,
            gradients=gradients,
            adc=adc,
            trajectory=acquired_trajectory,
            variant="rosette",
        )
        self.design_trajectory = path
        self.petals = petals
        self.angular_frequency_ratio = angular_frequency_ratio
        self.samples_per_petal = samples_per_petal
        self.min_echo_spacing_s = min_echo_spacing
        self.echo_spacing_s = read_duration / petals
        self.requested_echo_spacing_s = echo_spacing_s
        self.requested_bandwidth_hz_px = float(bandwidth_hz_px)
        self.resolution_m = fov_m / n


def _aligned_gradients(events, alignment, system):
    events = [copy.deepcopy(event) for event in events if event is not None]
    if not events:
        return []
    by_channel = defaultdict(list)
    for event in events:
        by_channel[event.channel].append(event)
    merged = []
    for channel in _AXES:
        group = by_channel.get(channel, [])
        if len(group) == 1:
            merged.append(group[0])
        elif group:
            # Concurrent slice-rephasing and stack-encoding moments can share
            # an axis.  Re-solving their summed area is safer than adding two
            # individually max-slew trapezoids, whose coincident ramps can
            # violate the hardware slew limit.
            area = sum(float(event.area) for event in group)
            first = sum(float(getattr(event, "first", 0.0)) for event in group)
            last = sum(float(getattr(event, "last", 0.0)) for event in group)
            if abs(area) < 1e-12 and abs(first) < 1e-12 and abs(last) < 1e-12:
                continue
            combined, _, _ = pp.make_extended_trapezoid_area(
                area=area,
                channel=channel,
                grad_start=first,
                grad_end=last,
                system=system,
            )
            merged.append(combined)
    return pp.align(**{alignment: merged}) if merged else []


def _scaled_gradient(gradient, scale):
    return pp.scale_grad(copy.deepcopy(gradient), float(scale))


class _NonCartesianTrain(Readout):
    def __init__(
        self,
        readout,
        *,
        num_echoes=1,
        slice_rephasing=None,
        rotation_label="LIN",
        phase_axis=None,
        phase_fov_m=None,
        phase_steps=None,
        phase_label="PAR",
    ):
        if not isinstance(readout, NonCartesianGradient):
            raise TypeError("readout must be a NonCartesianGradient")
        Readout.__init__(self, readout.system)
        if int(num_echoes) < 1:
            raise ValueError("num_echoes must be >= 1")
        self.readout = readout
        self._opts = readout.system
        self.num_echoes = int(num_echoes)
        self.slice_rephasing = slice_rephasing
        self.rotation_label = rotation_label
        self.phase_label = phase_label
        self._phase_axis = phase_axis
        self._phase_template = None
        self._phase_areas = None

        if phase_axis is not None:
            if phase_axis not in _AXES:
                raise ValueError("phase_axis must be x, y, or z")
            if phase_fov_m is None or phase_steps is None or phase_fov_m <= 0 or int(phase_steps) < 1:
                raise ValueError("phase_fov_m must be positive and phase_steps must be >= 1")
            self.phase_steps = int(phase_steps)
            delta = 1.0 / float(phase_fov_m)
            self._phase_areas = (np.arange(self.phase_steps) - 0.5 * self.phase_steps) * delta
            max_area = float(np.max(np.abs(self._phase_areas)))
            if max_area > 0:
                self._phase_template = pp.make_trapezoid(channel=phase_axis, area=max_area, system=self._opts)
            self.phase_fov_m = float(phase_fov_m)

        self.n_samples = readout.n_samples
        phase_pre_candidates = [[]]
        phase_post_candidates = [[]]
        if self._phase_template is not None:
            for area in (float(np.min(self._phase_areas)), float(np.max(self._phase_areas))):
                scale = area / abs(self._phase_template.area)
                phase_pre_candidates.append([_scaled_gradient(self._phase_template, scale)])
                phase_post_candidates.append([_scaled_gradient(self._phase_template, -scale)])

        self.t_prephase_s = max(
            (
                max(
                    (
                        pp.calc_duration(event)
                        for event in _aligned_gradients(
                            [*self.readout.prewinders, *candidate, self.slice_rephasing], "right", self._opts
                        )
                    ),
                    default=0.0,
                )
                for candidate in phase_pre_candidates
            ),
            default=0.0,
        )
        self.t_postphase_s = max(
            (
                max(
                    (
                        pp.calc_duration(event)
                        for event in _aligned_gradients([*self.readout.rewinders, *candidate], "left", self._opts)
                    ),
                    default=0.0,
                )
                for candidate in phase_post_candidates
            ),
            default=0.0,
        )
        self.duration = self.num_echoes * (self.t_prephase_s + readout.read_duration + self.t_postphase_s)

    def _build_blocks(self):
        state = self._require_state()
        return self._collect_blocks(
            lambda seq: self._run(
                seq,
                state.rotation,
                state.rotation_idx,
                state.phase_idx,
                state.adc_phase_rad,
            )
        )

    def _phase_events(self, phase_idx):
        if self._phase_axis is None:
            return [], []
        phase_idx = int(phase_idx)
        if not 0 <= phase_idx < self.phase_steps:
            raise IndexError(f"phase_idx must be in [0, {self.phase_steps}), got {phase_idx}")
        area = float(self._phase_areas[phase_idx])
        if self._phase_template is None or area == 0.0:
            return [], []
        scale = area / abs(self._phase_template.area)
        return [_scaled_gradient(self._phase_template, scale)], [_scaled_gradient(self._phase_template, -scale)]

    def _run(self, seq, rotation, rotation_idx, phase_idx, rf_phase_rad):
        if seq is None:
            from pulserver.pypulseq import Sequence

            seq = Sequence(self._opts)
        rotation = normalize_rotation(rotation)
        phase_pre, phase_post = self._phase_events(phase_idx)

        for echo in range(self.num_echoes):
            pre_extra = list(phase_pre)
            if echo == 0 and self.slice_rephasing is not None:
                pre_extra.append(self.slice_rephasing)
            pre = _aligned_gradients([*self.readout.prewinders, *pre_extra], "right", self._opts)
            if self.t_prephase_s > 0.0:
                pre_duration = max((pp.calc_duration(event) for event in pre), default=0.0)
                shift = self.t_prephase_s - pre_duration
                if shift > 0.0:
                    for event in pre:
                        event.delay += shift
                seq.add_block(
                    *pre,
                    pp.make_delay(self.t_prephase_s),
                    *([rotation] if rotation is not None else []),
                )

            adc = copy.deepcopy(self.readout.adc)
            adc.phase_offset = float(rf_phase_rad)
            labels = []
            if echo == 0:
                labels.append(pp.make_label(type="SET", label=self.rotation_label, value=int(rotation_idx)))
                if self._phase_axis is not None:
                    labels.append(pp.make_label(type="SET", label=self.phase_label, value=int(phase_idx)))
            if self.num_echoes > 1:
                labels.append(
                    pp.make_label(type="SET", label="ECO", value=0)
                    if echo == 0
                    else pp.make_label(type="INC", label="ECO", value=1)
                )
            seq.add_block(
                *self.readout.gradients,
                adc,
                *labels,
                *([rotation] if rotation is not None else []),
            )

            post = _aligned_gradients([*self.readout.rewinders, *phase_post], "left", self._opts)
            if self.t_postphase_s > 0.0:
                seq.add_block(
                    *post,
                    pp.make_delay(self.t_postphase_s),
                    *([rotation] if rotation is not None else []),
                )
        return seq


class NonCartesian2D(_NonCartesianTrain):
    """Planar readout; optional slice rephasing is right-aligned to its prewinder."""

    def __init__(self, readout, *, slice_rephasing=None, num_echoes=1, rotation_label="LIN"):
        super().__init__(
            readout,
            slice_rephasing=slice_rephasing,
            num_echoes=num_echoes,
            rotation_label=rotation_label,
        )

    def set_state(self, lin_idx=0, adc_phase_rad=0.0, rotation=None):
        """Set rotation label, receiver phase, and rotation for one shot."""
        self._replace_state(
            _NonCartesianState(
                rotation=normalize_rotation(rotation),
                rotation_idx=int(lin_idx),
                phase_idx=None,
                adc_phase_rad=float(adc_phase_rad),
            )
        )
        return self

    def __call__(self, seq=None, rotation=None, rotation_idx=0, rf_phase_rad=0.0, *, lin_idx=None):
        if lin_idx is not None:
            rotation_idx = lin_idx
        self.set_state(
            lin_idx=rotation_idx,
            adc_phase_rad=rf_phase_rad,
            rotation=rotation,
        )
        return self._add_or_get(seq)


class StackOfTrajectories(_NonCartesianTrain):
    """Promote a 2D readout to 3D with Cartesian phase encoding."""

    def __init__(
        self,
        readout,
        fov_z_m,
        nz,
        *,
        phase_axis="z",
        slice_rephasing=None,
        num_echoes=1,
        rotation_label="LIN",
        phase_label="PAR",
    ):
        super().__init__(
            readout,
            slice_rephasing=slice_rephasing,
            num_echoes=num_echoes,
            rotation_label=rotation_label,
            phase_axis=phase_axis,
            phase_fov_m=fov_z_m,
            phase_steps=nz,
            phase_label=phase_label,
        )
        self.nz = int(nz)

    def set_state(self, lin_idx=0, par_idx=0, adc_phase_rad=0.0, rotation=None):
        """Set rotation/partition labels, receiver phase, and rotation."""
        self._replace_state(
            _NonCartesianState(
                rotation=normalize_rotation(rotation),
                rotation_idx=int(lin_idx),
                phase_idx=int(par_idx),
                adc_phase_rad=float(adc_phase_rad),
            )
        )
        return self

    def __call__(
        self,
        seq=None,
        rotation=None,
        rotation_idx=0,
        phase_idx=0,
        rf_phase_rad=0.0,
        *,
        lin_idx=None,
        par_idx=None,
    ):
        if lin_idx is not None:
            rotation_idx = lin_idx
        if par_idx is not None:
            phase_idx = par_idx
        self.set_state(
            lin_idx=rotation_idx,
            par_idx=phase_idx,
            adc_phase_rad=rf_phase_rad,
            rotation=rotation,
        )
        return self._add_or_get(seq)


class Projection(NonCartesian2D):
    """A planar readout whose arbitrary 3D rotations form projection coverage."""


class NonCartesian3D(NonCartesian2D):
    """Native 3D trajectory readout; the canonical shot may contain x/y/z gradients."""

    def __init__(self, readout, *, num_echoes=1, rotation_label="LIN"):
        if readout.trajectory.ndim != 2 or readout.trajectory.shape[1] != 3:
            raise ValueError("NonCartesian3D requires a trajectory with three columns")
        super().__init__(readout, num_echoes=num_echoes, rotation_label=rotation_label)


# Descriptive aliases retained alongside the concise class names.
NonCartesianStack3D = StackOfTrajectories
NonCartesianProjection = Projection
