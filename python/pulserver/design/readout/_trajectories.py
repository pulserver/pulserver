"""Non-Cartesian base interleaves: one shot's worth of designed gradient.

A trajectory object here owns the *waveform*, not the acquisition: the
gradients that trace one canonical interleave, the ADC that samples it, and
the bridges that reach k-space and come back. Where that interleave is played,
and how many rotated copies of it a scan acquires, belongs to the readout
module that plays it and to the loop above that.
"""

from __future__ import annotations

__all__ = [
    "Arbitrary",
    "NonCartesianGradient",
    "Radial",
    "Rosette",
    "Spiral",
    "radial_trajectory",
    "rosette_trajectory",
    "spiral_trajectory",
]

import copy
import math

import numpy as np

from ... import pypulseq as pp
from ._common import AXES as _AXES
from ._common import DEFAULT_BANDWIDTH_HZ as DEFAULT_BANDWIDTH_HZ_PX

_SPIRAL_DIRECTIONS = ("outward", "inward", "in_out")
_SPIRAL_DENSITIES = ("constant", "variable", "dual")


def traj2grad(trajectory, system, *, oversampling=8, start_at_zero=True, end_at_zero=True):
    """The gradient tracing ``trajectory``, shaped ``(n_grad, 3)``.

    Reparameterises the path against the vector gradient and slew limits, so
    the samples handed in describe geometry rather than time.

    Parameters
    ----------
    trajectory : numpy.ndarray
        K-space path, ``(n, 2)`` or ``(n, 3)``, in 1/m.
    system : pypulseq.Opts
        System limits.
    oversampling : int, optional
        Path-resampling factor the solver works at.
    start_at_zero, end_at_zero : bool, optional
        Ramp up from and back down to zero amplitude. Disable an endpoint when
        a bridge will run straight into the readout instead.

    Returns
    -------
    numpy.ndarray
        Gradient of shape ``(n_grad, 3)`` in Hz/m, on the gradient raster.
    """
    trajectory = np.atleast_2d(np.asarray(trajectory, dtype=float))
    gradient, _ = pp.traj_to_grad(
        trajectory.T,
        system=system,
        oversampling=oversampling,
        start_at_zero=start_at_zero,
        end_at_zero=end_at_zero,
    )
    gradient = np.atleast_2d(gradient).T
    if gradient.shape[1] < 3:
        gradient = np.hstack([gradient, np.zeros((gradient.shape[0], 3 - gradient.shape[1]))])
    return gradient


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
    new_n = max(old_n, math.ceil(target_duration / raster - 1e-12))
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
    """One bridge per axis, solved so that together they stay inside the limits.

    Each bridge is solved on its own, because each has its own moment to
    deliver -- but they are *played together*, and the limits are on the
    vector. Two axes each solved against the full slew can combine to root-two
    times it, which no per-axis check sees and which the scanner does. Under a
    rotation it is not even hidden: the extension mixes the axes, so the
    combined slew turns up on a single one.

    So the slew is derated by the square root of the number of axes being
    bridged at once, which is the bound that makes any combination of them
    legal. The amplitude is *not*: a bridge's endpoints are the readout's own,
    fixed by a waveform already solved against the vector limit, and a derated
    ceiling would simply refuse them.
    """
    active = [
        index
        for index in range(len(axes))
        if abs(area[index]) >= 1e-12 or abs(grad_start[index]) >= 1e-12 or abs(grad_end[index]) >= 1e-12
    ]
    if not active:
        return ()

    limits = system
    if len(active) > 1:
        limits = copy.copy(system)
        limits.max_slew = system.max_slew / math.sqrt(len(active))

    events = []
    for index in active:
        grad, _, _ = pp.make_extended_trapezoid_area(
            area=float(area[index]),
            channel=axes[index],
            grad_start=float(grad_start[index]),
            grad_end=float(grad_end[index]),
            system=limits,
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
        kind=None,
    ):
        self.system = system
        self.gradients = tuple(gradients)
        self.adc = adc
        self.trajectory = np.asarray(trajectory, dtype=float)
        self.design_interleaves = None if design_interleaves is None else int(design_interleaves)
        self.recommended_rotations = None if recommended_rotations is None else int(recommended_rotations)
        self.prewinders = tuple(prewinders)
        self.rewinders = tuple(rewinders)
        self.kind = kind
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

    @property
    def axes(self) -> tuple[str, ...]:
        """The gradient channels this interleave drives, in waveform order."""
        return tuple(gradient.channel for gradient in self.gradients)

    def rotated(self, angle: float) -> "NonCartesianGradient":
        """This interleave turned by ``angle`` radians in its own plane.

        A rotation is normally left to the Pulseq rotation extension, which
        costs one quaternion per shot instead of one waveform. This is for the
        case that cannot express: an interpreter without the extension, or a
        set of interleaves the caller wants written out in full.

        The waveform samples are rotated directly rather than the path
        re-solved, so the result is the same gradient seen from a turned frame
        -- identical duration, identical slew, exactly the intended geometry.
        The prewinder and rewinder are re-solved, because they are one bridge
        per axis and their split between the axes changes with the angle.

        Parameters
        ----------
        angle : float
            In-plane rotation (rad).

        Returns
        -------
        NonCartesianGradient
            A new bundle sharing this one's ADC.

        Raises
        ------
        ValueError
            If the interleave is not planar in two channels.
        """
        axes = self.axes
        if len(axes) != 2:
            raise ValueError("only a two-channel planar interleave can be rotated in its own plane")

        waveforms = np.column_stack([np.asarray(g.waveform, dtype=float) for g in self.gradients])
        turn = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float
        )
        rotated = waveforms @ turn.T
        first, last = rotated[0], rotated[-1]

        moments = np.array([_bridge_area(g, axes) for g in self.prewinders]).sum(axis=0)
        pre_area = moments @ turn.T if self.prewinders else None
        moments = np.array([_bridge_area(g, axes) for g in self.rewinders]).sum(axis=0)
        rew_area = moments @ turn.T if self.rewinders else None

        turned = object.__new__(type(self))
        NonCartesianGradient.__init__(
            turned,
            system=self.system,
            gradients=_make_grad_events(self.system, rotated, axes, first=first, last=last),
            adc=self.adc,
            trajectory=self.trajectory[:, :2] @ turn.T,
            design_interleaves=self.design_interleaves,
            recommended_rotations=self.recommended_rotations,
            prewinders=(
                ()
                if pre_area is None
                else _moment_bridges(self.system, pre_area, np.zeros(2), first, axes)
            ),
            rewinders=(
                ()
                if rew_area is None
                else _moment_bridges(self.system, rew_area, last, np.zeros(2), axes)
            ),
            kind=self.kind,
        )
        for name in ("direction", "density"):
            if hasattr(self, name):
                setattr(turned, name, getattr(self, name))
        return turned


def _bridge_area(event, axes) -> np.ndarray:
    """One bridge's moment, placed on the axis it drives."""
    moment = np.zeros(len(axes))
    moment[axes.index(event.channel)] = float(np.trapezoid(event.waveform, event.tt))
    return moment


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
            pp.apply_system_derates(system)

        k_start = path[0]
        has_pre = not np.allclose(k_start, 0.0, atol=1e-12)
        has_rew = not np.allclose(path[-1], 0.0, atol=1e-12)
        n_adc = max(2, round(n * oversamp))
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
            kind="arbitrary",
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
            pp.apply_system_derates(system)
        raster = system.grad_raster_time
        kmax = n / (2.0 * fov_m)
        n_adc = max(2, round(n * oversamp))
        target_duration = n_adc / float(bandwidth_hz_px)
        read_duration = pp.ceil_to_raster(max(target_duration, 2.0 * kmax / system.max_grad), raster)
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
            recommended_rotations=math.ceil(np.pi * n / 2.0),
            prewinders=(pre,),
            rewinders=(rew,),
            kind="full",
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
    pitches around ``transition_radius``.  ``direction`` chooses whether the
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
        direction="outward",
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
        if direction not in _SPIRAL_DIRECTIONS:
            raise ValueError(f"direction must be one of {_SPIRAL_DIRECTIONS}, got {direction!r}")
        axes = tuple(axes)
        if len(axes) != 2 or len(set(axes)) != 2 or any(axis not in _AXES for axis in axes):
            raise ValueError("axes must contain two distinct gradient channels")
        if derate:
            pp.apply_system_derates(system)

        factor = 2 if direction == "in_out" else 1
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
        n_adc = max(2, round(n * oversamp * (2 if direction == "in_out" else 1)))
        grad_out = _stretch_gradient(grad_out, n_adc / bandwidth_hz_px, system.grad_raster_time)
        out_area = np.sum(grad_out, axis=0) * system.grad_raster_time

        if direction == "outward":
            path = path_out
            gradient = grad_out
            pre_area, rew_area = None, -out_area
        elif direction == "inward":
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
            kind="spiral",
        )
        self.direction = direction
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
            pp.apply_system_derates(system)
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
        samples_per_petal = max(1, math.floor(read_duration / petals / dwell + 1e-12))
        n_adc = petals * samples_per_petal
        adc = pp.make_adc(num_samples=n_adc, dwell=dwell, system=system)
        acquired_trajectory = _sample_gradient_trajectory(gradient, raster, adc)
        gradients = _make_grad_events(system, gradient, axes, first=np.zeros(2), last=np.zeros(2))
        super().__init__(
            system=system,
            gradients=gradients,
            adc=adc,
            trajectory=acquired_trajectory,
            kind="rosette",
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


