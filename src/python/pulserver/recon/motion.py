"""Rigid image registration and extended-Kalman motion tracking."""

from __future__ import annotations

__all__ = [
    "NavigatorMotionTracker",
    "RigidMotionEKF",
    "RigidMotionEstimate",
    "RigidRegistration",
]

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RigidMotionEstimate:
    """One 2D or 3D rigid transform in SimpleITK physical coordinates.

    Parameters
    ----------
    parameters
        ``(angle, tx, ty)`` in 2D or
        ``(angle_x, angle_y, angle_z, tx, ty, tz)`` in 3D. Angles are radians
        and translations use the image-spacing unit, conventionally mm.
    center
        Physical rotation center in SimpleITK ``(x, y[, z])`` order.
    covariance
        Optional pose covariance in the same parameter order.
    metric
        Final registration metric value, when available.
    stop_condition
        SimpleITK optimizer stop description, when available.

    Examples
    --------
    Six parameters -- three rotations and three translations -- about a centre.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> estimate = recon.RigidMotionEstimate(
    ...     parameters=torch.zeros(6), center=torch.zeros(3)
    ... )
    >>> tuple(estimate.parameters.shape), estimate.metric is None
    ((6,), True)
    """

    parameters: np.ndarray
    center: np.ndarray
    covariance: np.ndarray | None = None
    metric: float | None = None
    stop_condition: str | None = None

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters, dtype=np.float64).reshape(-1)
        if parameters.size not in {3, 6}:
            raise ValueError("rigid motion requires three 2D or six 3D parameters")
        dimension = 2 if parameters.size == 3 else 3
        center = np.asarray(self.center, dtype=np.float64).reshape(-1)
        if center.size != dimension:
            raise ValueError("center dimensionality does not match parameters")
        covariance = self.covariance
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=np.float64)
            if covariance.shape != (parameters.size, parameters.size):
                raise ValueError("covariance shape does not match motion parameters")
        object.__setattr__(self, "parameters", parameters.copy())
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(
            self,
            "covariance",
            None if covariance is None else covariance.copy(),
        )

    @property
    def dimension(self) -> int:
        """Return two or three spatial dimensions."""
        return 2 if self.parameters.size == 3 else 3

    @property
    def angles(self) -> np.ndarray:
        """Return Euler angles in radians."""
        count = 1 if self.dimension == 2 else 3
        return self.parameters[:count].copy()

    @property
    def translation(self) -> np.ndarray:
        """Return translation in physical-coordinate order."""
        count = 1 if self.dimension == 2 else 3
        return self.parameters[count:].copy()

    @property
    def matrix(self) -> np.ndarray:
        """Return the centered transform as a homogeneous matrix."""
        transform = _transform_from_estimate(self)
        rotation = np.asarray(transform.GetMatrix()).reshape(
            self.dimension,
            self.dimension,
        )
        offset = self.center + self.translation - rotation @ self.center
        matrix = np.eye(self.dimension + 1)
        matrix[: self.dimension, : self.dimension] = rotation
        matrix[: self.dimension, self.dimension] = offset
        return matrix


class RigidRegistration:
    """Fast multi-resolution SimpleITK rigid registration for MRI magnitude.

    Parameters
    ----------
    metric
        ``"correlation"``, ``"mutual_information"``, or ``"mean_squares"``.
    iterations
        Optimizer iterations at each multi-resolution execution.
    sampling_percentage
        Random metric-sampling fraction. One uses every voxel.
    shrink_factors, smoothing_sigmas
        Coarse-to-fine SimpleITK pyramid configuration.
    learning_rate, min_step
        Regular-step gradient-descent controls.
    threads
        How many threads the registration may use. ``None`` leaves ITK's own
        default, which is one per core. ITK threads the metric through a
        process-wide setting rather than a per-registration one, so this is
        applied around the call and restored after it. Threading is a loss on
        a small image -- dispatching twenty threads over a navigator plane
        costs several times what evaluating the metric does, and it makes the
        worst case several times the median, which is what a real-time
        deadline is actually made of.

    Examples
    --------
    Eight degrees and a few millimetres, measured and undone. The estimate is
    a transform in SimpleITK's physical ``(x, y)`` order, so resampling with
    an array-indexed tool means swapping its axes:

    .. plot::

       import numpy as np
       from scipy.ndimage import affine_transform, rotate, shift
       import pulserver.recon as recon
       from _figures import images, phantom

       truth = phantom(64)[0][0].numpy().real.astype(np.float32)
       moved = shift(rotate(truth, 8.0, reshape=False, order=1), (3.0, -4.0), order=1)

       estimate = recon.RigidRegistration(iterations=100)(truth, moved)
       swap = np.array([[0.0, 1.0], [1.0, 0.0]])
       matrix = np.asarray(estimate.matrix)
       registered = affine_transform(
           moved, swap @ matrix[:2, :2] @ swap, offset=swap @ matrix[:2, 2], order=1
       )
       images(
           [("object", truth), ("moved", moved), ("registered", registered)],
           title=f"rigid registration: {np.rad2deg(estimate.angles)[0]:.1f} degrees recovered",
       )
    """

    def __init__(
        self,
        *,
        metric: str = "correlation",
        iterations: int = 50,
        sampling_percentage: float = 0.2,
        shrink_factors: Sequence[int] = (4, 2, 1),
        smoothing_sigmas: Sequence[float] = (2.0, 1.0, 0.0),
        learning_rate: float = 1.0,
        min_step: float = 1e-4,
        threads: int | None = None,
    ) -> None:
        if metric not in {"correlation", "mutual_information", "mean_squares"}:
            raise ValueError("unsupported registration metric")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if not 0.0 < sampling_percentage <= 1.0:
            raise ValueError("sampling_percentage must lie in (0, 1]")
        shrink = tuple(int(value) for value in shrink_factors)
        smoothing = tuple(float(value) for value in smoothing_sigmas)
        if not shrink or len(shrink) != len(smoothing):
            raise ValueError("pyramid factors and sigmas must have equal length")
        if any(value < 1 for value in shrink) or any(value < 0 for value in smoothing):
            raise ValueError("invalid multi-resolution pyramid")
        if learning_rate <= 0.0 or min_step <= 0.0:
            raise ValueError("optimizer steps must be positive")
        if threads is not None and int(threads) < 1:
            raise ValueError("threads must be at least one")
        self.threads = None if threads is None else int(threads)
        self.metric = metric
        self.iterations = int(iterations)
        self.sampling_percentage = float(sampling_percentage)
        self.shrink_factors = shrink
        self.smoothing_sigmas = smoothing
        self.learning_rate = float(learning_rate)
        self.min_step = float(min_step)

    def estimate(
        self,
        fixed: Any,
        moving: Any,
        *,
        initial: RigidMotionEstimate | None = None,
        spacing: Sequence[float] | None = None,
    ) -> RigidMotionEstimate:
        """Register ``moving`` to ``fixed`` and return the rigid transform."""
        sitk = _simpleitk()
        fixed_image = _sitk_image(fixed, spacing)
        moving_image = _sitk_image(moving, spacing)
        if fixed_image.GetDimension() != moving_image.GetDimension():
            raise ValueError("fixed and moving images must have equal dimensionality")
        dimension = fixed_image.GetDimension()
        transform = _centered_transform(fixed_image, moving_image)
        if initial is not None:
            if initial.dimension != dimension:
                raise ValueError(
                    "initial transform dimensionality does not match images"
                )
            transform.SetParameters(tuple(map(float, initial.parameters)))

        registration = sitk.ImageRegistrationMethod()
        if self.metric == "correlation":
            registration.SetMetricAsCorrelation()
        elif self.metric == "mutual_information":
            registration.SetMetricAsMattesMutualInformation(32)
        else:
            registration.SetMetricAsMeanSquares()
        if self.sampling_percentage < 1.0:
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            registration.SetMetricSamplingPercentage(
                self.sampling_percentage,
                seed=42,
            )
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetShrinkFactorsPerLevel(self.shrink_factors)
        registration.SetSmoothingSigmasPerLevel(self.smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=self.learning_rate,
            minStep=self.min_step,
            numberOfIterations=self.iterations,
            gradientMagnitudeTolerance=1e-8,
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetInitialTransform(transform, inPlace=True)
        with _threads(self.threads):
            result = registration.Execute(fixed_image, moving_image)
        result = _rigid_component(result)
        return RigidMotionEstimate(
            parameters=np.asarray(result.GetParameters()),
            center=np.asarray(result.GetCenter()),
            metric=float(registration.GetMetricValue()),
            stop_condition=registration.GetOptimizerStopConditionDescription(),
        )

    def resample(
        self,
        moving: Any,
        estimate: RigidMotionEstimate,
        *,
        reference: Any | None = None,
        spacing: Sequence[float] | None = None,
        default_value: float = 0.0,
    ) -> Any:
        """Resample ``moving`` onto ``reference`` using an estimated transform."""
        sitk = _simpleitk()
        moving_image = _sitk_image(moving, spacing, normalize=False)
        reference_image = (
            moving_image
            if reference is None
            else _sitk_image(reference, spacing, normalize=False)
        )
        if estimate.dimension != moving_image.GetDimension():
            raise ValueError("motion dimensionality does not match the moving image")
        output = sitk.Resample(
            moving_image,
            reference_image,
            _transform_from_estimate(estimate),
            sitk.sitkLinear,
            float(default_value),
            sitk.sitkFloat32,
        )
        return _restore_array(sitk.GetArrayFromImage(output), moving)

    def __call__(self, fixed: Any, moving: Any, **kwargs: Any) -> RigidMotionEstimate:
        return self.estimate(fixed, moving, **kwargs)


class RigidMotionEKF:
    """Constant-velocity extended Kalman filter for registered rigid motion.

    SimpleITK supplies the nonlinear image measurement; the filter uses a
    constant angular/translational velocity state and wraps Euler-angle
    innovations. This keeps registration initialization close to the expected
    pose and suppresses frame-to-frame jitter for prospective motion control.

    Parameters
    ----------
    registration
        Rigid image-registration measurement model.
    dimension
        Two or three spatial dimensions.
    process_noise
        Scalar or per-pose-coordinate acceleration variance.
    measurement_noise
        Scalar or per-pose-coordinate registration variance.
    initial_covariance
        Initial pose/velocity covariance scale.

    Examples
    --------
    Tracks rigid motion across a scan rather than registering each volume
    independently: the filter carries the estimate forward, so a noisy
    registration is smoothed by what came before it::

        import pulserver.recon as recon

        tracker = recon.RigidMotionEKF(recon.RigidRegistration(), dimension=3)
        estimate = tracker.step(reference, volume)

    A head that turns and settles, watched by a navigator every 100 ms. Each
    navigator is registered on its own and carries the error of one
    registration; the filter knows the head has a velocity, so it separates
    the motion from the measurement rather than following every wobble into
    the gradients:

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       import pulserver.recon as recon
       from _figures import SERIES

       interval, count = 0.1, 60
       elapsed = np.arange(count) * interval
       turn = np.tanh((elapsed - 2.5) * 1.2)
       truth = np.zeros((count, 6))
       truth[:, 0], truth[:, 5] = np.deg2rad(4.0) * turn, 3.0 * turn

       # What one registration of one navigator is worth: a third of a degree
       # and a third of a millimetre, independently each time.
       angle, shift = np.deg2rad(0.3), 0.3
       error = np.random.default_rng(4).normal(size=(count, 6))
       measured = truth + error * np.array([angle] * 3 + [shift] * 3)

       tracker = recon.RigidMotionEKF(
           recon.RigidRegistration(),
           measurement_noise=(angle**2,) * 3 + (shift**2,) * 3,
           process_noise=(1e-2,) * 3 + (1e2,) * 3,
       )
       filtered = []
       for pose in measured:
           tracker.predict(interval)
           filtered.append(
               tracker.update(
                   recon.RigidMotionEstimate(parameters=pose, center=np.zeros(3))
               ).parameters
           )
       filtered = np.asarray(filtered)

       figure, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
       panels = (
           (0, np.rad2deg, "nod about x [deg]"),
           (5, lambda value: value, "slide along z [mm]"),
       )
       for axis, (column, scale, label) in zip(axes, panels):
           axis.plot(elapsed, scale(measured[:, column]), ".", color=SERIES[1],
                     markersize=5, label="registered")
           axis.plot(elapsed, scale(truth[:, column]), color="0.55",
                     linewidth=2.5, label="where the head was")
           axis.plot(elapsed, scale(filtered[:, column]), color=SERIES[0],
                     linewidth=1.8, label="filtered")
           axis.set_ylabel(label)
       axes[1].set_xlabel("time [s]")
       axes[0].legend(frameon=False, loc="lower right", fontsize=8)
       figure.suptitle("one navigator every 100 ms, filtered")
       figure.tight_layout()
    """

    def __init__(
        self,
        registration: RigidRegistration,
        *,
        dimension: int = 3,
        process_noise: float | Sequence[float] = 1e-4,
        measurement_noise: float | Sequence[float] = 1e-2,
        initial_covariance: float = 1.0,
    ) -> None:
        if dimension not in {2, 3}:
            raise ValueError("dimension must be two or three")
        if initial_covariance <= 0.0:
            raise ValueError("initial_covariance must be positive")
        self.registration = registration
        self.dimension = int(dimension)
        self.dof = 3 if dimension == 2 else 6
        self.rotation_count = 1 if dimension == 2 else 3
        self.process_noise = _noise_vector(process_noise, self.dof, "process_noise")
        self.measurement_noise = _noise_vector(
            measurement_noise,
            self.dof,
            "measurement_noise",
        )
        self.initial_covariance = float(initial_covariance)
        self.state = np.zeros(2 * self.dof, dtype=np.float64)
        self.covariance = np.eye(2 * self.dof) * self.initial_covariance
        self.center = np.zeros(dimension, dtype=np.float64)
        self.initialized = False
        self.last_measurement: RigidMotionEstimate | None = None

    @property
    def pose(self) -> RigidMotionEstimate:
        """Return the current filtered pose estimate."""
        return RigidMotionEstimate(
            parameters=self.state[: self.dof],
            center=self.center,
            covariance=self.covariance[: self.dof, : self.dof],
            metric=(
                None if self.last_measurement is None else self.last_measurement.metric
            ),
            stop_condition=(
                None
                if self.last_measurement is None
                else self.last_measurement.stop_condition
            ),
        )

    @property
    def velocity(self) -> np.ndarray:
        """Return angular and translational velocity in pose order."""
        return self.state[self.dof :].copy()

    def reset(self, initial: RigidMotionEstimate | None = None) -> None:
        """Reset covariance and optionally initialize the filter pose."""
        self.state.fill(0.0)
        self.covariance = np.eye(2 * self.dof) * self.initial_covariance
        self.center.fill(0.0)
        self.initialized = False
        self.last_measurement = None
        if initial is not None:
            self._initialize(initial)

    def predict(self, dt: float = 1.0) -> RigidMotionEstimate:
        """Propagate the constant-velocity state by ``dt``."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        transition = np.eye(2 * self.dof)
        transition[: self.dof, self.dof :] = dt * np.eye(self.dof)
        noise = np.zeros_like(self.covariance)
        diagonal = np.diag(self.process_noise)
        noise[: self.dof, : self.dof] = 0.25 * dt**4 * diagonal
        noise[: self.dof, self.dof :] = 0.5 * dt**3 * diagonal
        noise[self.dof :, : self.dof] = 0.5 * dt**3 * diagonal
        noise[self.dof :, self.dof :] = dt**2 * diagonal
        self.state = transition @ self.state
        self.state[: self.rotation_count] = _wrap_angles(
            self.state[: self.rotation_count]
        )
        self.covariance = transition @ self.covariance @ transition.T + noise
        return self.pose

    def update(
        self,
        measurement: RigidMotionEstimate | Sequence[float] | np.ndarray,
        *,
        covariance: Any | None = None,
    ) -> RigidMotionEstimate:
        """Fuse a registered pose measurement into the predicted state."""
        if not isinstance(measurement, RigidMotionEstimate):
            measurement = RigidMotionEstimate(
                parameters=np.asarray(measurement),
                center=self.center,
            )
        if measurement.dimension != self.dimension:
            raise ValueError("measurement dimensionality does not match the filter")
        if not self.initialized:
            self._initialize(measurement)
            return self.pose
        observed = measurement.parameters
        observation = np.zeros((self.dof, 2 * self.dof))
        observation[:, : self.dof] = np.eye(self.dof)
        innovation = observed - observation @ self.state
        innovation[: self.rotation_count] = _wrap_angles(
            innovation[: self.rotation_count]
        )
        selected_covariance = covariance
        if selected_covariance is None:
            selected_covariance = measurement.covariance
        if selected_covariance is None:
            selected_covariance = np.diag(self.measurement_noise)
        selected_covariance = np.asarray(selected_covariance, dtype=np.float64)
        if selected_covariance.shape != (self.dof, self.dof):
            raise ValueError("measurement covariance has the wrong shape")
        innovation_covariance = (
            observation @ self.covariance @ observation.T + selected_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance,
            observation @ self.covariance,
        ).T
        self.state = self.state + gain @ innovation
        self.state[: self.rotation_count] = _wrap_angles(
            self.state[: self.rotation_count]
        )
        identity = np.eye(2 * self.dof)
        residual = identity - gain @ observation
        self.covariance = (
            residual @ self.covariance @ residual.T
            + gain @ selected_covariance @ gain.T
        )
        self.center = measurement.center.copy()
        self.last_measurement = measurement
        return self.pose

    def step(
        self,
        fixed: Any,
        moving: Any,
        *,
        dt: float = 1.0,
        spacing: Sequence[float] | None = None,
    ) -> RigidMotionEstimate:
        """Predict, register the next frame, and update the filtered pose."""
        initial = self.predict(dt) if self.initialized else None
        measurement = self.registration.estimate(
            fixed,
            moving,
            initial=initial,
            spacing=spacing,
        )
        return self.update(measurement)

    # %% private module subroutines

    def _initialize(self, measurement: RigidMotionEstimate) -> None:
        self.state.fill(0.0)
        self.state[: self.dof] = measurement.parameters
        self.center = measurement.center.copy()
        self.covariance = np.eye(2 * self.dof) * self.initial_covariance
        if measurement.covariance is not None:
            self.covariance[: self.dof, : self.dof] = measurement.covariance
        self.last_measurement = measurement
        self.initialized = True


class NavigatorMotionTracker:
    """Rigid pose from a navigator's 2D planes, filtered across the scan.

    A navigator resolves six degrees of freedom out of images that are each
    only two-dimensional. One plane sees the in-plane part of the motion --
    rotation about its own normal, translation along its own two axes -- and
    planes that between them span the volume see all six. So a navigator is
    registered plane by plane against the first one acquired, the plane
    measurements are solved together for the pose that explains all of them,
    and a :class:`RigidMotionEKF` carries that pose across the scan.

    Three orthogonal planes are the usual set and the best conditioned one,
    but nothing here requires three, or requires them orthogonal: what it
    requires is that the plane normals span the rotations and the in-plane
    axes span the translations. Planes are matched to the reference by
    position, so a navigator must present them in the same order every time.

    What a plane measures is an in-plane rotation, so the pose is solved as a
    rotation vector and only then turned into Euler angles, which is exact.
    The approximation is upstream of that, and it is the one 2D registration
    of a plane rests on anyway: a plane only measures the motion it can see,
    so the estimate holds while the out-of-plane rotation is small enough
    that each plane still cuts the anatomy its reference cut.

    Parameters
    ----------
    registration
        The rigid registration each plane is measured with. The default is a
        :class:`RigidRegistration` on its own defaults.
    process_noise, measurement_noise, initial_covariance
        Passed to the :class:`RigidMotionEKF` this builds.

    Attributes
    ----------
    filter : RigidMotionEKF
        The filter carrying the pose. ``filter.velocity`` is the tracked
        angular and translational velocity.
    reference : list or None
        The planes of the first navigator, which every later one is
        registered against. ``None`` until the first :meth:`track`.

    Examples
    --------
    A navigator's planes and the axes each was sampled along, tracked across
    a scan::

        import pulserver.recon as recon

        tracker = recon.NavigatorMotionTracker()
        pose = tracker.track(planes, axes, dt=0.1, spacing=10.0)

    The first navigator is the reference and comes back as the zero pose;
    every later one comes back as the head's pose relative to it.

    >>> import numpy as np
    >>> import pulserver.recon as recon
    >>> tracker = recon.NavigatorMotionTracker()
    >>> planes = [np.zeros((8, 8)) for _ in range(3)]
    >>> axes = [
    ...     ((1, 0, 0), (0, 1, 0)),
    ...     ((0, 1, 0), (0, 0, 1)),
    ...     ((1, 0, 0), (0, 0, 1)),
    ... ]
    >>> pose = tracker.track(planes, axes)
    >>> pose.dimension, np.allclose(pose.parameters, 0.0)
    (3, True)
    """

    def __init__(
        self,
        *,
        registration: RigidRegistration | None = None,
        process_noise: float | Sequence[float] = 1e-4,
        measurement_noise: float | Sequence[float] = 1e-2,
        initial_covariance: float = 1.0,
    ) -> None:
        self.filter = RigidMotionEKF(
            RigidRegistration() if registration is None else registration,
            dimension=3,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_covariance=initial_covariance,
        )
        self.reference: list[np.ndarray] | None = None

    @property
    def registration(self) -> RigidRegistration:
        """Return the registration each plane is measured with."""
        return self.filter.registration

    @property
    def pose(self) -> RigidMotionEstimate:
        """Return the current filtered pose, about the navigator's centre."""
        return self.filter.pose

    def reset(self) -> None:
        """Forget the reference navigator and the filtered pose."""
        self.filter.reset()
        self.reference = None

    def track(
        self,
        planes: Sequence[Any],
        axes: Sequence[Sequence[Sequence[float]]],
        *,
        dt: float = 1.0,
        spacing: float | Sequence[float] = 1.0,
    ) -> RigidMotionEstimate:
        """Measure one navigator against the reference and filter its pose.

        Parameters
        ----------
        planes
            This navigator's images, one 2D array per plane.
        axes
            Where each plane lies, as ``(row_axis, column_axis)`` unit vectors
            in the frame the pose is wanted in -- the directions the image's
            first and second axis run along. Their cross product is the plane
            normal, and the three of them are what carries a plane's in-plane
            measurement out into the volume.
        dt
            Seconds since the previous navigator, which is what the filter
            propagates its velocity over.
        spacing
            Pixel size of the planes, one value or one per plane, isotropic in
            plane. The pose's translation comes back in this unit, and
            millimetres are the practical choice: the registration steps and
            stops on physical lengths, and its defaults are lengths a
            millimetre-scale image is measured in.

        Returns
        -------
        RigidMotionEstimate
            The filtered pose relative to the reference navigator, about the
            navigator's centre. The first navigator returns the zero pose.

        Raises
        ------
        ValueError
            If the navigator does not present as many planes as the reference,
            if the axes do not describe every plane as a perpendicular pair of
            directions, or if the planes leave any degree of freedom
            unmeasured.
        """
        row_axes, column_axes = _plane_axes(axes, len(planes))
        spacing = np.broadcast_to(
            np.asarray(spacing, dtype=np.float64).reshape(-1),
            (len(planes),),
        )
        if self.reference is None:
            self.reference = [np.asarray(plane).copy() for plane in planes]
            return self.filter.update(
                RigidMotionEstimate(parameters=np.zeros(6), center=np.zeros(3))
            )
        if len(planes) != len(self.reference):
            raise ValueError(
                f"this navigator has {len(planes)} planes and the reference "
                f"has {len(self.reference)}"
            )

        predicted = self.filter.predict(dt) if self.filter.initialized else None
        measurements = [
            self.filter.registration.estimate(
                reference,
                plane,
                initial=(
                    None
                    if predicted is None
                    else _project_onto_plane(predicted, row, column)
                ),
                spacing=(float(step), float(step)),
            )
            for reference, plane, row, column, step in zip(
                self.reference, planes, row_axes, column_axes, spacing, strict=True
            )
        ]
        return self.filter.update(
            _pose_from_planes(measurements, row_axes, column_axes)
        )


# %% private module subroutines


@contextmanager
def _threads(count: int | None) -> Any:
    """Run the block with ITK's process-wide thread count set to ``count``."""
    if count is None:
        yield
        return
    process = _simpleitk().ProcessObject
    restore = process.GetGlobalDefaultNumberOfThreads()
    process.SetGlobalDefaultNumberOfThreads(count)
    try:
        yield
    finally:
        process.SetGlobalDefaultNumberOfThreads(restore)


def _simpleitk() -> Any:
    try:
        return import_module("SimpleITK")
    except ImportError as error:
        raise ImportError(
            "Rigid motion tracking requires SimpleITK, which ships with "
            "pulserver; reinstall the package to restore it."
        ) from error


def _as_magnitude_array(value: Any, *, normalize: bool) -> np.ndarray:
    try:
        torch = import_module("torch")
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        value = value.abs() if value.is_complex() else value
        value = value.numpy()
    array = np.asarray(value)
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = np.squeeze(array)
    if array.ndim not in {2, 3}:
        raise ValueError("rigid registration requires one 2D or 3D image")
    if not np.isfinite(array).all():
        raise ValueError("registration images must contain only finite values")
    array = array.astype(np.float32, copy=False)
    if not normalize:
        return array
    deviation = float(array.std())
    return (array - float(array.mean())) / max(deviation, 1e-6)


def _sitk_image(
    value: Any,
    spacing: Sequence[float] | None,
    *,
    normalize: bool = True,
) -> Any:
    sitk = _simpleitk()
    image = sitk.GetImageFromArray(_as_magnitude_array(value, normalize=normalize))
    if spacing is not None:
        selected = tuple(float(item) for item in spacing)
        if len(selected) != image.GetDimension() or any(item <= 0 for item in selected):
            raise ValueError("spacing must contain one positive value per dimension")
        image.SetSpacing(selected)
    return image


def _centered_transform(fixed: Any, moving: Any) -> Any:
    sitk = _simpleitk()
    transform = (
        sitk.Euler2DTransform()
        if fixed.GetDimension() == 2
        else sitk.Euler3DTransform()
    )
    return sitk.CenteredTransformInitializer(
        fixed,
        moving,
        transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )


def _rigid_component(transform: Any) -> Any:
    if transform.GetName() != "CompositeTransform":
        return transform
    if transform.GetNumberOfTransforms() != 1:
        raise RuntimeError("registration returned an unexpected transform stack")
    return transform.GetNthTransform(0)


def _transform_from_estimate(estimate: RigidMotionEstimate) -> Any:
    sitk = _simpleitk()
    transform = (
        sitk.Euler2DTransform() if estimate.dimension == 2 else sitk.Euler3DTransform()
    )
    transform.SetCenter(tuple(map(float, estimate.center)))
    transform.SetParameters(tuple(map(float, estimate.parameters)))
    return transform


def _restore_array(array: np.ndarray, reference: Any) -> Any:
    try:
        torch = import_module("torch")
    except ImportError:
        torch = None
    if torch is not None and isinstance(reference, torch.Tensor):
        return torch.as_tensor(
            array, device=reference.device, dtype=reference.real.dtype
        )
    return array


def _plane_axes(
    axes: Sequence[Sequence[Sequence[float]]],
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(axes, dtype=np.float64)
    if selected.shape != (count, 2, 3):
        raise ValueError(
            "axes must give a row and a column direction, as three-vectors, "
            "for every plane"
        )
    norms = np.linalg.norm(selected, axis=-1, keepdims=True)
    if not np.all(norms > 0.0):
        raise ValueError("plane axes must be non-zero directions")
    selected = selected / norms
    if not np.allclose(
        np.sum(selected[:, 0] * selected[:, 1], axis=-1), 0.0, atol=1e-6
    ):
        raise ValueError(
            "a plane's row and column directions must be perpendicular: they "
            "are the axes of one image"
        )
    return selected[:, 0], selected[:, 1]


def _pose_from_planes(
    measurements: Sequence[RigidMotionEstimate],
    row_axes: np.ndarray,
    column_axes: np.ndarray,
) -> RigidMotionEstimate:
    """The one pose that best explains what every plane measured.

    A plane measures the object's motion in SimpleITK's ``(x, y)``, which is
    its image's column direction and then its row direction: an in-plane
    rotation about the plane normal, and the two components of the
    translation the plane can see. Written out over the plane axes those are
    linear in the pose, so the whole navigator is two least-squares solves --
    one for the rotation vector and one for the translation.
    """
    parameters = np.asarray(
        [measurement.parameters for measurement in measurements],
        dtype=np.float64,
    )
    normals = np.cross(row_axes, column_axes)
    # An in-plane rotation carries the column direction toward the row
    # direction, which about the plane normal is the negative sense.
    rotation = _solve(-normals, parameters[:, 0], "rotation")
    translation = _solve(
        np.concatenate([column_axes, row_axes]),
        np.concatenate([parameters[:, 1], parameters[:, 2]]),
        "translation",
    )
    return RigidMotionEstimate(
        parameters=np.concatenate([_rotvec_to_euler(rotation), translation]),
        center=np.zeros(3),
    )


def _project_onto_plane(
    pose: RigidMotionEstimate,
    row_axis: np.ndarray,
    column_axis: np.ndarray,
) -> RigidMotionEstimate:
    """What one plane would measure if the object were at ``pose``."""
    rotation = _euler_to_rotvec(pose.angles)
    translation = pose.translation
    return RigidMotionEstimate(
        parameters=(
            -float(rotation @ np.cross(row_axis, column_axis)),
            float(translation @ column_axis),
            float(translation @ row_axis),
        ),
        center=np.zeros(2),
    )


def _solve(rows: np.ndarray, values: np.ndarray, unknown: str) -> np.ndarray:
    solution, _, rank, _ = np.linalg.lstsq(rows, values, rcond=None)
    if rank < rows.shape[1]:
        raise ValueError(
            f"the navigator's planes do not span the {unknown}: they leave at "
            f"least one of its three components unmeasured"
        )
    return solution


# SimpleITK's Euler3DTransform composes its angles as Rz @ Rx @ Ry, which is
# neither of the two orders a reader expects; SciPy spells that "yxz", with
# the angles in that order rather than in the transform's.
_EULER_SEQUENCE = "yxz"
_EULER_ORDER = (1, 0, 2)


def _rotvec_to_euler(rotation: np.ndarray) -> np.ndarray:
    angles = Rotation.from_rotvec(rotation).as_euler(_EULER_SEQUENCE)
    return angles[list(np.argsort(_EULER_ORDER))]


def _euler_to_rotvec(angles: np.ndarray) -> np.ndarray:
    return Rotation.from_euler(
        _EULER_SEQUENCE,
        np.asarray(angles)[list(_EULER_ORDER)],
    ).as_rotvec()


def _noise_vector(value: float | Sequence[float], size: int, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=np.float64)
    if selected.ndim == 0:
        selected = np.full(size, float(selected))
    selected = selected.reshape(-1)
    if selected.size != size or np.any(selected <= 0.0):
        raise ValueError(f"{name} must be positive and have one value per parameter")
    return selected


def _wrap_angles(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2 * np.pi) - np.pi
