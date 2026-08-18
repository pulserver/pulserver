"""Magnetization-preparation modules: inversion, MT, T2-prep, diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pypulseq as pp

from .._gradients import make_spoiler
from .._system import round_to_raster
from . import _excitation_helpers as excitation

INVERSION_HARD_DURATION_S = 1.0e-3
INVERSION_ADIABATIC_DURATION_S = 10.24e-3
INVERSION_SPOIL_CYCLES = 4.0

MT_DEFAULT_DURATION_S = 8.0e-3
MT_DEFAULT_FLIP_DEG = 500.0
MT_DEFAULT_FREQ_OFFSET_HZ = -1500.0

T2PREP_90_DURATION_S = 0.5e-3
T2PREP_180_HARD_DURATION_S = 0.5e-3
T2PREP_180_ADIABATIC_DURATION_S = 10.24e-3

DIFFUSION_DELTA_FRACTION_OF_SEPARATION = 0.6

_DIFFUSION_DIRECTION_TABLES: dict[int, list[tuple[float, float, float]]] = {
    1: [(0.0, 0.0, 1.0)],
    3: [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    6: [
        (1.0, 1.0, 0.0), (1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0), (1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0), (0.0, 1.0, -1.0),
    ],
}


def ti_delay_seconds(
    ti_s: float,
    rf_inv,
    spoiler_duration_s: float,
    segment_duration_s: float,
    strict: bool,
) -> float | None:
    """Compute the post-spoiler delay needed to reach the requested TI.

    TI is measured from the center of the inversion RF pulse to the temporal
    center of the following excitation segment (``segment_duration_s / 2``),
    matching standard linear-order MPRAGE TI definitions.

    Parameters
    ----------
    ti_s : float
        Requested inversion time (s).
    rf_inv : SimpleNamespace
        The inversion RF event.
    spoiler_duration_s : float
        Duration of the post-inversion spoiler (s).
    segment_duration_s : float
        Duration of the excitation segment following the delay (s).
    strict : bool
        When True, return ``None`` if the required delay is negative;
        otherwise clip to zero.

    Returns
    -------
    float or None
        Raster-aligned delay (s), or ``None`` when infeasible under
        ``strict``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _excitation_helpers as excitation
    >>> from pulserver.design._rf import _preparation_helpers as preparations
    >>> rf = excitation.nonselective(pp.Opts(), 180, use="inversion")
    >>> preparations.ti_delay_seconds(0.9, rf, 1e-3, 0.1, strict=True)
    0.84855
    """
    inv_duration_s = pp.calc_duration(rf_inv)
    inv_center_s = pp.calc_rf_center(rf_inv)[0]
    elapsed_before_segment_s = (inv_duration_s - inv_center_s) + spoiler_duration_s
    ti_delay_s = ti_s - elapsed_before_segment_s - 0.5 * segment_duration_s
    ti_delay_s = round_to_raster(ti_delay_s)
    if ti_delay_s < -1e-9 and strict:
        return None
    if ti_delay_s < 0.0:
        ti_delay_s = 0.0
    return ti_delay_s


@dataclass(frozen=True)
class InversionModule:
    """Inversion-recovery preparation: RF pulse + 3-axis spoiler."""

    rf: object
    spoiler: tuple
    rf_duration_s: float
    spoiler_duration_s: float

    def ti_delay_s(self, ti_s: float, segment_duration_s: float, strict: bool) -> float | None:
        """Delay after the spoiler to hit ``ti_s`` (see :func:`ti_delay_seconds`)."""
        return ti_delay_seconds(ti_s, self.rf, self.spoiler_duration_s, segment_duration_s, strict)


def inversion(
    opts: pp.Opts,
    mode: str,
    voxel_size_m: float,
    *,
    spoil_cycles: float = INVERSION_SPOIL_CYCLES,
) -> InversionModule:
    """Build a non-selective inversion-recovery preparation module.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    mode : str
        ``"hard"`` (rectangular 180°) or ``"adiabatic"`` (hyperbolic secant).
    voxel_size_m : float
        Spoiling reference length for the 3-axis post-inversion spoiler.
    spoil_cycles : float, optional
        Dephasing cycles per ``voxel_size_m``.

    Returns
    -------
    InversionModule
        Bundled RF pulse, spoiler trapezoids, and precomputed durations.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _preparation_helpers as preparations
    >>> inv = preparations.inversion(pp.Opts(), "adiabatic", voxel_size_m=3e-3)
    >>> inv.ti_delay_s(0.9, segment_duration_s=0.1, strict=True) is not None
    True
    """
    if mode == "adiabatic":
        rf = excitation.adiabatic_inversion(opts, duration_s=INVERSION_ADIABATIC_DURATION_S)
    elif mode == "hard":
        rf = excitation.nonselective(opts, 180.0, duration_s=INVERSION_HARD_DURATION_S, use="inversion")
    else:
        raise ValueError(f"Unknown inversion mode '{mode}' (expected 'hard' or 'adiabatic')")
    spoiler = make_spoiler(opts, voxel_size_m, dephasing_cycles=spoil_cycles)
    return InversionModule(
        rf=rf,
        spoiler=spoiler,
        rf_duration_s=pp.calc_duration(rf),
        spoiler_duration_s=pp.calc_duration(*spoiler),
    )


@dataclass(frozen=True)
class MtModule:
    """Magnetization-transfer saturation: off-resonance pulse + spoiler."""

    rf: object
    spoiler: tuple
    duration_s: float


def mt_saturation(
    opts: pp.Opts,
    *,
    flip_deg: float = MT_DEFAULT_FLIP_DEG,
    freq_offset_hz: float = MT_DEFAULT_FREQ_OFFSET_HZ,
    duration_s: float = MT_DEFAULT_DURATION_S,
    voxel_size_m: float,
    spoil_cycles: float = INVERSION_SPOIL_CYCLES,
) -> MtModule:
    """Build a non-selective off-resonance Gaussian MT saturation module.

    Played once per TR immediately before excitation — MT contrast builds
    up over the steady state, so it is stacked on every view rather than
    once per segment.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    flip_deg : float, optional
        Saturation flip angle in degrees.
    freq_offset_hz : float, optional
        Off-resonance offset (Hz).
    duration_s : float, optional
        Pulse duration (s).
    voxel_size_m : float
        Spoiling reference length for the post-pulse spoiler.
    spoil_cycles : float, optional
        Dephasing cycles per ``voxel_size_m``.

    Returns
    -------
    MtModule
        Bundled RF pulse, spoiler trapezoids, and total duration.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _preparation_helpers as preparations
    >>> mt = preparations.mt_saturation(pp.Opts(), voxel_size_m=3e-3)
    >>> mt.duration_s > 0
    True
    """
    rf = pp.make_gauss_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=duration_s,
        freq_offset=freq_offset_hz,
        time_bw_product=4.0,
        system=opts,
        use="saturation",
    )
    spoiler = make_spoiler(opts, voxel_size_m, dephasing_cycles=spoil_cycles)
    return MtModule(
        rf=rf,
        spoiler=spoiler,
        duration_s=pp.calc_duration(rf) + pp.calc_duration(*spoiler),
    )


@dataclass(frozen=True)
class T2PrepModule:
    """Composite T2 preparation: 90x - tau/2 - 180y - tau/2 - tip pulse."""

    rf90_down: object
    rf180: object
    rf90_up: object
    spoiler: tuple
    delay1_s: float
    delay2_s: float
    duration_s: float


def t2_prep(
    opts: pp.Opts,
    te_prep_s: float,
    *,
    refocus_mode: str = "hard",
    final_tip: str = "up",
    voxel_size_m: float,
    strict: bool = True,
    spoil_cycles: float = INVERSION_SPOIL_CYCLES,
) -> T2PrepModule | None:
    """Build a composite non-selective T2-preparation module.

    ``te_prep_s`` (tau) sets the transverse dwell time from the tip-down RF
    center to the tip-up RF center, i.e. the T2 weighting.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    te_prep_s : float
        Preparation echo time (s).
    refocus_mode : str, optional
        ``"hard"`` (rectangular 180°) or ``"adiabatic"`` (hyperbolic secant).
    final_tip : str, optional
        ``"up"`` restores magnetization to +z (standard T2-prep);
        ``"down"`` stores it along -z (hybrid T1/T2 preparation) by rotating
        the final pulse's phase by pi relative to the ``"up"`` case.
    voxel_size_m : float
        Spoiling reference length for the post-module spoiler.
    strict : bool, optional
        When True, return ``None`` if a tau/2 delay would be negative.
    spoil_cycles : float, optional
        Dephasing cycles per ``voxel_size_m``.

    Returns
    -------
    T2PrepModule or None
        Bundled pulses, spoiler, tau/2 delays, and total duration; ``None``
        when infeasible under ``strict``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _preparation_helpers as preparations
    >>> prep = preparations.t2_prep(pp.Opts(), 50e-3, voxel_size_m=3e-3)
    >>> prep.delay1_s > 0 and prep.delay2_s > 0
    True
    """
    if final_tip not in ("up", "down"):
        raise ValueError(f"Unknown final_tip '{final_tip}' (expected 'up' or 'down')")

    rf90_down = pp.make_block_pulse(
        flip_angle=np.pi / 2.0, duration=T2PREP_90_DURATION_S, system=opts, use="preparation"
    )
    if refocus_mode == "adiabatic":
        rf180 = pp.make_adiabatic_pulse(
            pulse_type="hypsec",
            duration=T2PREP_180_ADIABATIC_DURATION_S,
            dwell=1e-5,
            system=opts,
            use="refocusing",
        )
    elif refocus_mode == "hard":
        rf180 = pp.make_block_pulse(
            flip_angle=np.pi, duration=T2PREP_180_HARD_DURATION_S, system=opts, use="refocusing"
        )
    else:
        raise ValueError(f"Unknown refocus_mode '{refocus_mode}' (expected 'hard' or 'adiabatic')")
    rf90_up = pp.make_block_pulse(
        flip_angle=np.pi / 2.0,
        duration=T2PREP_90_DURATION_S,
        phase_offset=np.pi if final_tip == "up" else 0.0,
        system=opts,
        use="preparation",
    )

    half_te_s = te_prep_s / 2.0
    d90_down_s = pp.calc_duration(rf90_down)
    c90_down_s = pp.calc_rf_center(rf90_down)[0]
    d180_s = pp.calc_duration(rf180)
    c180_s = pp.calc_rf_center(rf180)[0]
    c90_up_s = pp.calc_rf_center(rf90_up)[0]

    delay1_s = round_to_raster(half_te_s - (d90_down_s - c90_down_s) - c180_s)
    delay2_s = round_to_raster(half_te_s - (d180_s - c180_s) - c90_up_s)
    if (delay1_s < -1e-9 or delay2_s < -1e-9) and strict:
        return None
    delay1_s = max(delay1_s, 0.0)
    delay2_s = max(delay2_s, 0.0)

    spoiler = make_spoiler(opts, voxel_size_m, dephasing_cycles=spoil_cycles)
    duration_s = (
        d90_down_s + delay1_s + d180_s + delay2_s
        + pp.calc_duration(rf90_up) + pp.calc_duration(*spoiler)
    )
    return T2PrepModule(
        rf90_down=rf90_down,
        rf180=rf180,
        rf90_up=rf90_up,
        spoiler=spoiler,
        delay1_s=delay1_s,
        delay2_s=delay2_s,
        duration_s=duration_s,
    )


def diffusion_directions(n_directions: int) -> list[np.ndarray]:
    """Return ``n_directions`` diffusion-encoding unit vectors.

    Uses standard tables for 1/3/6 directions; any other count cycles
    through the x/y/z axes.

    Parameters
    ----------
    n_directions : int
        Number of directions (clamped to at least 1).

    Returns
    -------
    list of numpy.ndarray
        Unit vectors ``(x, y, z)``.

    Examples
    --------
    >>> from pulserver.design._rf import _preparation_helpers as preparations
    >>> len(preparations.diffusion_directions(6))
    6
    """
    n_directions = max(1, int(n_directions))
    raw = _DIFFUSION_DIRECTION_TABLES.get(n_directions)
    if raw is None:
        axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        raw = [axes[i % 3] for i in range(n_directions)]
    return [np.asarray(v, dtype=float) / np.linalg.norm(v) for v in raw]


def diffusion_timing(te_s: float) -> tuple[float, float]:
    """Return ``(delta_s, separation_s)`` for a Stejskal-Tanner pair given TE.

    Gradient separation (lobe start-to-start) is approximated as ``TE / 2``;
    the lobe duration ``delta`` is 60 % of the separation so both lobes fit
    inside their TE/2 windows with margin.

    Parameters
    ----------
    te_s : float
        Spin-echo time (s).

    Returns
    -------
    delta_s : float
        Diffusion lobe duration (s).
    separation_s : float
        Lobe start-to-start separation (s).

    Examples
    --------
    >>> from pulserver.design._rf import _preparation_helpers as preparations
    >>> preparations.diffusion_timing(80e-3)
    (0.024, 0.04)
    """
    separation_s = te_s / 2.0
    delta_s = DIFFUSION_DELTA_FRACTION_OF_SEPARATION * separation_s
    return delta_s, separation_s


def required_gradient_t_per_m(
    b_s_mm2: float, delta_s: float, separation_s: float, gamma_hz_per_t: float
) -> float:
    """Solve the Stejskal-Tanner equation for the gradient amplitude (T/m).

    Parameters
    ----------
    b_s_mm2 : float
        Target b-value (s/mm^2); ``<= 0`` returns 0 (b0 acquisition).
    delta_s : float
        Diffusion lobe duration (s).
    separation_s : float
        Lobe start-to-start separation (s).
    gamma_hz_per_t : float
        Gyromagnetic ratio (Hz/T).

    Returns
    -------
    float
        Required gradient amplitude in T/m.

    Examples
    --------
    >>> from pulserver.design._rf import _preparation_helpers as preparations
    >>> preparations.required_gradient_t_per_m(0.0, 0.024, 0.04, 42.576e6)
    0.0
    """
    if b_s_mm2 <= 0.0:
        return 0.0
    b_s_per_m2 = b_s_mm2 * 1e6
    gamma_rad_s_t = 2.0 * math.pi * gamma_hz_per_t
    denom = (gamma_rad_s_t**2) * (delta_s**2) * (separation_s - delta_s / 3.0)
    if denom <= 0.0:
        return float("inf")
    return math.sqrt(b_s_per_m2 / denom)


def build_diffusion_gradients(opts: pp.Opts, direction: np.ndarray, delta_s: float, grad_t_per_m: float):
    """Build one diffusion lobe: one trapezoid per non-zero direction axis.

    Callers play the returned lobe twice — once before and once after the
    refocusing pulse.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    direction : numpy.ndarray
        Unit vector ``(x, y, z)``.
    delta_s : float
        Lobe duration (s).
    grad_t_per_m : float
        Gradient amplitude (T/m); ``<= 0`` returns ``[]`` (b0).

    Returns
    -------
    list of SimpleNamespace
        Trapezoids for the non-zero axes.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _preparation_helpers as preparations
    >>> lobes = preparations.build_diffusion_gradients(
    ...     pp.Opts(), np.array([0.0, 0.0, 1.0]), 0.024, 0.01)
    >>> len(lobes)
    1
    """
    if grad_t_per_m <= 0.0:
        return []
    area_per_axis = grad_t_per_m * opts.gamma * delta_s
    trapezoids = []
    for axis, component in zip(("x", "y", "z"), direction, strict=True):
        if abs(component) < 1e-9:
            continue
        trapezoids.append(
            pp.make_trapezoid(channel=axis, area=area_per_axis * float(component), duration=delta_s, system=opts)
        )
    return trapezoids


@dataclass(frozen=True)
class DiffusionModule:
    """One Stejskal-Tanner lobe (played either side of the 180) + timing."""

    gradients: list
    delta_s: float
    separation_s: float
    grad_t_per_m: float


def diffusion_weighting(
    opts: pp.Opts,
    b_s_mm2: float,
    direction: np.ndarray,
    te_s: float,
    *,
    strict: bool = True,
) -> DiffusionModule | None:
    """Build a Stejskal-Tanner diffusion-weighting module for one direction.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    b_s_mm2 : float
        Target b-value (s/mm^2); ``<= 0`` yields an empty gradient list.
    direction : numpy.ndarray
        Unit vector ``(x, y, z)``.
    te_s : float
        Spin-echo time (s) the lobes must fit around.
    strict : bool, optional
        When True, return ``None`` if the required amplitude exceeds the
        system limit; otherwise clamp to the limit.

    Returns
    -------
    DiffusionModule or None
        Lobe gradients and timing, or ``None`` when infeasible under
        ``strict``.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _preparation_helpers as preparations
    >>> mod = preparations.diffusion_weighting(
    ...     pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s"),
    ...     500.0, np.array([0.0, 0.0, 1.0]), 80e-3)
    >>> len(mod.gradients)
    1
    """
    delta_s, separation_s = diffusion_timing(te_s)
    grad_t_per_m = required_gradient_t_per_m(b_s_mm2, delta_s, separation_s, opts.gamma)
    max_grad_t_per_m = opts.max_grad / opts.gamma
    if grad_t_per_m > max_grad_t_per_m:
        if strict:
            return None
        grad_t_per_m = max_grad_t_per_m
    gradients = build_diffusion_gradients(opts, direction, delta_s, grad_t_per_m)
    return DiffusionModule(
        gradients=gradients,
        delta_s=delta_s,
        separation_s=separation_s,
        grad_t_per_m=grad_t_per_m,
    )
