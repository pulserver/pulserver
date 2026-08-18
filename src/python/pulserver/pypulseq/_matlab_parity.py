"""Routines MATLAB Pulseq defines and PyPulseq has no equivalent of.

Each function here is a port of the file named in its ``References`` section,
under ``refcode/pulseq/matlab/+mr/``, with MATLAB's semantics kept and its
naming converted to PyPulseq's.
"""

from __future__ import annotations

__all__ = [
    "calc_rf_power",
    "get_supported_rf_use",
    "make_hexagon_gradient_area",
    "verify_file_signature",
]

import hashlib
from pathlib import Path

import numpy as np
from pypulseq.supported_labels_rf_use import (
    get_supported_rf_uses as _get_supported_rf_uses,
)

from . import _events

#: Slack allowed when comparing a computed slew or amplitude against a limit,
#: matching the ``1e-5`` MATLAB's solver uses throughout.
_LIMIT_TOLERANCE = 1e-5


def get_supported_rf_use() -> tuple[str, ...]:
    """Return the ``use`` tags an RF event may carry.

    Returns
    -------
    tuple of str
        ``excitation``, ``refocusing``, ``inversion``, ``saturation``,
        ``preparation``, ``other``, ``undefined``.

    Examples
    --------
    >>> from pulserver.pypulseq import get_supported_rf_use
    >>> "refocusing" in get_supported_rf_use()
    True

    References
    ----------
    ``getSupportedRfUse.m``.
    """
    return tuple(_get_supported_rf_uses())


#: PyPulseq's plural spelling of the same tuple.
get_supported_rf_uses = get_supported_rf_use


def calc_rf_power(rf, dt: float = 1e-6) -> tuple[float, float, float]:
    """Relative energy, peak power and RMS amplitude of one RF pulse.

    The quantities are *relative*: Pulseq stores RF amplitude in Hz, so the
    energy is Hz^2*s and the peak power Hz^2. Divide ``rf_rms`` by the
    gyromagnetic ratio for Tesla and the energy by its square for Tesla^2*s;
    absolute SAR needs coil and subject factors that are not knowable here.

    The pulse is resampled onto a uniform grid of step ``dt`` before
    integrating, which a block pulse requires — its ``t`` holds only the two
    endpoints. ``freq_offset``, ``phase_offset`` and ``freq_ppm`` do not enter,
    the integrand being ``|signal|^2``.

    Parameters
    ----------
    rf : SimpleNamespace or RfEvent
        RF event, carrying ``t``, ``signal`` and ``shape_dur``.
    dt : float, optional
        Resampling step (s). A narrower pulse wants a smaller one: the peak is
        estimated from the resampled waveform and so depends on ``dt``.

    Returns
    -------
    total_energy : float
        Integral of ``|signal|^2`` over the pulse, Hz^2*s.
    peak_power : float
        Largest ``|signal|^2``, Hz^2.
    rf_rms : float
        ``sqrt(total_energy / shape_dur)``, Hz. ``shape_dur`` excludes any
        delay or ringdown, so this is not a sequence duty cycle.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=170, slew_unit="T/m/s")
    >>> rf = pp.make_sinc_pulse(
    ...     flip_angle=np.pi / 2, duration=3e-3, time_bw_product=4, system=system
    ... )
    >>> energy, peak, rms = pp.calc_rf_power(rf)
    >>> bool(rms > 0 and peak >= rms**2)
    True

    A pulse of twice the flip angle over the same duration carries four times
    the energy:

    >>> rf2 = pp.make_sinc_pulse(
    ...     flip_angle=np.pi, duration=3e-3, time_bw_product=4, system=system
    ... )
    >>> float(np.round(pp.calc_rf_power(rf2)[0] / energy, 6))
    4.0

    See Also
    --------
    pulserver.pypulseq.Sequence.calc_rf_power : the same question of a whole sequence.

    References
    ----------
    ``calcRfPower.m``.
    """
    rf = _events.as_namespace(rf)
    duration = float(rf.shape_dur)
    if dt <= 0:
        raise ValueError("dt must be > 0")
    count = round(duration / dt)
    if count < 1:
        raise ValueError(f"dt={dt} is longer than the pulse it samples ({duration} s)")

    time = (np.arange(count) + 0.5) * dt
    samples = np.interp(
        time, np.asarray(rf.t, dtype=float), np.asarray(rf.signal), left=0.0, right=0.0
    )
    power = np.real(samples * np.conj(samples))

    total_energy = float(power.sum() * dt)
    return total_energy, float(power.max()), float(np.sqrt(total_energy / duration))


def make_hexagon_gradient_area(
    area: float,
    channel: str,
    grad_start: float,
    grad_end: float,
    system=None,
):
    """Shortest gradient of a given area between two amplitudes, without a plateau.

    The counterpart to :func:`pypulseq.make_extended_trapezoid_area`, which
    holds its two middle vertices at one amplitude. Freeing them reaches some
    areas in fewer raster steps, and where it cannot the flat-topped shape is
    what comes back.

    Parameters
    ----------
    area : float
        Target gradient area (1/m).
    channel : str
        Gradient channel (``"x"``, ``"y"`` or ``"z"``).
    grad_start, grad_end : float
        Amplitudes the gradient must begin and end at (Hz/m).
    system : pypulseq.Opts, optional
        System limits. The solver works to 99% of ``max_grad`` and ``max_slew``.

    Returns
    -------
    grad : GradEvent
        The extended trapezoid.
    times : numpy.ndarray
        Its vertex times (s).
    amplitudes : numpy.ndarray
        Its vertex amplitudes (Hz/m).

    Raises
    ------
    ValueError
        If no waveform within the limits reaches ``area``.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> grad, times, amplitudes = pp.make_hexagon_gradient_area(
    ...     area=1000, channel="x", grad_start=0.0, grad_end=0.0, system=system
    ... )
    >>> bool(abs(np.trapezoid(amplitudes, times) - 1000) < 1e-3)
    True

    It also starts from a gradient that is already running, which is what makes
    it usable for a crusher wrapped around a slice-select lobe:

    >>> grad, times, amplitudes = pp.make_hexagon_gradient_area(
    ...     area=1000, channel="x", grad_start=5000.0, grad_end=0.0, system=system
    ... )
    >>> float(amplitudes[0])
    5000.0

    See Also
    --------
    make_extended_trapezoid_area : the flat-topped shape this relaxes.

    References
    ----------
    ``makeHexagonGradientArea.m``.
    """
    from pypulseq.opts import Opts

    if system is None:
        system = Opts.default
    max_slew = system.max_slew * 0.99
    max_grad = system.max_grad * 0.99
    raster = system.grad_raster_time

    def ramp_steps(from_amplitude: float, to_amplitude: float) -> int:
        return int(np.ceil(abs(from_amplitude - to_amplitude) / max_slew / raster))

    minimum = max(ramp_steps(grad_end, grad_start), 2)
    maximum = max(ramp_steps(0.0, grad_start), ramp_steps(0.0, grad_end), minimum)

    def solve(steps):
        return _hexagon_solution(
            steps, area, grad_start, grad_end, max_slew, max_grad, raster
        )

    found = None
    for duration in range(minimum, maximum + 1):
        found = solve(duration)
        if found is not None:
            break

    if found is None:
        duration = maximum
        while found is None:
            duration *= 2
            found = solve(duration)
        found = _shortest_solution(solve, duration // 2, duration)

    times, amplitudes = found
    if np.any(np.diff(times) == 0):
        times = np.array([times[0], times[1], times[3]])
        amplitudes = np.array([amplitudes[0], amplitudes[1], amplitudes[3]])

    # The vertices are what the solver reasoned about, and a piecewise-linear
    # waveform's area is exact under the trapezoidal rule, so check them rather
    # than the event built from them.
    if abs(float(np.trapezoid(amplitudes, times)) - area) >= 1e-3:
        raise ValueError(f"could not find a solution for area={area:.6f}")

    grad = _events.make_extended_trapezoid(
        channel=channel, system=system, times=times, amplitudes=amplitudes
    )
    return grad, times, amplitudes


def _shortest_solution(solve, low: int, high: int):
    """The fewest raster steps in ``(low, high]`` that ``solve`` still answers for."""
    while low < high - 1:
        middle = (low + high) // 2
        if solve(middle) is not None:
            high = middle
        else:
            low = middle
    return solve(high)


def _hexagon_solution(duration, area, grad_start, grad_end, max_slew, max_grad, raster):
    """Vertex times and amplitudes for one candidate duration, or ``None``.

    ``duration`` counts raster steps. The search walks the ramp lengths outward
    from their minima, and at each step solves for the pair of corner
    amplitudes that hits ``area`` exactly, rejecting any that breaks a limit.
    """
    sign = np.sign(area) if area != 0 else 1.0
    amplitude = sign * max_grad

    up_min = abs(amplitude - grad_start) / max_slew / raster
    down_min = abs(amplitude - grad_end) / max_slew / raster
    flat = max(duration - up_min - down_min, 0.0)
    reachable = (
        up_min * (amplitude + grad_start)
        + down_min * (amplitude + grad_end)
        + 2 * flat * amplitude
    )

    if abs(2 * area / raster) > abs(reachable):
        return None

    flat_topped = _flat_topped_solution(
        duration, area, grad_start, grad_end, max_slew, max_grad, raster, sign
    )
    if flat_topped is not None:
        return flat_topped

    while abs(2 * area / raster) < abs(reachable):
        amplitude /= 2
        if abs(amplitude) < abs(max_grad) / 10:
            up_min = down_min = 0.0
            flat = max(duration - up_min - down_min, 0.0)
            break
        up_min = abs(amplitude - grad_start) / max_slew / raster
        down_min = abs(amplitude - grad_end) / max_slew / raster
        flat = max(duration - up_min - down_min, 0.0)
        reachable = (
            up_min * (amplitude + grad_start)
            + down_min * (amplitude + grad_end)
            + 2 * flat * amplitude
        )

    up_min = int(np.floor(up_min))
    down_min = int(np.floor(down_min))
    up_limit = int(np.ceil(abs(sign * max_grad - grad_start) / max_slew / raster))
    down_limit = int(np.ceil(abs(sign * max_grad - grad_end) / max_slew / raster))

    flat = duration - min(down_min, down_limit) - min(up_min, up_limit)
    flat_floor = duration - down_limit - up_limit

    closest = area
    triangles: list[tuple[int, int]] = []
    step = -1

    while flat > max(flat_floor, -1):
        step += 1
        up = up_min + step
        down = down_min + step
        flat = (
            duration - min(down_min + step, down_limit) - min(up_min + step, up_limit)
        )

        if flat <= 0:
            triangles = _triangle_candidates(
                duration, area, grad_start, grad_end, max_slew, raster, sign
            )
            break

        corner_up = sign * max_slew * up * raster + grad_start
        corner_down = sign * max_slew * down * raster + grad_end

        if abs(corner_up) >= max_grad:
            corner_up, up, up_limit = _clamp_corner(
                sign,
                max_slew,
                max_grad,
                raster,
                grad_start,
                corner_down,
                up_limit,
                flat,
            )
            flat = duration - down - up
        if abs(corner_down) >= max_grad:
            corner_down, down, down_limit = _clamp_corner(
                sign, max_slew, max_grad, raster, grad_end, corner_up, down_limit, flat
            )
            flat = duration - down - up

        if flat > 0 and abs(corner_up - corner_down) / (flat * raster) > max_slew:
            if abs(corner_up) < abs(corner_down):
                corner_down = corner_up + flat * raster * max_slew * sign
                down = int(np.ceil(abs(grad_end - corner_down) / max_slew / raster))
            else:
                corner_up = corner_down + flat * raster * max_slew * sign
                up = int(np.ceil(abs(grad_start - corner_up) / max_slew / raster))
            flat = duration - down - up

        current = (raster / 2) * (
            up * (corner_up + grad_start)
            + flat * (corner_up + corner_down)
            + down * (corner_down + grad_end)
        )

        if np.sign(closest) != np.sign(area - current):
            solved = _corner_solution(
                area,
                grad_start,
                grad_end,
                corner_up,
                corner_down,
                up,
                down,
                flat,
                raster,
                max_slew,
            )
            if solved is not None:
                return solved
            fallback = _flat_topped_scan(
                duration,
                area,
                grad_start,
                grad_end,
                max_slew,
                max_grad,
                raster,
                up,
                down,
            )
            if fallback is not None:
                return fallback

        if abs(current - area) < abs(closest):
            closest = area - current

    return _triangle_solution(
        triangles, duration, area, grad_start, grad_end, max_slew, max_grad, raster
    )


def _clamp_corner(sign, max_slew, max_grad, raster, edge, other_corner, limit, flat):
    """A corner amplitude pinned to ``max_grad``, and the ramp length that reaches it.

    Two ways to stay inside the limit: stop one raster step short of it, or sit
    on it exactly. Whichever leaves the larger amplitude wins, since that is
    the one that accrues area faster.
    """
    one_step_short = abs(sign * max_slew * (limit - 1) * raster + edge)
    on_the_limit = abs(
        ((limit + flat) * sign * max_grad + edge - other_corner) / (limit + flat)
    )
    if one_step_short > on_the_limit:
        corner = sign * max_slew * (limit - 1) * raster + edge
        steps = int(np.floor(abs(sign * max_grad - edge) / max_slew / raster))
    else:
        corner = sign * max_grad
        steps = int(np.ceil(abs(sign * max_grad - edge) / max_slew / raster))
    return corner, steps, steps


def _corner_solution(
    area, grad_start, grad_end, corner_up, corner_down, up, down, flat, raster, max_slew
):
    """Solve the free corner for exactly ``area``, or ``None`` if it breaks the slew limit."""
    times = np.round(np.cumsum([0, up, flat, down]) * raster, 5)
    if np.any(np.round(np.diff(times) / raster) < 1):
        return None

    if abs(corner_up) < abs(corner_down):
        fixed = corner_down
        free = -(
            grad_start * up
            + grad_end * down
            + fixed * (flat + down)
            - 2 * area / raster
        ) / (flat + up)
        amplitudes = np.array([grad_start, free, fixed, grad_end])
    else:
        fixed = corner_up
        free = -(
            grad_start * up + grad_end * down + fixed * (flat + up) - 2 * area / raster
        ) / (flat + down)
        amplitudes = np.array([grad_start, fixed, free, grad_end])

    if (
        np.max(np.abs(np.diff(amplitudes) / np.diff(times)))
        <= max_slew + _LIMIT_TOLERANCE
    ):
        return times, amplitudes
    return None


def _flat_topped_solution(
    duration, area, grad_start, grad_end, max_slew, max_grad, raster, sign
):
    """The plateau shape, tried first because it is the one that usually works."""
    up = (duration * max_slew * raster + sign * (grad_end - grad_start)) / (
        2 * max_slew * raster
    )
    if sign * grad_start + up * max_slew * raster <= max_grad + _LIMIT_TOLERANCE:
        return None
    up_steps = round(abs(grad_start - sign * max_grad) / max_slew / raster)
    down_steps = round(abs(grad_end - sign * max_grad) / max_slew / raster)
    return _plateau(
        duration,
        area,
        grad_start,
        grad_end,
        max_slew,
        max_grad,
        raster,
        up_steps,
        down_steps,
    )


def _flat_topped_scan(
    duration, area, grad_start, grad_end, max_slew, max_grad, raster, up, down
):
    """Widen both ramps until a plateau shape fits, when the corner solve will not."""
    for up_steps in range(max(up - 1, 0), duration - down + 1):
        for down_steps in range(max(down - 1, 0), duration - up_steps + 1):
            found = _plateau(
                duration,
                area,
                grad_start,
                grad_end,
                max_slew,
                max_grad,
                raster,
                up_steps,
                down_steps,
            )
            if found is not None:
                return found
    return None


def _plateau(
    duration,
    area,
    grad_start,
    grad_end,
    max_slew,
    max_grad,
    raster,
    up_steps,
    down_steps,
):
    """One flat-topped candidate: the plateau that hits ``area``, if it is legal."""
    flat = duration - up_steps - down_steps
    if flat <= 0:
        return None
    amplitude = -(
        up_steps * raster * grad_start + down_steps * raster * grad_end - 2 * area
    ) / ((up_steps + 2 * flat + down_steps) * raster)
    amplitudes = np.array([grad_start, amplitude, amplitude, grad_end])
    times = np.cumsum([0, up_steps, flat, down_steps]) * raster
    if np.any(np.diff(times) <= 0):
        return None
    slew = np.abs(np.diff(amplitudes) / np.diff(times))
    if slew.max() < max_slew + _LIMIT_TOLERANCE and np.abs(amplitudes).max() < max_grad:
        return times, amplitudes
    return None


def _triangle_candidates(duration, area, grad_start, grad_end, max_slew, raster, sign):
    """Ramp-length pairs to try once no plateau is left to shrink."""
    numerator = 2 * area - duration * (grad_end + grad_start) * raster
    low = (
        numerator
        / (grad_start - grad_end + sign * duration * max_slew * raster)
        / raster
    )
    high = (
        duration
        - numerator
        / (-grad_start + grad_end + sign * duration * max_slew * raster)
        / raster
    )
    return [
        (up, down)
        for up in range(int(np.floor(low)), int(np.ceil(high)) + 1)
        for down in (duration - up - 1, duration - up)
    ]


def _triangle_solution(
    candidates, duration, area, grad_start, grad_end, max_slew, max_grad, raster
):
    """The first candidate pair whose single-peak shape stays inside both limits."""
    for up, down in candidates:
        if up <= 0 or down <= 0:
            continue
        flat = duration - up - down
        if flat < 0:
            continue
        amplitude = -(
            up * raster * grad_start + down * raster * grad_end - 2 * area
        ) / ((up + 2 * flat + down) * raster)
        if abs(amplitude) > max_grad + _LIMIT_TOLERANCE:
            continue
        if abs(grad_start - amplitude) / (up * raster) > max_slew + _LIMIT_TOLERANCE:
            continue
        if abs(grad_end - amplitude) / (down * raster) > max_slew + _LIMIT_TOLERANCE:
            continue
        times = np.cumsum([0, up, flat, down]) * raster
        return times, np.array([grad_start, amplitude, amplitude, grad_end])
    return None


#: The bytes a binary Pulseq file opens with, so a file is identified by what
#: it holds rather than by what it is called.
_BINARY_MAGIC = b"\x01pulseq\x02"


def verify_file_signature(pulseq_file: str | Path) -> tuple[bool, str, str]:
    """Check a ``.seq`` file against the hash stored in its ``[SIGNATURE]``.

    Parameters
    ----------
    pulseq_file : str or pathlib.Path
        The file to verify.

    Returns
    -------
    ok : bool
        Whether the stored and recomputed hashes agree.
    stored : str
        The hash the file carries.
    computed : str
        The hash of the file's contents up to the ``[SIGNATURE]`` section.

    Raises
    ------
    ValueError
        If the file carries no ``[SIGNATURE]`` section, or is a binary Pulseq
        file — that format has no signature section to check.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> seq = pp.Sequence(system)
    >>> _ = seq.add_block(pp.make_trapezoid(channel="x", area=1000, system=system))
    >>> signature = seq.write("signed.seq")  # doctest: +SKIP
    >>> ok, stored, computed = pp.verify_file_signature("signed.seq")  # doctest: +SKIP
    >>> ok and stored == signature  # doctest: +SKIP
    True

    References
    ----------
    ``verifyFileSignature.m``.
    """
    payload = Path(pulseq_file).read_bytes()
    if payload.startswith(_BINARY_MAGIC):
        raise ValueError(
            f"{pulseq_file} is a binary Pulseq file, which carries no [SIGNATURE] section"
        )

    marker = b"\n[SIGNATURE]\n"
    at = payload.rfind(marker)
    if at < 0:
        raise ValueError(f"{pulseq_file} has no [SIGNATURE] section")

    stored = ""
    for line in payload[at:].split(b"\n"):
        if line.startswith(b"Hash "):
            stored = line[len(b"Hash ") :].strip().decode("ascii").lower()

    computed = hashlib.md5(payload[:at]).hexdigest()  # noqa: S324 - the format specifies md5
    return stored == computed, stored, computed
