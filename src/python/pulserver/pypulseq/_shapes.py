"""Recovering the raster-edge samples of a gradient stored on a centers raster.

A Pulseq arbitrary gradient is sampled at the *centres* of the gradient raster
intervals. That is all a scanner needs, but it loses something an analysis
wants: a trapezoid that was converted to a shape no longer has its corners on
the sample grid, so reading the waveform back gives ramps rounded over one
raster interval and a peak slew rate lower than the one the sequence really
asks for.

The samples on the interval *edges* are recoverable, because the centre samples
are their pairwise means. Writing ``w`` for the centres and ``e`` for the edges,
``w[i] = (e[i] + e[i+1]) / 2``, so given ``e[0]`` -- which the event carries as
``first`` -- the rest follow by alternating sum. That reconstruction is exact
for a waveform that really was piecewise linear on the raster, and drifts for
one that never was; the drift against the recorded ``last`` is what decides
whether the result is used.

This is a port of ``refcode/pulseq/matlab/+mr/restoreAdditionalShapeSamples.m``.
PyPulseq has the fallback branch only -- ``Sequence.waveforms()`` carries a
``TODO: Implement restoreAdditionalShapeSamples`` where the reconstruction
would go -- so on a shape that takes the bail-out this agrees with upstream
exactly, and on one that does not it is upstream that is approximating.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["restore_additional_shape_samples"]

#: How far the reconstructed last sample may sit from the recorded ``last``,
#: relative to the waveform's peak, before the reconstruction is judged not to
#: describe this shape and is dropped. MATLAB's, along with MATLAB's own
#: ``% what's the reasonable threshold?`` -- do not tune it here.
RESTORE_TOLERANCE = 2e-5

#: How far a sample may sit from the straight line through its neighbours and
#: still be dropped as interior to a linear segment. MATLAB's ``1e-8``, applied
#: to the second difference of the doubled-rate waveform.
COLLINEAR_TOLERANCE = 1e-8


def restore_additional_shape_samples(
    tt: np.ndarray,
    waveform: np.ndarray,
    first: float,
    last: float,
    grad_raster_time: float,
    block: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Put a centres-raster gradient back on the raster edges and decimate it.

    Parameters
    ----------
    tt : np.ndarray
        Sample times, at the centres of the raster intervals.
    waveform : np.ndarray
        Sample amplitudes, one per entry of ``tt``.
    first, last : float
        The amplitudes at the start and end of the gradient, which the event
        carries alongside the shape.
    grad_raster_time : float
        The gradient raster interval, in seconds.
    block : int, optional
        The block the gradient came from, named in the warning when the
        reconstruction is dropped.

    Returns
    -------
    tt_chg, waveform_chg : np.ndarray
        Times and amplitudes with the raster-edge samples restored and the
        interior samples of straight segments removed. On a shape the
        reconstruction does not describe, the waveform as stored, bracketed by
        ``first`` and ``last`` -- which is what PyPulseq returns always.

    Examples
    --------
    A trapezoid stored on sample centres: three ramp samples, four flat, three
    down.

    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> raster = 10e-6
    >>> waveform = np.array([0.5, 1.5, 2.5, 3.0, 3.0, 3.0, 3.0, 2.5, 1.5, 0.5]) / 3.0
    >>> tt = (np.arange(waveform.size) + 0.5) * raster

    The straight segments collapse onto their endpoints, so ten samples come
    back as the four vertices that define them, on the raster edges:

    >>> times, amplitudes = pp.restore_additional_shape_samples(
    ...     tt, waveform, 0.0, 0.0, raster
    ... )
    >>> (times / raster).round(3)
    array([ 0.,  3.,  7., 10.])
    >>> amplitudes.round(4)
    array([0., 1., 1., 0.])
    """
    waveform = np.asarray(waveform, dtype=float).ravel()
    tt = np.asarray(tt, dtype=float).ravel()
    if waveform.size == 0 or tt.size != waveform.size:
        raise ValueError(
            f"gradient shape has {waveform.size} samples and {tt.size} sample times"
        )
    peak = float(np.max(np.abs(waveform)))

    # The edge samples by alternating sum: e[k] = -e[k-1] + 2*w[k-1], seeded
    # with e[0] = first. Signs are folded in and back out so the recurrence is
    # one cumulative sum.
    alternating = np.empty(waveform.size + 1, dtype=float)
    alternating[0] = first
    alternating[1:] = 2.0 * waveform
    signs = np.where(np.arange(alternating.size) % 2 == 0, 1.0, -1.0)
    restored = np.cumsum(alternating * signs) * signs

    if abs(restored[-1] - last) > RESTORE_TOLERANCE * peak:
        # The shape was never piecewise linear on the raster -- a spiral, most
        # often -- so the reconstruction has integrated its way somewhere the
        # recorded last sample says it should not be. MATLAB warns here and so
        # do we: which branch a shape took changes what the waveform means.
        where = "" if block is None else f"[block {block}] "
        deviation = abs(restored[-1] - last)
        relative = "" if peak == 0.0 else f" ({deviation / peak * 100:.3g}%)"
        warnings.warn(
            f"{where}restoring the gradient's raster-edge samples put the last "
            f"one {deviation:.6g} Hz/m{relative} away from the recorded value, "
            f"so the restoration was dropped. This is typical for spirals and "
            f"other genuinely non-linear gradients.",
            stacklevel=2,
        )
        return (
            np.concatenate(([0.0], tt, [tt[-1] + grad_raster_time / 2.0])),
            np.concatenate(([first], waveform, [last])),
        )

    # Where the reconstruction agrees with plain interpolation, prefer the
    # interpolation: it is the better-conditioned of the two, and the
    # alternating sum accumulates error along the shape.
    interpolated = np.concatenate(
        ([first], 0.5 * (waveform[:-1] + waveform[1:]), [last])
    )
    agree = (
        np.abs(restored - interpolated)
        <= np.finfo(float).eps + RESTORE_TOLERANCE * peak
    )
    edges = np.where(agree, interpolated, restored)

    # Interleave edges and centres into one waveform on the half raster.
    oversampled = np.empty(edges.size + waveform.size, dtype=float)
    oversampled[0::2] = edges
    oversampled[1::2] = waveform
    times = np.arange(oversampled.size, dtype=float) * grad_raster_time * 0.5

    # Keep only the samples that bend: on a straight segment the second
    # difference vanishes and the interior samples say nothing. The endpoints
    # are always kept.
    curvature = np.concatenate(([1.0], np.diff(oversampled, n=2), [1.0]))
    keep = np.abs(curvature) > COLLINEAR_TOLERANCE
    return times[keep], oversampled[keep]
