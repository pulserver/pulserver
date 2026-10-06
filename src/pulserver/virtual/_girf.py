"""The gradient system's impulse response, which the played gradients pass through."""

from __future__ import annotations

__all__ = ["Girf"]

import argparse

import numpy as np


class Girf:
    """A gradient impulse response function per physical axis.

    The gradient played along a physical axis is the one the cache asks for
    convolved with that axis's impulse response, so the moment is too.

    Parameters
    ----------
    impulse
        ``(taps, 3)`` the response along x, y and z at ``taps`` delays
        ``dt_s`` apart from zero, in 1/s; each column's sum times ``dt_s`` is
        the axis's gain at zero frequency.
    dt_s
        The spacing of the delays, in s.
    """

    def __init__(self, impulse: np.ndarray, dt_s: float) -> None:
        impulse = np.asarray(impulse, dtype=float)
        if impulse.ndim != 2 or impulse.shape[1] != 3:
            raise ValueError(f"an impulse response is (taps, 3), not {impulse.shape}")
        if not dt_s > 0.0:
            raise ValueError(f"the delays are spaced above zero, not {dt_s} s")
        self.impulse = impulse
        self.dt_s = float(dt_s)

    @classmethod
    def delay_lowpass(
        cls, delay_s: float, time_constant_s: float, dt_s: float = 1e-6
    ) -> Girf:
        """Return a delay followed by a first-order low pass, alike on every axis, of unit gain at zero frequency."""
        if time_constant_s <= 0.0:
            taps = np.zeros(round(delay_s / dt_s) + 1)
            taps[-1] = 1.0
        else:
            tau = np.arange(0.0, delay_s + 8.0 * time_constant_s, dt_s)
            taps = np.where(
                tau >= delay_s, np.exp(-(tau - delay_s) / time_constant_s), 0.0
            )
            taps /= taps.sum()
        return cls(np.repeat(taps[:, None] / dt_s, 3, axis=1), dt_s)

    def filtered(self, moment, starts_us: np.ndarray, times_us: np.ndarray):
        """Return the moment the response plays by ``times_us``, from ``moment(block, since_us)`` of the gradients asked for, ``(n, 3)`` along the physical axes."""
        out = np.zeros((times_us.size, 3))
        for tap, weights in enumerate(self.impulse):
            if not weights.any():
                continue
            at = times_us - 1e6 * tap * self.dt_s
            block = np.clip(
                np.searchsorted(starts_us, at, side="right") - 1, 0, starts_us.size - 2
            )
            since = np.maximum(at - starts_us[block], 0.0)
            out += (self.dt_s * weights) * moment(block, since)
        return out


def girf_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the option :func:`system_girf` reads to ``parser``."""
    parser.add_argument(
        "--girf",
        type=float,
        nargs=2,
        metavar=("DELAY_US", "TIME_CONSTANT_US"),
        help="the gradients play delayed by DELAY_US and through a first-order "
        "low pass of TIME_CONSTANT_US, both in µs",
    )


def system_girf(args: argparse.Namespace) -> Girf | None:
    """Return the impulse response ``--girf`` describes; None without it."""
    if args.girf is None:
        return None
    delay_us, time_constant_us = args.girf
    return Girf.delay_lowpass(1e-6 * delay_us, 1e-6 * time_constant_us)
