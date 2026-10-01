"""The slabs a scan's excitation pulses excite, outside which its isochromats carry no transverse magnetization."""

from __future__ import annotations

__all__ = ["Slabs", "excited"]

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import ir
from ._bloch import _gradients, _rf
from ._isochromats import Isochromats

#: Fraction of the largest change a pulse makes to the magnetization at rest
#: below which a field is outside the slab it excites.
THRESHOLD = 1e-2

# Scales of a pulse's amplitude its slab is found at, spanning the transmit
# fields of a coil over a head.
_SCALES = (1.0, 2.0)

# Half-width, in m, of the line along a pulse's gradient its slab is found on.
_REACH = 0.3

# Points of that line per 1/T, T the pulse's duration.
_POINTS = 16.0

# The cache's use of an excitation pulse, PULSEG_RF_USE_EXCITATION.
_EXCITATION = 1

# Largest change of a gradient during a pulse, relative to its largest axis,
# at which it is held.
_HELD = 1e-6

# Gradients excitations may play under beyond which they excite every
# isochromat: between them, the slabs of a pulse per spoke, as a ZTE scan
# plays, leave out next to none of the object.
_GRADIENTS = 16


@dataclass(frozen=True)
class Slabs:
    """Slabs of the physical frame, each the isochromats that see a field within its bounds during a pulse.

    An isochromat at ``r``, in m, precessing at ``f``, in Hz, sees the field
    ``g . r + f`` during a pulse played under the gradient ``g``, in Hz/m,
    along the physical axes; it lies in the slab when that field lies within
    the slab's bounds, in Hz, so that off-resonance moves an isochromat's slab
    as it moves the pulse's.

    Attributes
    ----------
    gradients
        Each slab's gradient, in Hz/m, along the physical axes.
    bounds
        Each slab's lowest and highest field, in Hz.
    """

    gradients: tuple[tuple[float, float, float], ...]
    bounds: tuple[tuple[float, float], ...]

    def __call__(self, positions: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        """Return which of the isochromats at ``(n, 3)`` positions, in m, precessing at ``(n,)`` frequencies, in Hz, lie in a slab."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        frequencies = np.broadcast_to(
            np.asarray(frequencies, dtype=float), (len(positions),)
        )
        kept = np.zeros(len(positions), dtype=bool)
        for gradient, (low, high) in zip(self.gradients, self.bounds, strict=True):
            field = positions @ np.asarray(gradient) + frequencies
            kept |= (field >= low) & (field <= high)
        return kept


def excited(
    seq_path: Path | str,
    rotation: np.ndarray | None = None,
    *,
    cache_ext: str = ".pseg",
    threshold: float = THRESHOLD,
) -> Slabs | None:
    """Return the slabs the excitation pulses of the cache beside a sequence file excite; None where one excites every isochromat.

    A pulse's slab holds the fields at which it changes the magnetization at
    rest by at least ``threshold`` of the most it changes it, at its nominal
    amplitude or twice it: the pulse is played on a line of isochromats along
    its gradient, turned by ``rotation`` as
    :func:`~pulserver.virtual.simulate` turns it. A pulse played without a
    gradient, or under one that changes during it, excites every isochromat,
    and so do pulses played under more than 16 gradients, as a ZTE scan's are.
    Only the pulses the cache labels excitations are counted: what the others
    tip into the transverse plane outside the slabs, such as the free
    induction decay of an imperfect refocusing pulse, is left out of a scan
    simulated in them.
    """
    played = ir.playout(Path(seq_path), waveforms=True, cache_ext=cache_ext)["blocks"]
    turn = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    slabs: dict[tuple[float, float, float], set[tuple[float, float]]] = {}
    profiles: dict[tuple[bytes, bytes], list[tuple[float, float]]] = {}
    for block in np.flatnonzero(played["rf_use"] == _EXCITATION):
        start, stop = played["rf_span"][block]
        if stop <= start or played["rf_amp_hz"][block] == 0.0:
            continue
        held = _held(played, block)
        if held is None:
            return None
        gradient = turn @ held if played["rotate"][block] else held
        if not np.any(gradient):
            return None
        if tuple(gradient.tolist()) not in slabs and len(slabs) == _GRADIENTS:
            return None
        key = (gradient.tobytes(), played["rf_waveform_hz"][start:stop].tobytes())
        if key not in profiles:
            profiles[key] = _profile(played, block, gradient, turn, threshold)
        # A frequency offset moves the fields a pulse excites by as much.
        offset = float(played["rf_freq_hz"][block])
        bounds = slabs.setdefault(tuple(gradient.tolist()), set())
        bounds.update((low + offset, high + offset) for low, high in profiles[key])
    gradients, bounds = [], []
    for gradient, held in slabs.items():
        for low, high in sorted(held):
            gradients.append(gradient)
            bounds.append((low, high))
    return Slabs(tuple(gradients), tuple(bounds))


def _held(played: dict, block: int) -> np.ndarray | None:
    """Return the block's gradient during its RF pulse along the logical axes, in Hz/m; None where it changes during it."""
    start, stop = played["rf_span"][block]
    times = 1e-6 * played["rf_time_us"][start:stop].astype(float)
    values = np.zeros((3, times.size))
    for axis, corners in enumerate(_gradients(played, block)):
        if corners is not None:
            values[axis] = np.interp(times, corners[0], corners[1], left=0.0, right=0.0)
    largest = np.abs(values).max()
    if np.ptp(values, axis=1).max() > _HELD * largest:
        return None
    return values[:, 0].copy()


def _profile(
    played: dict,
    block: int,
    gradient: np.ndarray,
    turn: np.ndarray,
    threshold: float,
) -> list[tuple[float, float]]:
    """Return the bounds, in Hz, of the runs of fields ``gradient . r`` at which the block's pulse, played without its frequency offset, excites."""
    rf = _rf(played, block)
    duration = float(np.max(rf.t) + np.min(rf.t))
    strength = float(np.linalg.norm(gradient))
    step = 1.0 / (_POINTS * duration)
    fields = np.arange(-_REACH * strength, _REACH * strength + step, step)
    positions = np.outer(fields / strength, gradient / strength)
    changed = np.zeros(fields.size)
    for scale in _SCALES:
        spins = Isochromats(positions)
        spins.play(
            1e-6 * float(played["duration_us"][block]),
            gradients=_gradients(played, block),
            rotation=turn if played["rotate"][block] else None,
            rf=SimpleNamespace(
                **{**vars(rf), "signal": scale * rf.signal, "freq_offset": 0.0}
            ),
        )
        m = spins.magnetization
        change = np.sqrt(m[:, 0] ** 2 + m[:, 1] ** 2 + (m[:, 2] - 1.0) ** 2)
        if change.max() > 0.0:
            changed = np.maximum(changed, change / change.max())
    inside = np.concatenate([[0], (changed >= threshold).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(inside))
    return [
        (float(fields[first] - step), float(fields[last - 1] + step))
        for first, last in zip(edges[::2], edges[1::2], strict=True)
    ]
