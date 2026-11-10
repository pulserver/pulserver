"""The slew rate a design is lowered to when it exceeds the nerve-stimulation or slew-rate limit."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from . import _memo

#: Designs made again at a lower slew rate modelled on the readings, and
#: again at a lower one where the plugin cannot make the layout, before a
#: request is refused.
PASSES = 3

#: Fraction of each limit a lowered slew rate aims at: the response a model
#: predicts is not the response the design then plays.
TARGET = 0.95

#: The readings of :func:`pulserver.ir.check` a lower slew rate eases.
READINGS = ("pns", "slew", "vector_slew")


#: The readings of :func:`pulserver.ir.check` no slew rate eases.
AMPLITUDES = ("grad", "vector_grad")

# The entries of :mod:`pulserver.host._memo` the remembered slew rates are.
_KIND = "slew"


def exceeds(readings: Mapping[str, float]) -> bool:
    """Whether a lower slew rate could bring a design within its limits.

    A reading a lower slew rate eases is above its limit and no gradient
    amplitude is, which no slew rate brings down.
    """
    return any(readings.get(key, 0.0) > 1.0 for key in READINGS) and not any(
        readings.get(key, 0.0) > 1.0 for key in AMPLITUDES
    )


def lowered(
    samples: Sequence[tuple[float, Mapping[str, float]]],
    nerve: tuple[float, float] | None = None,
    response: float | None = None,
) -> float:
    """Return the slew rate the next design is made under, from the designs made so far.

    Each sample is the slew rate a design was made under and its readings,
    each a fraction of its limit. Slew rate scales the peak slew rates in
    proportion. The nerve response of a ramp of amplitude step ``dG`` at slew
    rate ``S`` is ``dG / (rheobase (chronaxie + dG / S))``, so a response
    follows ``A S / (S + x)``, a straight line in ``1 / S``: two designs fix
    it. One design fixes it too where ``x`` is known: ``response``, the
    ``x`` an earlier search found (:func:`response_of`), or else that of one
    ramp, ``A`` times the rheobase and the limit, with ``nerve`` the rheobase
    in the unit of the slew rates and the largest response allowed. Without
    either the one design is extrapolated as proportional, which, the
    response being concave in ``S``, never lowers it enough, so a second pass
    then fits the line. The slew rate returned brings every reading past its
    limit to :data:`TARGET` of it, and is below the last one tried.
    """
    slew = samples[-1][0]
    return min(fitted(samples, slew * TARGET, nerve, response), slew * TARGET)


def fitted(
    samples: Sequence[tuple[float, Mapping[str, float]]],
    limit: float,
    nerve: tuple[float, float] | None = None,
    response: float | None = None,
) -> float:
    """Return the slew rate, at most ``limit``, at which the readings a lower slew rate eases reach :data:`TARGET` of their limits.

    As :func:`lowered` models them, from the last of ``samples``; the slew
    rate may be above the last one tried, where its readings leave room.
    A slew-rate reading within its limit holds nothing back: the design
    limits already bound it.
    """
    slew, readings = samples[-1]
    candidates = [
        slew * TARGET / readings[k]
        for k in ("slew", "vector_slew")
        if readings.get(k, 0.0) > 1.0
    ]
    felt = [(s, r["pns"]) for s, r in samples if r.get("pns", 0.0) > 0.0]
    if felt and readings.get("pns", 0.0) > 0.0:
        candidates.append(_nerve_slew(felt, nerve, response))
    return min([*candidates, limit])


def response_of(samples: Sequence[tuple[float, Mapping[str, float]]]) -> float | None:
    """Return the ``x`` of the nerve response ``A S / (S + x)`` through the last two of ``samples`` that state one, ``None`` where they fix none."""
    felt = [(s, r["pns"]) for s, r in samples if r.get("pns", 0.0) > 0.0]
    line = _line(felt)
    return None if line is None else line[0] / line[1]


def _line(samples: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Return the slope and intercept of the line through the last two samples in ``(1/S, 1/response)``, ``None`` where it models no nerve response.

    A nerve response rises with the slew rate and levels off: the slope and
    the intercept, ``x / A`` and ``1 / A``, are positive.
    """
    if len(samples) < 2:
        return None
    (s0, r0), (s1, r1) = samples[-2], samples[-1]
    if s0 == s1 or r0 == r1:
        return None
    slope = (1.0 / r1 - 1.0 / r0) / (1.0 / s1 - 1.0 / s0)
    intercept = 1.0 / r1 - slope / s1
    if not slope > 0.0 or not intercept > 0.0:
        return None
    return slope, intercept


def _nerve_slew(
    samples: Sequence[tuple[float, float]],
    nerve: tuple[float, float] | None = None,
    response: float | None = None,
) -> float:
    """Return the slew rate at which the nerve response modelled on ``samples`` reaches :data:`TARGET`; see :func:`lowered`.

    The line through the last two samples where they fix one; else the
    response through the last sample with ``x`` as ``response`` or, without
    it, as one ramp's under ``nerve``; else, a response taken as proportional
    to the slew rate. Infinite where the modelled response never reaches
    :data:`TARGET`.
    """
    slew, reading = samples[-1]
    line = _line(samples)
    if line is not None:
        slope, intercept = line
        inverse = (1.0 / TARGET - intercept) / slope
        return 1.0 / inverse if inverse > 0.0 else float("inf")
    if response is None and nerve is not None:
        rheobase, limit = nerve
        # One ramp: A = dG / (rheobase chronaxie limit) and x = dG / chronaxie.
        floor = reading * limit * rheobase
        if slew > floor > 0.0:
            response = reading * slew / (slew - floor) * limit * rheobase
    if response is None or not response > 0.0:
        return slew * TARGET / reading
    scale = reading * (slew + response) / slew
    return TARGET * response / (scale - TARGET) if scale > TARGET else float("inf")


class Remembered:
    """The slew rate each request was last lowered to, so a design starts from what its validation found.

    A request is a key built by :func:`key`. Kept in this process for the
    most recent requests and, under a warm server, among the entries its
    calls keep (:mod:`pulserver.host._memo`), where the call that generates a
    request finds what the call that validated it left. A request searched
    and left at the design limits is remembered as such, with no slew rate.
    ``kind`` names the entries, for a value other than a slew rate kept the
    same way.
    """

    def __init__(self, size: int = 256, kind: str = _KIND) -> None:
        self._size = size
        self._kind = kind
        self._slews: OrderedDict[Hashable, float | None] = OrderedDict()
        self._lock = threading.Lock()

    def entry(self, key: str) -> tuple[bool, float | None]:
        """Return whether a request was searched, and the slew rate it was lowered to, ``None`` if it was not lowered."""
        found, slew = _memo.read(self._kind, key)
        if found:
            return True, None if slew is None else float(slew)
        with self._lock:
            if key not in self._slews:
                return False, None
            self._slews.move_to_end(key)
            return True, self._slews[key]

    def get(self, key: str) -> float | None:
        """Return the slew rate a request was lowered to, ``None`` if it was not."""
        return self.entry(key)[1]

    def put(self, key: str, slew: float | None) -> None:
        """Remember the slew rate a request was lowered to, or that it was searched and not lowered."""
        with self._lock:
            self._slews[key] = slew
            self._slews.move_to_end(key)
            while len(self._slews) > self._size:
                self._slews.popitem(last=False)
        _memo.write(self._kind, key, slew)


def key(
    variant: Hashable, limits: Mapping[str, Any], request: Mapping[Any, Any]
) -> str:
    """Return the key a request is remembered by: its plugin and options, limits and requested values."""
    return json.dumps(
        [
            repr(variant),
            sorted((str(k), str(v)) for k, v in limits.items()),
            sorted((str(k), str(v)) for k, v in request.items()),
        ]
    )
