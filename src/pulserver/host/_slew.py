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


def lowered(samples: Sequence[tuple[float, Mapping[str, float]]]) -> float:
    """Return the slew rate the next design is made under, from the designs made so far.

    Each sample is the slew rate a design was made under and its readings,
    each a fraction of its limit. Slew rate scales the peak slew rates in
    proportion. The nerve response of a ramp of amplitude step ``dG`` at slew
    rate ``S`` is ``dG / (rheobase (chronaxie + dG / S))``, so a response
    follows ``A S / (S + x)``, a straight line in ``1 / S``: two designs fix
    it, and one is extrapolated as proportional, which, the response being
    concave in ``S``, never lowers it enough, so a second pass then fits the
    line. The slew rate returned brings every reading to :data:`TARGET` of
    its limit, and is below the last one tried.
    """
    slew, readings = samples[-1]
    candidates = [
        slew * TARGET / readings[k]
        for k in ("slew", "vector_slew")
        if readings.get(k, 0.0) > 1.0
    ]
    nerve = [(s, r["pns"]) for s, r in samples if r.get("pns", 0.0) > 0.0]
    if readings.get("pns", 0.0) > 1.0:
        candidates.append(_nerve_slew(nerve))
    chosen = min(candidates) if candidates else slew * TARGET
    return min(chosen, slew * TARGET)


def _nerve_slew(samples: Sequence[tuple[float, float]]) -> float:
    """Return the slew rate at which the line through the last two samples in ``(1/S, 1/response)`` reaches :data:`TARGET`.

    With one sample, or two the line cannot extrapolate (a response that did
    not fall as the slew rate did), the response is taken as proportional to
    the slew rate.
    """
    slew, response = samples[-1]
    proportional = slew * TARGET / response
    if len(samples) < 2:
        return proportional
    (s0, r0), (s1, r1) = samples[-2], samples[-1]
    if s0 == s1 or r0 == r1:
        return proportional
    slope = (1.0 / r1 - 1.0 / r0) / (1.0 / s1 - 1.0 / s0)
    intercept = 1.0 / r1 - slope / s1
    if not slope > 0.0:
        return proportional
    inverse = (1.0 / TARGET - intercept) / slope
    return 1.0 / inverse if inverse > 0.0 else proportional


class Remembered:
    """The slew rate each request was last lowered to, so a design starts from what its validation found.

    A request is a key built by :func:`key`. Kept in this process for the
    most recent requests and, under a warm server, among the entries its
    calls keep (:mod:`pulserver.host._memo`), where the call that generates a
    request finds what the call that validated it left. A request searched
    and left at the design limits is remembered as such, with no slew rate.
    """

    def __init__(self, size: int = 256) -> None:
        self._size = size
        self._slews: OrderedDict[Hashable, float | None] = OrderedDict()
        self._lock = threading.Lock()

    def entry(self, key: str) -> tuple[bool, float | None]:
        """Return whether a request was searched, and the slew rate it was lowered to, ``None`` if it was not lowered."""
        found, slew = _memo.read(_KIND, key)
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
        _memo.write(_KIND, key, slew)


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
