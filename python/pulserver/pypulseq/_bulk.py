"""Registering a repeating group of blocks for many shots at once.

``add_block`` costs a fixed amount of Python per event, and a large 3D protocol
has millions of them: on a 512-cubed MPRAGE it is around two thirds of the
whole design pass. Nothing in that work is interesting -- the same module is
re-rendered per shot, and the same payload fields are rebuilt with two or three
numbers changed -- so :meth:`pulserver.pypulseq.Sequence.add_range` registers a
whole *range* of shots in one go, and this module holds the small helpers that
turn a call's keyword states into per-shot columns.

Two properties make that safe rather than merely fast.

**Order.** Event IDs are handed out in visit order, and ``remove_duplicates``
picks each rounded representative by first occurrence, so registering the same
events in a different order changes the emitted file even though the sequence
is physically identical. A group of ``k`` entries into one library, repeated
``n`` times, therefore reserves ``n * k`` consecutive entries and fills slot
``j`` at ``slab[j::k]`` -- exactly where the per-shot loop would have put it.

**Payloads.** Every shot is really rendered and really walked; what the range
saves is the *registration*, not the design. The first shot's walk fixes the
structure -- block count, event order, which library row each event owns -- and
later shots only write numbers into rows it already reserved. Nothing declares
which fields vary, so nothing can declare it wrongly: a shot that moves
something structural stops matching and the whole group falls back to
block-by-block insertion.
"""

from __future__ import annotations

__all__ = ["expand_states", "infer_length"]

from typing import Any

import numpy as np


def expand_states(n: int, states: dict[str, Any]) -> dict[str, np.ndarray]:
    """Broadcast a mixed dict of scalars and sequences to ``(n,)`` arrays.

    A shot range is usually described with a few arrays and a few constants —
    ``lin_idx`` sweeping while ``par_idx`` holds — and the constants should not
    have to be spelled as arrays.
    """
    out: dict[str, np.ndarray] = {}
    for name, value in states.items():
        array = np.asarray(value)
        if array.ndim == 0:
            array = np.repeat(array, n)
        elif len(array) != n:
            raise ValueError(f"state {name!r} has length {len(array)}, expected {n}")
        out[name] = array
    return out


def infer_length(states: dict[str, Any]) -> int | None:
    """Length implied by the first non-scalar entry, or ``None`` if all are scalar."""
    for value in states.values():
        array = np.asarray(value)
        if array.ndim > 0:
            return len(array)
    return None
