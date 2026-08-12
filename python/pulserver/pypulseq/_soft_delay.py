"""Soft delays: reading their defaults, and applying named values."""

from __future__ import annotations

import warnings

import numpy as np

from . import _results


class SoftDelayMixin:
    """Soft-delay defaults and application. Mixed into :class:`Sequence`."""

    def apply_soft_delay(self, **kwargs: float) -> None:
        """Set named soft delays, rewriting the block durations that carry them.

        Each block holding a soft delay named in ``kwargs`` gets the duration
        ``value / factor + offset``, rounded to the block-duration raster.

        Parameters
        ----------
        **kwargs
            Soft-delay hints and their values in seconds -- ``TE=40e-3``. Only
            the delays named are touched; the sequence may hold others.

        Raises
        ------
        ValueError
            If a named hint is not in the sequence, if a hint and a numeric ID
            disagree with an earlier block's pairing of the two, or if a value
            works out to a negative duration.

        Notes
        -----
        Upstream's arithmetic and upstream's diagnostics, on this class's
        blocks. A soft delay is a *design-time* parameter -- the sequence
        arrives at the scanner with its durations already resolved -- so this
        is for a script that builds one sequence and writes several timings of
        it, not for anything on the interpreter's path.

        See Also
        --------
        get_default_soft_delay_values : the values already in the blocks.
        """
        raster = float(self.system.block_duration_raster)
        seen: dict[str, int] = {}
        warned: set[int] = set()

        for index in range(1, self.num_blocks + 1):
            delay = getattr(self.get_block(index), "soft_delay", None)
            if delay is None:
                continue

            _check_soft_delay_consistency(delay, index, seen)
            if delay.hint not in kwargs:
                continue

            exact = float(kwargs[delay.hint]) / delay.factor + delay.offset
            rounded = round(exact / raster) * raster
            if abs(rounded - exact) > 0.5e-6 and delay.numID not in warned:
                warnings.warn(
                    f"Soft delay '{delay.hint}' in block {index}: duration rounded by "
                    f"{abs(rounded - exact) * 1e6:.1f} us to align with the block raster "
                    f"({raster * 1e6:.1f} us). Reported once per soft delay.",
                    stacklevel=2,
                )
                warned.add(delay.numID)
            if rounded < 0:
                raise ValueError(
                    f"Soft delay '{delay.hint}' in block {index}: the value works out to a "
                    f"negative duration ({rounded * 1e6:.1f} us). Check the offset "
                    f"({delay.offset * 1e6:.1f} us) and factor ({delay.factor})."
                )
            self._native.set_block_duration(index, rounded)

        unknown = [hint for hint in kwargs if hint not in seen]
        if unknown:
            available = sorted(seen) or ["none"]
            raise ValueError(
                f"apply_soft_delay(): {unknown} not in the sequence. "
                f"Available soft delays: {available}"
            )
        self._touch()

    def get_default_soft_delay_values(self) -> tuple[dict, list[str]]:
        """The soft-delay values the sequence was built with, and what is wrong.

        Returns
        -------
        values : dict
            Hint to the delay value in seconds implied by the block durations
            as they stand -- ``(duration - offset) * factor``, the inverse of
            what :meth:`apply_soft_delay` writes. This is what a user
            interface offers as the starting point of each slider.
        report : list of str
            One line per inconsistency found. Empty when the sequence is sound.

        Notes
        -----
        MATLAB's ``getDefaultSoftDelayValues``, which PyPulseq has no
        equivalent of. Two of its three outputs: the ``softDelayState`` cell
        array is MATLAB's intermediate, and its ``min``/``max`` are carried on
        the dict's values here instead --- each is a
        :class:`~._results.SoftDelay`, whose ``value`` is what the mapping
        compares and prints, so the dict reads as MATLAB's ``easyStruct``
        while still carrying the range the delay may be set over.

        The limits come from the duration reaching zero, which is the only
        bound the sequence itself states; a delay that would collide with the
        events inside its block is not caught here, and is not caught by
        MATLAB either.
        """
        report: list[str] = []
        seen: dict[str, int] = {}
        state: dict[int, dict] = {}

        for index in range(1, self.num_blocks + 1):
            delay = getattr(self.get_block(index), "soft_delay", None)
            if delay is None:
                continue

            if delay.numID < 0:
                report.append(f"Block {index}: soft delay '{delay.hint}' has a negative numeric ID")
                continue
            if delay.factor == 0:
                report.append(
                    f"Block {index}: soft delay '{delay.hint}'/{delay.numID} has factor 0"
                )
                continue
            try:
                _check_soft_delay_consistency(delay, index, seen)
            except ValueError as problem:
                report.append(str(problem))

            default = (float(self.block_durations[index - 1]) - delay.offset) * delay.factor
            # A duration of zero is the only limit the sequence itself states;
            # which side of the range it is depends on the sign of the factor.
            limit = -delay.offset * delay.factor
            held = state.get(delay.numID)
            if held is None:
                state[delay.numID] = {
                    "hint": delay.hint,
                    "value": default,
                    "block": index,
                    "minimum": limit if delay.factor > 0 else 0.0,
                    "maximum": np.inf if delay.factor > 0 else limit,
                }
                continue

            if abs(default - held["value"]) > 1e-7:
                report.append(
                    f"Block {index}: soft delay '{delay.hint}'/{delay.numID} implies "
                    f"{default * 1e6:.1f} us, against {held['value'] * 1e6:.1f} us from block "
                    f"{held['block']}"
                )
            if delay.factor > 0:
                held["minimum"] = max(held["minimum"], limit)
            else:
                held["maximum"] = min(held["maximum"], limit)

        values: dict[str, _results.SoftDelay] = {}
        for numeric in sorted(state):
            held = state[numeric]
            values[held["hint"]] = _results.SoftDelay(
                hint=held["hint"],
                numeric_id=numeric,
                value=held["value"],
                minimum=held["minimum"],
                maximum=held["maximum"],
            )
        return values, report


def _check_soft_delay_consistency(delay, index: int, seen: dict[str, int]) -> None:
    """One hint means one numeric ID and back, across the whole sequence."""
    known = seen.setdefault(delay.hint, delay.numID)
    if known != delay.numID:
        raise ValueError(
            f"Block {index}: soft delay '{delay.hint}' has numeric ID {delay.numID}, "
            f"against {known} where the same hint appeared before"
        )
    for hint, numeric in seen.items():
        if numeric == delay.numID and hint != delay.hint:
            raise ValueError(
                f"Block {index}: soft delay numeric ID {delay.numID} is called "
                f"'{delay.hint}' here and '{hint}' earlier"
            )

