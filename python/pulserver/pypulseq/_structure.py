"""The scan structure of a sequence, as the C safety core recovers it.

Derived on demand and cached on the sequence until its blocks change, since
every TR-scoped analysis -- PNS, gradient spectrum, ``plot(tr=)`` -- asks the
same question of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _safety

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._sequence import Sequence

class _Structure:
    """The scan structure of one sequence, as the C safety core sees it.

    Building this is the expensive half of every TR-based analysis: the
    sequence is serialised, parsed back by the C library, and its repetition
    time and segmentation recovered. Everything after that -- extracting a TR
    waveform, the acoustic line spectrum, PNS -- is comparatively cheap and
    runs against the collection held here, so the cost is paid once per
    sequence rather than once per question asked about it.

    Held by :class:`Sequence` behind a revision number, and rebuilt rather
    than updated when the sequence changes. A structure that has silently
    outlived its sequence would answer about blocks it no longer holds.
    """

    def __init__(self, sequence: Sequence) -> None:
        from .._ext._pulseg_wrapper import _find_tr, _PulseqCollection

        self.system = sequence.system
        system = self.system
        # Binary, not text: this is written only to be parsed straight back,
        # so the number formatting and the sscanf that the text format would
        # spend on both ends are pure waste -- and the six significant digits
        # `%12g` rounds a gradient amplitude to would be a needless loss on
        # the way into a safety analysis.
        #
        # Serialising compresses shapes and republishes definitions, which
        # touches the C++ sequence -- so read the revision after it, not
        # before, or the cache is stale the moment it is built.
        written = sequence._to_binary()
        self.revision = sequence._revision
        self.collection = _PulseqCollection(
            [written],
            float(system.gamma),
            float(system.B0),
            float(system.max_grad),
            float(system.max_slew),
            float(system.rf_raster_time),
            float(system.grad_raster_time),
            float(system.adc_raster_time),
            float(system.block_duration_raster),
            True,
            1,
        )
        self.tr = _find_tr(self.collection)
        self._segments: list | None = None

    @property
    def tr_duration(self) -> float:
        """float : The canonical TR, in seconds."""
        return float(self.tr["tr_duration_us"]) * 1e-6

    @property
    def num_trs(self) -> int:
        """int : How many structural TRs the repeating region holds."""
        return int(self.tr["num_trs"])

    @property
    def folded_prep_or_cooldown(self) -> bool:
        """bool : Whether a leading or trailing section is *not* just another TR.

        A prep or cooldown region that repeats the TR pattern is folded away
        by the C core -- its blocks are TR instances like any other, merely
        discarded downstream. Only a genuinely different one survives here,
        and when it does the core's unit of analysis stops being the TR and
        becomes the whole pass it sits in.
        """
        return bool(self.tr["num_prep_blocks"] and not self.tr["degenerate_prep"]) or bool(
            self.tr["num_cooldown_blocks"] and not self.tr["degenerate_cooldown"]
        )

    @property
    def num_instances(self) -> int:
        """int : How many things ``tr=<int>`` can name.

        Passes when a non-degenerate section folds prep and cooldown into the
        unit of analysis (see :attr:`folded_prep_or_cooldown`), and otherwise
        every TR the scanner plays -- the prep and cooldown TRs included, and
        counted once per average, which is how the C core indexes them.
        """
        if self.folded_prep_or_cooldown:
            return max(int(self.tr["num_passes"]), 1)
        averages = max(int(self.tr["num_averages"]), 1)
        return (
            int(self.tr["num_prep_trs"])
            + averages * self.num_trs
            + int(self.tr["num_cooldown_trs"])
        )

    @property
    def segments(self) -> list:
        """list : The segment layout, resolved to each segment's max-energy instance.

        Reporting only. Neither the acoustic nor the PNS analysis selects a
        segment instance -- both run on the per-sample maximum over every TR
        instance -- so this says which blocks a reader should look at, not
        which blocks were analysed.
        """
        if self._segments is None:
            from .._ext._pulseg_wrapper import _get_segments

            self._segments = _get_segments(self.collection, 0)
        return self._segments

    def waveform(self, tr, *, rf_channel: int = 0) -> _safety.TRSequence:
        """The TR ``tr`` names, as a sequence upstream can read.

        Everything in it -- gradients, RF magnitude and phase, ADC events,
        block boundaries -- comes out of one ``pulseg_get_tr_waveforms`` call,
        so what is returned is the canonical TR the C core itself constructs
        rather than one rebuilt from the timeline.

        Parameters
        ----------
        tr : {"worst_case", "zero_variable"} or int
            ``"worst_case"`` for the per-sample maximum over every instance --
            the waveform ``pulseg_check_safety`` judges, which is not any one
            TR the scanner plays. ``"zero_variable"`` for the same TR with the
            gradients that vary between instances zeroed, leaving the
            structure the constant ones make. An integer for that instance as
            it really plays, signed amplitudes and all.
        rf_channel : int, default 0
            Which transmit channel the returned blocks report, for a pTx TR.
        """
        from .._ext._pulseg_wrapper import _get_tr_waveforms

        mode, index = self.resolve(tr)
        return _safety.TRSequence.from_c(
            _get_tr_waveforms(self.collection, 0, mode, index),
            self.system,
            rf_channel=rf_channel,
        )

    def resolve(self, tr) -> tuple[int, int]:
        """``tr`` as the ``(amplitude_mode, tr_index)`` pair the binding takes."""
        if isinstance(tr, str):
            if tr not in _safety.AMPLITUDE_MODES:
                raise ValueError(
                    f"tr must be an integer or one of {sorted(_safety.AMPLITUDE_MODES)}, got {tr!r}"
                )
            return _safety.AMPLITUDE_MODES[tr], 0

        index = int(tr)
        # The core's own bound, which is passes when prep or cooldown is
        # folded into the unit of analysis and TR instances otherwise --
        # not num_trs, which counts only the structural repeat.
        if not 0 <= index < self.num_instances:
            named = "passes" if self.folded_prep_or_cooldown else "TR instances"
            raise ValueError(
                f"tr={index} is out of range; the sequence holds {self.num_instances} {named}"
            )
        return _safety.AMPLITUDE_MODES["actual"], index


def _worst_window(durations: np.ndarray, values: np.ndarray, width: float) -> float:
    """The largest sum of ``values`` over any window of at most ``width`` seconds.

    MATLAB grows the window by one block and then trims from the front while
    it is too wide, taking its maximum *before* the trim -- so the window that
    wins may be up to one block wider than asked for, which is its documented
    "rounded up to a certain number of complete blocks". Reproduced here as
    two cumulative sums and a search rather than a loop, which is the same
    answer without MATLAB's index bug (see :meth:`Sequence.calc_rf_power`).
    """
    if values.size == 0:
        return 0.0

    elapsed = np.concatenate(([0.0], np.cumsum(durations)))
    accumulated = np.concatenate(([0.0], np.cumsum(values)))
    # Window entering block i spans blocks `start(i)..i`, where start is the
    # front the trim left after block i-1: the earliest block whose tail is
    # still within `width` of the end of block i-1.
    starts = np.searchsorted(elapsed, elapsed[:-1] - width, side="left")
    return float((accumulated[1:] - accumulated[starts]).max())


