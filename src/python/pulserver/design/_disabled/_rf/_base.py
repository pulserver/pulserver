"""Common stateful collection protocol for RF and preparation modules."""

from __future__ import annotations

import copy
from math import isfinite
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from ..._core._module import Block, SequenceModule, _block_payload, _peak
from .._rotation import normalize_rotation
from .._system import copy_event

_ModuleT = TypeVar("_ModuleT", bound="RfModule")
_PulseT = TypeVar("_PulseT", bound="RfPulse")


@dataclass(frozen=True)
class RfState:
    """Dynamic offsets applied to an RF module's immutable templates."""

    freq_offset_hz: float = 0.0
    phase_offset_rad: float = 0.0
    amplitude_scale: float = 1.0
    rotation: object | None = None


class RfModule(SequenceModule):
    """Stateful RF module exposed as an immutable sequence of Pulseq blocks.

    Factory-time events are retained as templates. ``set_state`` invalidates
    the cached block snapshot; the next collection operation deep-copies the
    templates and applies one global state while preserving relative values
    inside composite preparations.
    """

    def __init__(self, system, blocks: Sequence[Block], **declared) -> None:
        super().__init__(system, **declared)
        self._template_blocks = tuple(tuple(block) for block in blocks if block)
        if not self._template_blocks:
            raise ValueError("an RF module must contain at least one block")
        self._state = RfState()
        self._block_cache: tuple[Block, ...] | None = None
        self._scaled_signals: dict[tuple[int, float], tuple] = {}

    def _set_state(
        self: _ModuleT,
        *,
        freq_offset_hz: float = 0.0,
        phase_offset_rad: float = 0.0,
        amplitude_scale: float = 1.0,
        rotation=None,
    ) -> _ModuleT:
        """Replace frequency, phase, amplitude, and explicit rotation state."""
        # math.isfinite, not np.isfinite: this runs once per shot and the
        # numpy call builds a scalar array per value to answer a question about
        # a Python float.
        values = (freq_offset_hz, phase_offset_rad, amplitude_scale)
        if not all(isfinite(value) for value in values):
            raise ValueError("RF state values must be finite")
        if amplitude_scale < 0:
            raise ValueError("amplitude_scale must be >= 0")
        self._state = RfState(
            freq_offset_hz=float(freq_offset_hz),
            phase_offset_rad=float(phase_offset_rad),
            amplitude_scale=float(amplitude_scale),
            rotation=normalize_rotation(rotation),
        )
        self._block_cache = None
        return self

    def _build_blocks(self) -> tuple[Block, ...]:
        state = self._state
        return self._retuned_blocks(
            # Only a rotation is structural: it appends an event to every
            # gradient-bearing block. Frequency, phase and amplitude are all
            # payload fields of an RF event whose shapes are already registered.
            (state.rotation,),
            self._render,
            lambda record: self._retune(record, self._state),
        )

    def _render(self, record) -> tuple[Block, ...]:
        blocks: list[Block] = []
        state = self._state
        pulses: list[tuple] = []
        for template in self._template_blocks:
            block = [copy_event(event) for event in template]
            has_gradient = False
            for source, event in zip(template, block, strict=True):
                event_type = getattr(event, "type", None)
                if event_type == "rf":
                    self._apply_scale(event, source.signal, state.amplitude_scale)
                    event.freq_offset += state.freq_offset_hz
                    event.phase_offset += state.phase_offset_rad
                    # The template alongside it: a later state is an absolute
                    # offset from the *template's* value, not a further
                    # increment on the last one's.
                    pulses.append((event, source))
                elif event_type in ("grad", "trap"):
                    has_gradient = True
            if has_gradient and state.rotation is not None:
                block.append(copy_event(state.rotation))
            blocks.append(tuple(block))
        record["pulses"] = tuple(pulses)
        return tuple(blocks)

    def _retune(self, record, state) -> None:
        """Re-point every RF event at this state, without rebuilding a block."""
        scale = state.amplitude_scale
        freq_offset_hz = state.freq_offset_hz
        phase_offset_rad = state.phase_offset_rad
        for event, source in record["pulses"]:
            self._apply_scale(event, source.signal, scale)
            event.freq_offset = source.freq_offset + freq_offset_hz
            event.phase_offset = source.phase_offset + phase_offset_rad

    def _apply_scale(self, event, template_signal, scale: float) -> None:
        """Point ``event`` at ``template_signal * scale``, and say where it came from.

        ``signal`` stays the real scaled envelope, because everything that
        plots or integrates a pulse reads it. ``shape_source`` additionally
        names the *unscaled* envelope and the factor, which is what lets the
        sequence writer register one magnitude shape for a whole flip-angle
        schedule instead of one per distinct flip -- see
        :meth:`pulserver.pypulseq.Sequence._compress_rf_cached`.
        """
        event.signal = template_signal if scale == 1.0 else self._scaled_signal(template_signal, scale)
        event.shape_source = (template_signal, scale)

    def _scaled_signal(self, signal, scale: float):
        """``signal * scale``, handing back the same array for a repeated scale.

        Amplitude schedules revisit a small set of factors -- an echo-train
        flip schedule, a preparation module -- so keeping one array per factor
        lets the sequence writer reuse its registered shape rather than
        recompressing an identical waveform once per shot.

        Unscaled state does not come through here at all: ``* 1.0`` would
        allocate a copy of the template for no change, and the template array
        is never written into (see
        :func:`pulserver.design._system.copy_event`).
        """
        key = (id(signal), scale)
        cached = self._scaled_signals.get(key)
        if cached is None or cached[0] is not signal:
            cached = (signal, np.asarray(signal) * scale)
            self._scaled_signals[key] = cached
        return cached[1]

    def _current_blocks(self) -> tuple[Block, ...]:
        if self._block_cache is None:
            self._block_cache = self._build_blocks()
        return self._block_cache

    def _direct_payloads(self):
        """This module's payloads, rewritten in place rather than re-derived.

        An RF module's state moves exactly three numbers, and all three are
        arithmetic on the *template's* values: the transmit frequency and phase
        are offsets from it, and the registered amplitude is its envelope's peak
        times the scale -- the same product
        ``Sequence._compress_rf_cached`` registers, which is why the peak is
        taken from the unscaled envelope here rather than from the scaled one.
        Nothing has to be scaled, copied or re-measured to know any of them.

        A rotation is declined: it appends an event to every gradient block, so
        it is a change of structure and the generic path owns it.
        """
        state = self._state
        if state.rotation is not None:
            return None

        cache = self._payload_cache
        if cache is None:
            blocks = self._rendered_blocks()
            payloads, entries = [], []
            peaks = self._peak_cache = {}
            for block, template in zip(blocks, self._template_blocks, strict=True):
                payload = _block_payload(block, peaks)
                if "rf" in payload:
                    # The base has to come off the *template* event, the way
                    # ``_render`` and ``_retune`` take it. Reading it off the
                    # rendered event instead looks equivalent only because the
                    # cache is normally built at the zero state: anything that
                    # drops the cache mid-scan -- naming a new label does --
                    # would rebuild it against an event this state's
                    # offsets are already in, and then add them a second time.
                    # Silent, and a wrong file: doubled RF phase, and doubled
                    # slice frequency offset on a multi-slice 2D scan.
                    source = next(event for event in template if getattr(event, "type", None) == "rf")
                    values = list(payload["rf"])
                    payload["rf"] = values
                    entries.append((values, _peak(source.signal, peaks),
                                    source.freq_offset, source.phase_offset))
                payloads.append(payload)
            cache = self._payload_cache = (tuple(payloads), tuple(entries))

        payloads, entries = cache
        for values, peak, freq, phase in entries:
            values[0] = peak * state.amplitude_scale
            values[1] = freq + state.freq_offset_hz
            values[2] = phase + state.phase_offset_rad
        return payloads

    @property
    def state(self) -> RfState:
        return self._state


class RfPulse(RfModule):
    """One RF/encoding block followed by zero or more rephasing events."""

    def __init__(self, system, rf, gradients: Sequence = (), rephasers: Sequence = (), **declared) -> None:
        self.rf = rf
        self.gradients = tuple(gradients)
        self.rephasers = tuple(rephasers)
        blocks: list[Block] = [(rf, *self.gradients)]
        if self.rephasers:
            blocks.append(self.rephasers)
        super().__init__(system, blocks, **declared)

    def without_rephasers(self: _PulseT) -> _PulseT:
        """Return a copy that stops playing its own rephasing block.

        The counterpart to a readout's ``slice_rephasing``: once the readout
        folds the rephaser into its prewinder, the excitation must not also
        play it, or the slice is rephased twice and the shot ends with a net
        selection moment of exactly one rephaser.

        The rephasers stay readable on the returned module, so the pairing is
        one expression::

            readout = design.make_line_readout(
                system, fov, matrix, slice_rephasing=excitation.rephasers
            )
            excitation = excitation.without_rephasers()

        Returns
        -------
        RfPulse
            A new module with the same RF and selection gradients and an
            empty rephasing block. ``self`` is left unchanged.
        """
        if not self.rephasers:
            return self
        clone = copy.copy(self)
        clone._template_blocks = ((self.rf, *self.gradients),)
        clone._block_cache = None
        clone._invalidate()
        return clone
