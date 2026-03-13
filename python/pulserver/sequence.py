"""Fast sequence helpers for production bridge execution."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pypulseq as pp
from pypulseq.block_to_events import block_to_events
from pypulseq.compress_shape import compress_shape
from pypulseq.Sequence.sequence import Sequence
from pypulseq.supported_labels_rf_use import get_supported_labels


class Sequence(pp.Sequence):
    """Sequential-only Sequence variant with reduced per-block overhead.

    Intended for production generation where events are always appended in order.
    Per-block deduplication and gradient continuity checks are skipped entirely;
    each call to add_block() performs direct library insertion.
    """

    def __init__(
        self,
        system : pp.Opts | None = None,
        use_block_cache: bool = False,
    ):
        super().__init__(system=system, use_block_cache=use_block_cache)

    def add_block(self, *args: SimpleNamespace | float) -> None:
        """Append a block assuming strictly sequential insertion."""
        self._fast_set_block(self.next_free_block_ID, *args)
        self.next_free_block_ID += 1

    def set_block(self, _block_index: int, *args : SimpleNamespace | float) -> None:  # noqa: ARG002
        """Disable positional insertion/update in fast mode."""
        raise NotImplementedError(
            "FastSequence only supports sequential add_block(). " "Use pypulseq.Sequence for random block updates."
        )

    def _fast_set_block(self, block_index: int, *args : SimpleNamespace | float) -> None:
        """Direct-insert block registration: no dedup, no continuity checks, no trace."""
        events = block_to_events(*args)
        new_block = np.zeros(7, dtype=np.int32)
        duration = 0
        extensions = []

        for event in events:
            if isinstance(event, float):
                duration = max(duration, event)
                continue

            if event.type == "rf":
                rf_id, _ = self._fast_register_rf(event)
                new_block[1] = rf_id
                duration = max(duration, event.shape_dur + event.delay + event.ringdown_time)

            elif event.type == "grad":
                channel_num = ["x", "y", "z"].index(event.channel)
                grad_id, _ = self._fast_register_grad(event)
                new_block[2 + channel_num] = grad_id
                grad_duration = (
                    event.delay + math.ceil(event.tt[-1] / self.grad_raster_time - 1e-10) * self.grad_raster_time
                )
                duration = max(duration, grad_duration)

            elif event.type == "trap":
                channel_num = ["x", "y", "z"].index(event.channel)
                new_block[2 + channel_num] = self._fast_register_trap(event)
                duration = max(duration, event.delay + event.rise_time + event.flat_time + event.fall_time)

            elif event.type == "adc":
                adc_id, _ = self._fast_register_adc(event)
                new_block[5] = adc_id
                duration = max(duration, event.delay + event.num_samples * event.dwell + event.dead_time)

            elif event.type == "delay":
                duration = max(duration, event.delay)

            elif event.type in ("output", "trigger"):
                event_id = self._fast_register_control(event)
                extensions.append({"type": self.get_extension_type_ID("TRIGGERS"), "ref": event_id})
                duration = max(duration, event.delay + event.duration)

            elif event.type in ("labelset", "labelinc"):
                label_id = self._fast_register_label(event)
                extensions.append({"type": self.get_extension_type_ID(event.type.upper()), "ref": label_id})

            elif event.type == "soft_delay":
                sd_id = self._fast_register_soft_delay(event)
                extensions.append({"type": self.get_extension_type_ID("DELAYS"), "ref": sd_id})
                duration = max(duration, event.default_duration)

            else:
                raise ValueError(f"Unknown event type {event.type} passed to FastSequence.add_block().")

        if extensions:
            sort_idx = np.argsort([e["ref"] for e in extensions])
            extensions = np.take(extensions, sort_idx)
            extension_id = 0
            for ext in extensions:
                data = (ext["type"], ext["ref"], extension_id)
                extension_id = self.extensions_library.insert(0, data)
            new_block[6] = extension_id

        self.block_events[block_index] = new_block
        self.block_durations[block_index] = float(duration)

    # ------------------------------------------------------------------
    # Private direct-insert helpers (no find_or_insert, no dedup)
    # ------------------------------------------------------------------

    def _fast_register_rf(self, event: SimpleNamespace):
        mag = np.abs(event.signal)
        amplitude = np.max(mag)
        mag = mag / amplitude
        mag[np.isnan(mag)] = 0
        phase = np.angle(event.signal)
        phase[phase < 0] += 2 * np.pi
        phase /= 2 * np.pi

        shape_IDs = [0, 0, 0]
        mag_shape = compress_shape(mag)
        shape_IDs[0] = self.shape_library.insert(0, np.concatenate(([mag_shape.num_samples], mag_shape.data)))
        phase_shape = compress_shape(phase)
        shape_IDs[1] = self.shape_library.insert(0, np.concatenate(([phase_shape.num_samples], phase_shape.data)))
        if not (np.floor(event.t / self.rf_raster_time) == np.arange(len(event.t))).all():
            time_shape = compress_shape(event.t / self.rf_raster_time)
            shape_IDs[2] = self.shape_library.insert(0, [time_shape.num_samples, *time_shape.data])

        if not hasattr(event, "use"):
            raise ValueError('Parameter "use" is not optional since v1.5.0')
        use = (
            event.use[0] if event.use in ("excitation", "refocusing", "inversion", "saturation", "preparation") else "u"
        )

        data = (
            amplitude,
            *shape_IDs,
            event.center,
            event.delay,
            event.freq_ppm,
            event.phase_ppm,
            event.freq_offset,
            event.phase_offset,
        )
        rf_id = self.rf_library.insert(0, data, use)
        return rf_id, shape_IDs

    def _fast_register_grad(self, event: SimpleNamespace):
        amplitude = np.max(np.abs(event.waveform))
        if amplitude > 0:
            fnz = event.waveform[np.nonzero(event.waveform)[0][0]]
            amplitude *= np.sign(fnz) if fnz != 0 else 1

        shape_IDs = [0, 0]
        g = event.waveform / amplitude if amplitude != 0 else event.waveform
        c_shape = compress_shape(g)
        shape_IDs[0] = self.shape_library.insert(0, np.concatenate(([c_shape.num_samples], c_shape.data)))

        c_time = compress_shape(event.tt / self.grad_raster_time)
        t_data = np.concatenate(([c_time.num_samples], c_time.data))
        if len(c_time.data) == 4 and np.allclose(c_time.data, [0.5, 1, 1, c_time.num_samples - 3]):
            pass  # standard raster, shape_IDs[1] stays 0
        elif len(c_time.data) == 3 and np.allclose(c_time.data, [0.5, 0.5, c_time.num_samples - 2]):
            shape_IDs[1] = -1
        else:
            shape_IDs[1] = self.shape_library.insert(0, t_data)

        data = (amplitude, event.first, event.last, *shape_IDs, event.delay)
        grad_id = self.grad_library.insert(0, data, "g")
        return grad_id, shape_IDs

    def _fast_register_trap(self, event: SimpleNamespace) -> int:
        data = (event.amplitude, event.rise_time, event.flat_time, event.fall_time, event.delay)
        return self.grad_library.insert(0, data, "t")

    def _fast_register_adc(self, event: SimpleNamespace):
        shape_id = 0
        if (
            hasattr(event, "phase_modulation")
            and event.phase_modulation is not None
            and len(event.phase_modulation) > 0
        ):
            phase_shape = compress_shape(np.asarray(event.phase_modulation).flatten())
            shape_data = np.concatenate(([phase_shape.num_samples], phase_shape.data))
            shape_id = self.shape_library.insert(0, shape_data)

        data = (
            event.num_samples,
            event.dwell,
            max(event.delay, event.dead_time),
            event.freq_ppm,
            event.phase_ppm,
            event.freq_offset,
            event.phase_offset,
            shape_id,
            event.dead_time,
        )
        adc_id = self.adc_library.insert(0, data)
        return adc_id, shape_id

    def _fast_register_control(self, event: SimpleNamespace) -> int:
        event_type = ["output", "trigger"].index(event.type)
        event_channel = (["osc0", "osc1", "ext1"] if event_type == 0 else ["physio1", "physio2"]).index(event.channel)
        data = (event_type + 1, event_channel + 1, event.delay, event.duration)
        return self.trigger_library.insert(0, data)

    def _fast_register_label(self, event: SimpleNamespace) -> int:
        label_idx = get_supported_labels().index(event.label) + 1
        data = (event.value, label_idx)
        lib = self.label_set_library if event.type == "labelset" else self.label_inc_library
        return lib.insert(0, data)

    def _fast_register_soft_delay(self, event: SimpleNamespace) -> int:
        if event.hint in self.soft_delay_hints:
            assigned_numID = self.soft_delay_hints[event.hint]
            if event.numID is not None and event.numID != assigned_numID:
                raise ValueError(
                    f"Soft delay hint '{event.hint}' is already assigned to numID {assigned_numID}. "
                    f"Cannot use numID {event.numID}."
                )
            event.numID = assigned_numID
        else:
            if event.numID is None:
                event.numID = max([-1, *self.soft_delay_hints.values()]) + 1
            elif event.numID in self.soft_delay_hints.values():
                existing_hint = next(h for h, n in self.soft_delay_hints.items() if n == event.numID)
                raise ValueError(f"numID {event.numID} is already used by soft delay '{existing_hint}'.")
            self.soft_delay_hints[event.hint] = event.numID
        data = (event.numID, event.offset, event.factor, event.hint)
        return self.soft_delay_library.insert(0, data)
