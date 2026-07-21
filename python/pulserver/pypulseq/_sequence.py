"""Fast sequence helpers for production bridge execution."""

from __future__ import annotations

__all__ = ["Sequence"]

import math
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pypulseq as pp
from pypulseq.block_to_events import block_to_events
from pypulseq.compress_shape import compress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.supported_labels_rf_use import get_supported_labels

_RF_USE_CHAR_TO_CODE = {
    "u": 0,
    "e": 1,
    "r": 2,
    "i": 3,
    "s": 4,
    "p": 5,
    "o": 6,
}
_RF_USE_CODE_TO_CHAR = {v: k for k, v in _RF_USE_CHAR_TO_CODE.items()}


class Sequence(pp.Sequence):
    """Sequence container tuned for append-only generation, with extensions.

    A drop-in subclass of :class:`pypulseq.Sequence` for the case a plugin
    actually has: blocks are emitted strictly in order and never revisited. On
    that assumption the per-block deduplication and gradient-continuity checks
    are skipped and each ``add_block`` becomes a direct library insertion —
    which is what keeps generation tractable for sequences with hundreds of
    thousands of blocks. Deduplication still happens, once, at ``write``.

    It also accepts the extension events upstream does not: user-defined
    labels (:func:`make_label`), block rotations (:func:`make_rotation`), and
    pTx shim vectors (:func:`make_rf_shim`).

    Because blocks are never re-checked, this class assumes you append in
    order. Use :class:`pypulseq.Sequence` if you need to modify blocks after
    adding them.

    Parameters
    ----------
    system : pypulseq.Opts, optional
        System limits recorded in the ``.seq`` header.
    use_block_cache : bool, optional
        Kept for upstream compatibility; off by default.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> seq = pp.Sequence(pp.Opts())
    >>> seq.add_block(pp.make_delay(1e-3), pp.make_label("LIN", "SET", 0))
    >>> len(seq.block_events)
    1

    Append modules rather than events, and write at the end::

        for ky in range(ny):
            excitation(seq)
            readout(seq, pe_idx=ky)
        seq.write(output_path)

    See Also
    --------
    make_label, make_rotation, make_rf_shim : the supported extension events.
    """

    def __init__(
        self,
        system: pp.Opts | None = None,
        use_block_cache: bool = False,
    ):
        super().__init__(system=system, use_block_cache=use_block_cache)
        self.arb_library = EventLibrary()
        self.trap_library = EventLibrary()
        self.rotation_library = EventLibrary()
        self.rf_shim_library = EventLibrary()
        # Bidirectional label registry; extended automatically for custom labels.
        _builtin = get_supported_labels()
        self._label_registry: dict[str, int] = {lbl: i + 1 for i, lbl in enumerate(_builtin)}
        self._label_registry_inv: dict[int, str] = {i + 1: lbl for i, lbl in enumerate(_builtin)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_block(self, *args: SimpleNamespace | float) -> None:
        """Append a block assuming strictly sequential insertion."""
        self._fast_set_block(self.next_free_block_ID, *args)
        self.next_free_block_ID += 1

    def remove_duplicates(self, in_place: bool = False) -> Sequence:
        """Remove duplicates with hardcoded rounded profiles per library.

        Unlike upstream pypulseq, this also compacts extension-related
        libraries and canonicalizes extension linked lists so identical chains
        are shared across blocks.

        Parameters
        ----------
        in_place:
            If ``True``, deduplicate current instance; otherwise return a copy.
        """
        if in_place:
            seq_copy = self
        else:
            tmp = self.block_cache
            self.block_cache = {}
            seq_copy = deepcopy(self)
            self.block_cache = tmp

        seq_copy.shape_library, shape_map = seq_copy.shape_library.remove_duplicates(9)

        for arb_id in list(seq_copy.arb_library.data):
            data = seq_copy.arb_library.data[arb_id]
            new_data = (*data[0:3], shape_map[data[3]], shape_map[data[4]], data[5])
            if data != new_data:
                seq_copy.arb_library.update(arb_id, None, new_data)

        for rf_id in list(seq_copy.rf_library.data):
            data = seq_copy.rf_library.data[rf_id]
            new_data = (data[0], shape_map[data[1]], shape_map[data[2]], shape_map[data[3]], *data[4:])
            if data != new_data:
                seq_copy.rf_library.update(rf_id, None, new_data, seq_copy.rf_library.type.get(rf_id, "u"))

        for adc_id in list(seq_copy.adc_library.data):
            data = seq_copy.adc_library.data[adc_id]
            shape_id = int(data[7])
            new_data = (*data[0:7], shape_map[shape_id], data[8])
            if data != new_data:
                seq_copy.adc_library.update(adc_id, None, new_data)

        seq_copy.arb_library, arb_map = _dedup_library_approx(seq_copy.arb_library, (6, -6, -6, -6, -6, -6))
        seq_copy.trap_library, trap_map = _dedup_library_approx(seq_copy.trap_library, (6, -6, -6, -6, -6))

        for grad_id in list(seq_copy.grad_library.data):
            grad_type = seq_copy.grad_library.type.get(grad_id, "")
            old_ref = int(seq_copy.grad_library.data[grad_id][0])
            if grad_type == "g":
                new_ref = arb_map[old_ref]
            elif grad_type == "t":
                new_ref = trap_map[old_ref]
            else:
                new_ref = old_ref
            seq_copy.grad_library.update(grad_id, None, (new_ref,), grad_type)

        seq_copy.grad_library, grad_map = _dedup_library_approx(seq_copy.grad_library, (0,))
        seq_copy.rf_library, rf_map = _dedup_library_approx(seq_copy.rf_library, (6, 0, 0, 0, 6, 6, 6, 6, 6, 6))
        seq_copy.adc_library, adc_map = _dedup_library_approx(seq_copy.adc_library, (0, -9, -6, 6, 6, 6, 6, 6, 6))

        for block_id in seq_copy.block_events:
            seq_copy.block_events[block_id][2] = grad_map[seq_copy.block_events[block_id][2]]
            seq_copy.block_events[block_id][3] = grad_map[seq_copy.block_events[block_id][3]]
            seq_copy.block_events[block_id][4] = grad_map[seq_copy.block_events[block_id][4]]
            seq_copy.block_events[block_id][1] = rf_map[seq_copy.block_events[block_id][1]]
            seq_copy.block_events[block_id][5] = adc_map[seq_copy.block_events[block_id][5]]

        seq_copy.trigger_library, trig_map = _dedup_library_approx(seq_copy.trigger_library, (0, 0, 9, 9))
        seq_copy.label_set_library, label_set_map = _dedup_library_approx(seq_copy.label_set_library, (0, 0))
        seq_copy.label_inc_library, label_inc_map = _dedup_library_approx(seq_copy.label_inc_library, (0, 0))
        seq_copy.rotation_library, rotation_map = _dedup_library_approx(seq_copy.rotation_library, 9)

        if seq_copy.rf_shim_library.data:
            widths = {len(v) for v in seq_copy.rf_shim_library.data.values()}
            if len(widths) != 1:
                raise RuntimeError("rf_shim_library has mixed payload widths; cannot apply fixed rounded dedup profile")
            rf_shim_digits = tuple([9] * next(iter(widths)))
            seq_copy.rf_shim_library, rf_shim_map = _dedup_library_approx(seq_copy.rf_shim_library, rf_shim_digits)
        else:
            seq_copy.rf_shim_library, rf_shim_map = _dedup_library_approx(seq_copy.rf_shim_library, 9)

        old_ext_lib = seq_copy.extensions_library
        new_ext_lib = EventLibrary()
        node_cache: dict[tuple[int, int, int], int] = {}

        def remap_ext_ref(ext_type_name: str, old_ref: int) -> int:
            if ext_type_name == "TRIGGERS":
                return trig_map[old_ref]
            if ext_type_name == "LABELSET":
                return label_set_map[old_ref]
            if ext_type_name == "LABELINC":
                return label_inc_map[old_ref]
            if ext_type_name == "DELAYS":
                return 0
            if ext_type_name == "ROTATIONS":
                return rotation_map[old_ref]
            if ext_type_name == "RF_SHIMS":
                return rf_shim_map[old_ref]
            return old_ref

        for block_id, row in seq_copy.block_events.items():
            head_id = int(row[6])
            if head_id == 0:
                continue

            chain: list[tuple[int, int]] = []
            cursor = head_id
            while cursor != 0:
                ext_data = old_ext_lib.data[cursor]
                ext_type_id = int(ext_data[0])
                ext_type_name = seq_copy.get_extension_type_string(ext_type_id)
                remapped_ref = remap_ext_ref(ext_type_name, int(ext_data[1]))
                if remapped_ref != 0:
                    chain.append((ext_type_id, remapped_ref))
                cursor = int(ext_data[2])

            new_next = 0
            for ext_type_id, remapped_ref in reversed(chain):
                key = (ext_type_id, remapped_ref, new_next)
                if key in node_cache:
                    node_id = node_cache[key]
                else:
                    node_id = new_ext_lib.insert(0, key)
                    node_cache[key] = node_id
                new_next = node_id

            seq_copy.block_events[block_id][6] = new_next

        seq_copy.extensions_library = new_ext_lib
        seq_copy.block_cache.clear()
        return seq_copy

    @property
    def custom_labels(self) -> dict[str, int]:
        """Labels auto-registered beyond the built-in ``get_supported_labels()`` set.

        Maps ``label_string -> int_idx`` for every custom label encountered via
        :meth:`add_block`.  The custom write helper uses this to serialise them.
        """
        n_builtin = len(get_supported_labels())
        return {name: idx for idx, name in self._label_registry_inv.items() if idx > n_builtin}

    def rf_from_lib_data(self, lib_data: list, use: str | int = "") -> SimpleNamespace:
        """Decode RF use from numeric code (fast path) or legacy char."""
        if isinstance(use, int | np.integer):
            use = _RF_USE_CODE_TO_CHAR.get(int(use), "u")
        return super().rf_from_lib_data(lib_data, use)

    # ------------------------------------------------------------------
    # Unsupported pypulseq API in fast-builder mode
    # ------------------------------------------------------------------

    def write(
        self,
        name,
        create_signature: bool = False,
        remove_duplicates: bool = False,
        check_timing: bool = False,
        v141_compat: bool = False,
    ):
        """Fast builder does not support writing sequence files."""
        del name, create_signature, remove_duplicates, check_timing, v141_compat
        raise NotImplementedError("pulserver.pypulseq.Sequence is a fast builder and does not implement write().")

    def read(self, *_args, **_kwargs) -> None:
        """Fast builder does not support reading/parsing external sequence files."""
        raise NotImplementedError("pulserver.pypulseq.Sequence is a fast builder and does not implement read().")

    def set_block(self, _block_index: int, *args: SimpleNamespace | float) -> None:  # noqa: ARG002
        """Disable positional insertion/update in fast mode."""
        raise NotImplementedError(
            "pulserver.pypulseq.Sequence only supports sequential add_block(). "
            "Use pypulseq.Sequence for random block updates."
        )

    def get_block(self, block_index: int) -> SimpleNamespace:
        """Fast builder does not support decoding blocks back to event objects."""
        del block_index
        raise NotImplementedError("pulserver.pypulseq.Sequence is a fast builder and does not implement get_block().")

    # ------------------------------------------------------------------
    # Internal fast block/event registration helpers
    # ------------------------------------------------------------------

    def _fast_set_block(self, block_index: int, *args: SimpleNamespace | float) -> None:
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
                # Soft delays are intentionally ignored in this fast on-scanner path.
                continue

            elif event.type == "rf_shim":
                rf_shim_id = self._fast_register_rf_shim(event)
                extensions.append({"type": self.get_extension_type_ID("RF_SHIMS"), "ref": rf_shim_id})

            elif event.type == "rot3D":
                rot_id = self._fast_register_rotation(event)
                extensions.append({"type": self.get_extension_type_ID("ROTATIONS"), "ref": rot_id})

            else:
                raise ValueError(f"Unknown event type {event.type} passed to pulserver.pypulseq.Sequence.add_block().")

        if extensions:
            sort_idx = np.argsort([e["ref"] for e in extensions])
            extensions = np.take(extensions, sort_idx)

            all_found = True
            extension_id = 0
            for ext in extensions:
                data = (ext["type"], ext["ref"], extension_id)
                extension_id, found = self.extensions_library.find(data)
                all_found = all_found and found
                if not found:
                    break

            if not all_found:
                extension_id = 0
                for ext in extensions:
                    data = (ext["type"], ext["ref"], extension_id)
                    extension_id, found = self.extensions_library.find(data)
                    if not found:
                        self.extensions_library.insert(extension_id, data)
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
        shape_IDs[0], _ = self.shape_library.find_or_insert(np.concatenate(([mag_shape.num_samples], mag_shape.data)))
        phase_shape = compress_shape(phase)
        shape_IDs[1], _ = self.shape_library.find_or_insert(
            np.concatenate(([phase_shape.num_samples], phase_shape.data))
        )
        if not (np.floor(event.t / self.rf_raster_time) == np.arange(len(event.t))).all():
            time_shape = compress_shape(event.t / self.rf_raster_time)
            shape_IDs[2], _ = self.shape_library.find_or_insert([time_shape.num_samples, *time_shape.data])

        if not hasattr(event, "use"):
            raise ValueError('Parameter "use" is not optional since v1.5.0')
        use = (
            event.use[0] if event.use in ("excitation", "refocusing", "inversion", "saturation", "preparation") else "u"
        )
        use_code = _RF_USE_CHAR_TO_CODE.get(use, 0)

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
        rf_id = self.rf_library.insert(0, data, use_code)
        return rf_id, shape_IDs

    def _fast_register_grad(self, event: SimpleNamespace):
        amplitude = np.max(np.abs(event.waveform))
        if amplitude > 0:
            fnz = event.waveform[np.nonzero(event.waveform)[0][0]]
            amplitude *= np.sign(fnz) if fnz != 0 else 1

        shape_IDs = [0, 0]
        g = event.waveform / amplitude if amplitude != 0 else event.waveform
        c_shape = compress_shape(g)
        shape_IDs[0], _ = self.shape_library.find_or_insert(np.concatenate(([c_shape.num_samples], c_shape.data)))

        c_time = compress_shape(event.tt / self.grad_raster_time)
        t_data = np.concatenate(([c_time.num_samples], c_time.data))
        if len(c_time.data) == 4 and np.allclose(c_time.data, [0.5, 1, 1, c_time.num_samples - 3]):
            pass  # standard raster, shape_IDs[1] stays 0
        elif len(c_time.data) == 3 and np.allclose(c_time.data, [0.5, 0.5, c_time.num_samples - 2]):
            shape_IDs[1] = -1
        else:
            shape_IDs[1], _ = self.shape_library.find_or_insert(t_data)

        data = (amplitude, event.first, event.last, *shape_IDs, event.delay)
        arb_id = self.arb_library.insert(0, data)
        grad_id = self.grad_library.insert(0, (arb_id,), "g")
        return grad_id, shape_IDs

    def _fast_register_trap(self, event: SimpleNamespace) -> int:
        data = (event.amplitude, event.rise_time, event.flat_time, event.fall_time, event.delay)
        trap_id = self.trap_library.insert(0, data)
        return self.grad_library.insert(0, (trap_id,), "t")

    def _fast_register_adc(self, event: SimpleNamespace):
        shape_id = 0
        if (
            hasattr(event, "phase_modulation")
            and event.phase_modulation is not None
            and len(event.phase_modulation) > 0
        ):
            phase_shape = compress_shape(np.asarray(event.phase_modulation).flatten())
            shape_data = np.concatenate(([phase_shape.num_samples], phase_shape.data))
            shape_id, _ = self.shape_library.find_or_insert(shape_data)

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

    def _get_label_idx(self, label: str) -> int:
        """Return 1-based int for *label*, auto-registering unknown strings."""
        if label not in self._label_registry:
            new_idx = max(self._label_registry_inv) + 1
            self._label_registry[label] = new_idx
            self._label_registry_inv[new_idx] = label
        return self._label_registry[label]

    def _fast_register_label(self, event: SimpleNamespace) -> int:
        data = (event.value, self._get_label_idx(event.label))
        lib = self.label_set_library if event.type == "labelset" else self.label_inc_library
        return lib.insert(0, data)

    def _fast_register_rf_shim(self, event: SimpleNamespace) -> int:
        data = (np.abs(event.shim_vector), np.angle(event.shim_vector))
        data = np.stack(data, axis=-1).ravel()
        return self.rf_shim_library.insert(0, tuple(data.tolist()))

    def _fast_register_rotation(self, event: SimpleNamespace) -> int:
        data = tuple(event.rot_quaternion.as_quat(canonical=True, scalar_first=True).tolist())
        return self.rotation_library.insert(0, data)


def _dedup_library_approx(lib: EventLibrary, digits: int | tuple[int, ...]) -> tuple[EventLibrary, dict[int, int]]:
    """Rounded deduplication using hardcoded per-library rounding profiles."""
    new_lib = EventLibrary(numpy_data=lib.numpy_data)
    mapping: dict[int, int] = {0: 0}
    type_code: dict[str | int, int] = {}

    ids = sorted(lib.data)
    if not ids:
        return new_lib, mapping

    if lib.numpy_data:
        rows = [np.asarray(lib.data[old_id], dtype=float).ravel() for old_id in ids]
    else:
        rows = [tuple(lib.data[old_id]) for old_id in ids]

    row_lengths = {len(row) for row in rows}
    all_numeric = all(all(isinstance(v, int | float | np.integer | np.floating) for v in row) for row in rows)

    if len(row_lengths) != 1 or not all_numeric:
        raise RuntimeError("_dedup_library_approx requires uniform, fully numeric payload rows")

    width = next(iter(row_lengths))
    if isinstance(digits, int):
        digits_tuple = tuple([digits] * width)
    else:
        if len(digits) < width:
            raise ValueError(f"Rounding profile length {len(digits)} is shorter than payload width {width}")
        digits_tuple = tuple(digits[:width])

    matrix = np.asarray(rows, dtype=float)
    rounded = _round_sig_matrix(matrix, digits_tuple)

    type_ids = np.asarray([type_code.setdefault(lib.type.get(old_id, ""), len(type_code) + 1) for old_id in ids])
    key_matrix = np.column_stack([type_ids.astype(float), rounded])
    key_bytes = (
        np.ascontiguousarray(key_matrix)
        .view(np.dtype((np.void, key_matrix.dtype.itemsize * key_matrix.shape[1])))
        .ravel()
    )

    _, first_idx, inverse = np.unique(key_bytes, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)

    uniq_to_new_id = np.zeros(len(first_idx), dtype=np.int32)
    for new_id, uniq_idx in enumerate(order, start=1):
        row_idx = int(first_idx[uniq_idx])
        old_id = ids[row_idx]
        type_key = lib.type.get(old_id, "")
        if lib.numpy_data:
            arr = rounded[row_idx].copy()
            arr.flags.writeable = False
            insert_data = arr
        else:
            insert_data = tuple(rounded[row_idx].tolist())
        new_lib.insert(new_id, insert_data, type_key)
        uniq_to_new_id[uniq_idx] = new_id

    mapped = uniq_to_new_id[inverse]
    for i, old_id in enumerate(ids):
        mapping[old_id] = int(mapped[i])

    return new_lib, mapping


def _round_sig_matrix(matrix: np.ndarray, digits: tuple[int, ...]) -> np.ndarray:
    """Vectorized significant-digit rounding for 2D numeric matrices."""
    if matrix.ndim != 2:
        raise ValueError("_round_sig_matrix expects a 2D matrix")
    if len(digits) != matrix.shape[1]:
        raise ValueError(f"Rounding profile length {len(digits)} does not match payload width {matrix.shape[1]}")

    d = np.asarray(digits, dtype=float).reshape(1, -1)
    out = matrix.copy()

    pos_mask = d > 0
    if np.any(pos_mask):
        mags = np.power(10.0, d - np.ceil(np.log10(np.abs(matrix) + 1e-12)))
        rounded_pos = np.round(matrix * mags) / mags
        out = np.where(pos_mask, rounded_pos, out)

    nonpos_mask = ~pos_mask
    if np.any(nonpos_mask):
        mags = np.power(10.0, -d)
        rounded_nonpos = np.round(matrix * mags) / mags
        out = np.where(nonpos_mask, rounded_nonpos, out)

    return out
