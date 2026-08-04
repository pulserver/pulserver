"""A Pulseq sequence that can hold the two extensions PyPulseq 1.5.0 cannot.

Block rotations and pTx shim vectors are part of the Pulseq 1.5.1 file format
and of how this interpreter plays non-Cartesian and parallel-transmit scans, but
upstream PyPulseq has neither a library to keep them in nor a writer that emits
them. Rather than fork `pp.Sequence`, this module wraps one: `_seq` stays an
ordinary PyPulseq sequence and does everything it already does, and the two
libraries it lacks live beside it.

That leaves rotations and shims looking exactly like triggers and labels --
extension events, registered into a library, chained onto a block -- which is
what lets `PeriodicSequence` vary them per iteration with no special case at
all.

Three things are this module's own rather than delegated:

`write` is the 1.5.1 writer. It emits `ROTATIONS` and `RF_SHIMS`, and label
names outside PyPulseq's built-in set, and raises the file's revision to match
what it actually used.

`read` is its inverse, and reads plain 1.5.0 files too. Whatever upstream
understands is read by upstream; the sections it would refuse are parsed here
and put back afterwards.

`plot` shows the sequence as it will play rather than as it is stored: rotations
become rotated waveforms and a shimmed RF pulse becomes its channels, because a
quaternion in an extension library is not something anybody can look at.
"""

from __future__ import annotations

__all__ = ['Sequence', 'make_rf_shim', 'make_rotation', 'read', 'rotate3D', 'write']

import hashlib
import itertools
import tempfile

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pypulseq as pp

from pypulseq.add_gradients import add_gradients
from pypulseq.event_lib import EventLibrary
from pypulseq.scale_grad import scale_grad
from pypulseq.supported_labels_rf_use import get_supported_labels, get_supported_rf_uses

try:
    #: The compiled kernels, on the same terms as in `fastseq.modules`: optional,
    #: with a NumPy twin beside every one of them, and reached whether this
    #: module was imported as part of a package or on its own.
    try:
        from . import _fastseq_wrapper as _accel
    except ImportError:
        import _fastseq_wrapper as _accel
except ImportError:  # pragma: no cover - depends on whether the wheel was built
    _accel = None


# =============================================================================
# Extension events
# =============================================================================


def make_rotation(rot_quaternion) -> SimpleNamespace:
    """Create a rotation extension event.

    Attaching one of these to a block rotates every gradient in it, without
    redesigning any waveform. That is what makes the non-Cartesian readouts
    cheap: one base spoke or interleaf is designed, and each shot is the same
    waveform under a different rotation.

    `rot_quaternion` is anything exposing `as_quat(canonical=True,
    scalar_first=True)` -- a `scipy.spatial.transform.Rotation` -- or the four
    numbers themselves, in scalar-first order. A scan loop that varies the
    rotation per block is better off with the numbers: converting a `Rotation`
    costs more than everything else `add_block` does put together.
    """
    event = SimpleNamespace()
    event.type = 'rot3D'
    event.rot_quaternion = rot_quaternion
    return event


def make_rf_shim(shim_vector) -> SimpleNamespace:
    """Create an RF shim extension event.

    On a parallel-transmit system each channel plays the same RF envelope scaled
    by its own complex weight. This event carries that weight vector for a
    block, so one designed pulse can be re-shimmed per slice or per excitation
    without redesigning the envelope.
    """
    event = SimpleNamespace()
    event.type = 'rf_shim'
    event.shim_vector = np.asarray(shim_vector, dtype=np.complex128)
    return event


def rotation_row(event) -> tuple:
    """The four numbers a rotation event contributes to the library.

    Scalar-first, canonical, which is the order the file format writes and the
    order `scipy` produces when asked for it.
    """
    quaternion = event.rot_quaternion
    as_quat = getattr(quaternion, 'as_quat', None)
    if as_quat is not None:
        quaternion = as_quat(canonical=True, scalar_first=True)
    return tuple(float(value) for value in quaternion)


def rf_shim_row(event) -> tuple:
    """A shim vector as the file stores it: magnitude and phase, channel by channel."""
    vector = event.shim_vector
    if np.iscomplexobj(vector):
        values = np.stack((np.abs(vector), np.angle(vector)), axis=-1).ravel()
    else:
        # Already magnitude/phase interleaved -- what a loop hands in when it
        # would rather not pay for the conversion once per block.
        values = np.asarray(vector, dtype=float).ravel()
    return tuple(values.tolist())


_AXES = ('x', 'y', 'z')


def _grad_peak(grad: SimpleNamespace) -> float:
    """Peak absolute amplitude of a trapezoid or arbitrary gradient."""
    if grad.type == 'trap':
        return abs(grad.amplitude)
    return float(np.max(np.abs(grad.waveform)))


def rotate3D(*args: SimpleNamespace, rotation_matrix, system=None) -> list:
    """Rotate the gradients in `args` by a 3x3 rotation matrix.

    `pp.rotate` turns gradients about a single axis; a `ROTATIONS` extension
    carries a full quaternion, so applying one needs the general form -- which
    MATLAB Pulseq has as `rotate3D` and PyPulseq 1.5.0 does not have at all.
    Non-gradient events pass through untouched.
    """
    if system is None:
        system = pp.Opts.default

    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if rotation_matrix.shape != (3, 3):
        raise ValueError('The rotation matrix must have shape (3, 3).')

    to_rotate: dict[str, SimpleNamespace] = {}
    bypass: list[SimpleNamespace] = []

    for event in args:
        if event.type not in ('grad', 'trap'):
            bypass.append(event)
            continue
        if event.channel not in _AXES:
            raise ValueError(f'Invalid event channel. Expected one of {list(_AXES)}')
        if event.channel in to_rotate:
            raise ValueError(f'More than one gradient provided for channel {event.channel}')
        to_rotate[event.channel] = event

    max_magnitude = max((_grad_peak(grad) for grad in to_rotate.values()), default=0.0)
    floor = 1e-6
    threshold = floor * max_magnitude

    rotated: list[SimpleNamespace] = []
    for target in range(3):
        out = None
        for source in range(3):
            if _AXES[source] not in to_rotate or abs(rotation_matrix[target, source]) < floor:
                continue
            scaled = scale_grad(grad=to_rotate[_AXES[source]], scale=rotation_matrix[target, source])
            scaled.channel = _AXES[target]
            out = scaled if out is None else add_gradients((out, scaled), system=system)
        if out is not None and _grad_peak(out) >= threshold:
            rotated.append(out)

    return [*bypass, *rotated]


def _rotation_matrix(quaternion) -> np.ndarray:
    """A scalar-first quaternion as a 3x3 rotation matrix."""
    w, x, y, z = (float(value) for value in quaternion)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# =============================================================================
# Labels
# =============================================================================
#
# Pulseq names a label in the file and numbers it in the libraries, and the
# numbering is positional in a list every interpreter agrees on. A label outside
# that list -- which a research sequence may well want -- keeps working here by
# being numbered past its end and written out by name; only the file says what
# it means, which is all a custom label can ever be.

_BUILTIN_LABELS = tuple(get_supported_labels())

#: Label name -> numeric ID, one-based, extended as custom names turn up. Ids
#: are process-wide rather than per sequence so that an event carries the same
#: number wherever it is played, which is what lets the hot path read one dict.
LABEL_ID = {name: n for n, name in enumerate(_BUILTIN_LABELS, start=1)}
LABEL_NAME = {n: name for name, n in LABEL_ID.items()}


def label_id(name: str) -> int:
    """The numeric ID of `name`, registering it if it is not a built-in label."""
    number = LABEL_ID.get(name)
    if number is None:
        number = len(LABEL_ID) + 1
        LABEL_ID[name] = number
        LABEL_NAME[number] = name
    return number


# =============================================================================
# The sequence
# =============================================================================

#: Attributes that belong to the wrapper rather than to the sequence it wraps.
#: Everything else -- `block_events`, `definitions`, the event libraries -- is
#: read from and written to the wrapped `pp.Sequence`, so that code holding one
#: of these cannot tell the difference.
_OWN = frozenset({'rotation_library', 'rf_shim_library'})

#: Delegated names that cannot be looking at the block table, and so do not have
#: to pay for materializing it. Everything else reached through `__getattr__`
#: might be -- a bound `pp.Sequence` method reads `_seq.block_events` itself,
#: where a deferred table would look like an empty sequence -- so it does.
_BLOCK_FREE = frozenset({
    'system', 'definitions', 'rf_raster_time', 'adc_raster_time',
    'grad_raster_time', 'block_duration_raster', 'version_major',
    'version_minor', 'version_revision', 'rf_library', 'grad_library',
    'adc_library', 'shape_library', 'trigger_library',
    'label_set_library', 'label_inc_library', 'soft_delay_library',
    'soft_delay_hints', 'extension_string_idx', 'extension_numeric_idx',
    'set_definition', 'get_definition', 'set_extension_string_ID',
    'get_extension_type_ID', 'get_extension_type_string',
    'use_block_cache', 'block_cache', 'next_free_block_ID',
})


class Sequence:
    """A `pp.Sequence`, plus the rotation and RF-shim libraries it has no room for.

    Everything PyPulseq does is delegated: attribute reads and writes that are
    not this class's own go straight through to `_seq`, so a `Sequence` can be
    handed to anything expecting a `pp.Sequence` and used as one. What is added
    is the two extension libraries, a writer and reader that know about them,
    and a `plot` that can show what they mean.
    """

    def __init__(self, system: pp.Opts | None = None, seq: pp.Sequence | None = None):
        object.__setattr__(self, '_seq', seq if seq is not None else pp.Sequence(system=system))
        object.__setattr__(self, '_native_seq', None)
        object.__setattr__(self, '_deferred', None)
        object.__setattr__(self, '_deferred_chains', None)
        object.__setattr__(self, 'rotation_library', EventLibrary())
        object.__setattr__(self, 'rf_shim_library', EventLibrary())

    # -- the block table ----------------------------------------------------

    def defer_blocks(self, table: np.ndarray, durations: np.ndarray) -> None:
        """Hold the block table as two arrays, and build the dicts only if asked.

        A scan's block table is one row per block by definition, so on a large
        3D protocol it is millions of rows -- and Pulseq wants it as a dict of
        rows keyed by block number, which costs an object per row to build and
        another pass to take apart again. Writing a file needs neither: it wants
        the columns, which is what came out of deduplication in the first place.

        So the arrays are kept as they are, and `block_events` and
        `block_durations` build their dicts on the first ask. Nothing that reads
        a sequence can tell, because asking is what materializes them; the only
        thing that goes faster is the path that never asks.
        """
        object.__setattr__(self, '_deferred', (table, durations))
        self._seq.next_free_block_ID = table.shape[0] + 1

    def defer_chains(self, table: np.ndarray) -> None:
        """Hold the extension chains as an array, on the same terms as the blocks.

        A block carrying two labels needs a chain row of its own, so a 3D scan
        has one per phase-encode pair -- half a million of them on an MPRAGE,
        against a couple of thousand rows in every other library put together.
        Building that dictionary, and its keymap of tuples, is the whole cost of
        turning deduplicated tables into a Pulseq sequence; writing a file wants
        the columns back again.

        So it is kept as the array deduplication produced. `extensions_library`
        builds the real thing on the first ask, which is what plotting or
        decoding a block does and what writing one never does.
        """
        object.__setattr__(self, '_deferred_chains', table)

    def materialize(self) -> None:
        """Build the deferred libraries now, if they are still being deferred."""
        chains = self._deferred_chains
        if chains is not None:
            object.__setattr__(self, '_deferred_chains', None)
            rows = list(map(tuple, chains.tolist()))
            library = self._seq.extensions_library
            library.data = dict(enumerate(rows, start=1))
            library.keymap = dict(zip(rows, range(1, len(rows) + 1)))
            library.next_free_ID = len(rows) + 1

        deferred = self._deferred
        if deferred is None:
            return
        table, durations = deferred
        object.__setattr__(self, '_deferred', None)
        numbers = range(1, table.shape[0] + 1)
        self._seq.block_events = dict(zip(numbers, table))
        self._seq.block_durations = dict(zip(numbers, durations.tolist()))

    @property
    def extensions_library(self):
        self.materialize()
        return self._seq.extensions_library

    @extensions_library.setter
    def extensions_library(self, value) -> None:
        object.__setattr__(self, '_deferred_chains', None)
        self._seq.extensions_library = value

    @property
    def block_events(self) -> dict:
        self.materialize()
        return self._seq.block_events

    @block_events.setter
    def block_events(self, value) -> None:
        object.__setattr__(self, '_deferred', None)
        self._seq.block_events = value

    @property
    def block_durations(self) -> dict:
        self.materialize()
        return self._seq.block_durations

    @block_durations.setter
    def block_durations(self, value) -> None:
        self.materialize()
        self._seq.block_durations = value

    # -- delegation ---------------------------------------------------------

    def __getattr__(self, name: str):
        # Only reached when the wrapper has no such attribute of its own.
        if name.startswith('__'):
            raise AttributeError(name)
        seq = object.__getattribute__(self, '_seq')
        if name not in _BLOCK_FREE and (
            object.__getattribute__(self, '_deferred') is not None
            or object.__getattribute__(self, '_deferred_chains') is not None
        ):
            # Whatever this is, it may be a `pp.Sequence` method that reads the
            # block table off `_seq` without coming back through the wrapper.
            self.materialize()
        return getattr(seq, name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith('_') or name in _OWN:
            object.__setattr__(self, name, value)
        elif isinstance(getattr(type(self), name, None), property):
            getattr(type(self), name).__set__(self, value)
        else:
            setattr(self._seq, name, value)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._seq)))

    def __len__(self) -> int:
        """How many blocks, without building the table to find out."""
        deferred = self._deferred
        return deferred[0].shape[0] if deferred is not None else len(self._seq.block_events)

    def __repr__(self) -> str:
        return (
            f'<fastseq Sequence: {len(self)} blocks, '
            f'{len(self.rotation_library.data)} rotations, '
            f'{len(self.rf_shim_library.data)} RF shims>'
        )

    # -- building -----------------------------------------------------------

    def add_block(self, *events) -> None:
        """Add one block, taking the two extension events PyPulseq would refuse.

        The native events go to PyPulseq, which builds whatever extension chain
        they call for; a rotation or a shim is then registered here and chained
        on top of it. Order within a block's chain carries no meaning -- Pulseq
        stores it sorted by library ID -- so appending is all this has to do.
        """
        native, extra = [], []
        for event in events:
            if getattr(event, 'type', None) in ('rot3D', 'rf_shim'):
                extra.append(event)
            else:
                native.append(event)

        self._seq.add_block(*native)
        if extra:
            self._chain(self._seq.next_free_block_ID - 1, extra)
        self._native_seq = None

    def _chain(self, block_id: int, events) -> None:
        """Register `events` and hang them off block `block_id`'s extension chain."""
        self.materialize()
        seq = self._seq
        row = seq.block_events[block_id]
        head = int(row[6])
        for event in events:
            if event.type == 'rot3D':
                name, library, data = 'ROTATIONS', self.rotation_library, rotation_row(event)
            else:
                name, library, data = 'RF_SHIMS', self.rf_shim_library, rf_shim_row(event)
            ref, _ = library.find_or_insert(data)
            head, _ = seq.extensions_library.find_or_insert(
                (seq.get_extension_type_ID(name), ref, head)
            )
        row[6] = head
        if seq.use_block_cache:
            # The block was cached before the chain was finished, and what is
            # cached would now decode without its rotation.
            seq.block_cache.pop(block_id, None)

    # -- inspection ---------------------------------------------------------

    def duration(self):
        """`(total, blocks, event_count)`, exactly as PyPulseq reports it."""
        self.materialize()
        return self._seq.duration()

    def extension_refs(self, block_id: int) -> tuple:
        """`(rotation ref, rf shim ref, native chain head)` for one block.

        Walking a chain is how anything finds out what a block carries, and this
        is the one walk that has to separate what PyPulseq can decode from what
        it cannot -- both to plot the sequence and to hand the rest to
        `get_block`.
        """
        self.materialize()
        seq = self._seq
        rotation = shim = None
        native = []
        ext_id = int(seq.block_events[block_id][6])
        while ext_id > 0:
            type_id, ref, ext_id = seq.extensions_library.data[ext_id]
            name = seq.get_extension_type_string(int(type_id))
            if name == 'ROTATIONS':
                rotation = int(ref)
            elif name == 'RF_SHIMS':
                shim = int(ref)
            else:
                native.append((int(type_id), int(ref)))
        head = 0
        for type_id, ref in reversed(native):
            head, _ = seq.extensions_library.find_or_insert((type_id, ref, head))
        return rotation, shim, head

    def get_block(self, block_index: int):
        """One decoded block, with its rotation and shim alongside the rest.

        PyPulseq refuses a chain naming an extension it has no decoder for, so
        it is shown a chain without those and told about them afterwards: the
        block comes back with `rotation` and `rf_shim` set to the library rows,
        or to `None`, and everything else exactly as upstream built it.
        """
        seq = self._seq
        rotation, shim, native_head = self.extension_refs(block_index)
        if rotation is None and shim is None:
            block = seq.get_block(block_index)
            block.rotation = block.rf_shim = None
            return block

        row = seq.block_events[block_index]
        head = int(row[6])
        row[6] = native_head
        if seq.use_block_cache:
            seq.block_cache.pop(block_index, None)
        try:
            block = seq.get_block(block_index)
        finally:
            row[6] = head
            if seq.use_block_cache:
                seq.block_cache.pop(block_index, None)

        block.rotation = None if rotation is None else self.rotation_library.data[rotation]
        block.rf_shim = None if shim is None else self.rf_shim_library.data[shim]
        return block

    def native(self) -> pp.Sequence:
        """The sequence as it will play, in terms PyPulseq alone can express.

        Rotations stop being quaternions and become rotated waveforms; a shimmed
        RF pulse stops being one envelope and a weight vector and becomes its
        channels, laid one after another in the signal and sharing the time
        axis, so that a plot of it shows every channel at once.

        Built on demand and kept until the sequence changes, because it is a
        second copy of the whole scan.
        """
        if self._native_seq is not None:
            return self._native_seq

        self.materialize()
        seq = self._seq
        native = pp.Sequence(system=seq.system)
        native.definitions = dict(seq.definitions)

        for block_id in list(seq.block_events):
            block = self.get_block(block_id)
            duration = seq.block_durations[block_id]
            events = self._playable(block, block.rotation, block.rf_shim, seq.system)
            events.append(pp.make_delay(duration))
            native.add_block(*events)
            native.block_durations[block_id] = duration

        self._native_seq = native
        return native

    def _playable(self, block, rotation, shim, system) -> list:
        """One decoded block's events, with its rotation and shim applied."""
        events = []
        for name in ('rf', 'gx', 'gy', 'gz', 'adc'):
            event = getattr(block, name, None)
            if event is not None:
                events.append(event)
        for name in ('label', 'trigger'):
            carried = getattr(block, name, None)
            if carried:
                events.extend(carried.values())

        if shim is not None:
            weights = np.asarray(shim, dtype=float)
            magnitude, phase = weights[0::2], weights[1::2]
            for n, event in enumerate(events):
                if getattr(event, 'type', None) == 'rf':
                    events[n] = _multichannel(event, magnitude * np.exp(1j * phase))

        if rotation is not None:
            events = rotate3D(*events, rotation_matrix=_rotation_matrix(rotation), system=system)

        return events

    def plot(self, *args, **kwargs):
        """Plot the sequence as it will play. See `native`."""
        return self.native().plot(*args, **kwargs)

    # -- files --------------------------------------------------------------

    def write(self, path=None, **kwargs):
        """Write a Pulseq 1.5.1 file. See `fastseq.sequence.write`."""
        return write(self, path, **kwargs)

    @classmethod
    def read(cls, source, system=None) -> 'Sequence':
        """Read a `.seq` file, 1.5.0 or 1.5.1. See `fastseq.sequence.read`."""
        return read(source, system=system)


def _multichannel(rf: SimpleNamespace, weights: np.ndarray) -> SimpleNamespace:
    """One shimmed RF pulse as the several pulses it really is.

    The channels are laid one after another in `signal` and share one time axis,
    so plotting the result draws every channel over the same interval -- which
    is what a shim vector means and what no single-channel event can show.
    """
    channels = len(weights)
    if channels == 0:
        return rf
    shimmed = SimpleNamespace(**vars(rf))
    shimmed.signal = np.concatenate([weight * rf.signal for weight in weights])
    shimmed.t = np.tile(rf.t, channels)
    for cached in ('id', 'shape_IDs'):
        if hasattr(shimmed, cached):
            delattr(shimmed, cached)
    return shimmed


# =============================================================================
# Writing
# =============================================================================


def _render_int_rows(matrix: np.ndarray, widths: tuple) -> str | None:
    """Render a non-negative integer matrix as space-separated right-aligned text.

    `[BLOCKS]` is the one section with a row per block rather than per distinct
    event, so on a large 3D protocol it is most of the file and rendering it is
    most of the cost of writing one.

    The compiled kernel, when there is one, writes each value's digits straight
    into the output buffer. `_render_int_rows_py` below is the NumPy version it
    has to agree with character for character, and what runs when the extension
    was not built.

    Returns `None` when a value will not fit its field -- `%3d` widens rather
    than truncating, so those rows are not fixed-width and the caller has to
    fall back to formatting them individually.
    """
    if _accel is not None and matrix.ndim == 2 and matrix.dtype == np.int64:
        return _accel.render_int_rows(
            np.ascontiguousarray(matrix), [int(width) for width in widths]
        )
    return _render_int_rows_py(matrix, widths)


def _render_int_rows_py(matrix: np.ndarray, widths: tuple) -> str | None:
    """`_render_int_rows` in NumPy alone -- the reference the kernel is checked against.

    Formatting row by row means turning some five million array elements into
    Python ints purely to hand them to `%`; here the digits are computed with
    arithmetic on the whole column and the section comes out as one string.
    """
    rows, columns = matrix.shape
    if columns != len(widths) or rows == 0 or matrix.min() < 0:
        return None

    maxima = matrix.max(axis=0)
    spans = [max(width, len(str(int(value)))) for width, value in zip(widths, maxima)]

    line_width = sum(spans) + len(spans)  # one separator per field, newline last
    text = np.full((rows, line_width), 32, dtype=np.uint8)  # spaces
    keep = np.ones((rows, line_width), dtype=bool)

    cursor = 0
    for column, (span, width) in enumerate(zip(spans, widths)):
        values = matrix[:, column]
        field = text[:, cursor : cursor + span]
        remainder = values
        for digit in range(span - 1, -1, -1):
            field[:, digit] = 48 + (remainder % 10)
            remainder = remainder // 10

        # How many characters %{width}d would have emitted for each value.
        digits = np.ones(rows, dtype=np.int64)
        for power in range(1, span):
            digits += values >= 10**power
        printed = np.maximum(digits, width)
        positions = np.arange(span)
        # Everything left of the first significant digit is padding, not a zero;
        # of that padding, only what %{width}d would have emitted is kept.
        field[positions < (span - digits)[:, None]] = 32
        keep[:, cursor : cursor + span] = positions >= (span - printed)[:, None]
        cursor += span + 1

    text[:, -1] = 10  # newline
    return text[keep].tobytes().decode('ascii')


def _label_string(number: int) -> str:
    """The name a label ID is written under, whether or not Pulseq defines it."""
    return LABEL_NAME.get(int(number), f'CUSTOM_{int(number)}')


def write(seq, output=None, *, create_signature: bool = False, check_timing: bool = False):
    """Write a sequence as a Pulseq file, at the revision it actually needs.

    Rotations and RF shims are revision 1.5.1; a label outside Pulseq's built-in
    set is 1.5.2. A sequence using none of them is written as plain 1.5.0, so the
    common case produces a file any interpreter reads.

    `output` may be a path, a binary file object, or `None` to get the bytes
    back. Deduplication is not done here: a `PeriodicSequence` has already
    collapsed its libraries by the time it can be written, and a module is small.
    """
    plain = getattr(seq, '_seq', seq)
    rotations = getattr(seq, 'rotation_library', None)
    shims = getattr(seq, 'rf_shim_library', None)

    # The block table, as columns. A sequence a `PeriodicSequence` built still
    # has it that way and hands it over untouched; any other sequence keeps it
    # as Pulseq's dicts, and it is gathered here. Either way what follows reads
    # arrays, since this is the one section whose size is the length of the scan.
    deferred = getattr(seq, '_deferred', None)
    if deferred is not None:
        table, seconds = deferred
        numbers = np.arange(1, table.shape[0] + 1, dtype=np.int64)
    else:
        n = len(plain.block_events)
        numbers = np.fromiter(plain.block_events, dtype=np.int64, count=n)
        table = np.stack(list(plain.block_events.values())) if n else np.zeros((0, 7), int)
        seconds = np.fromiter(plain.block_durations.values(), dtype=float, count=n)

    if check_timing:
        ok, errors = plain.check_timing()
        if not ok:
            from warnings import warn

            warn(f'write(): {len(errors)} timing errors found in the sequence', stacklevel=2)

    plain.set_definition('TotalDuration', float(seconds.sum()))

    has_rotations = rotations is not None and len(rotations.data) != 0
    has_shims = shims is not None and len(shims.data) != 0
    label_numbers = {
        int(data[1])
        for library in (plain.label_set_library, plain.label_inc_library)
        for data in library.data.values()
    }
    has_custom_labels = any(number > len(_BUILTIN_LABELS) for number in label_numbers)

    revision = int(plain.version_revision)
    if has_custom_labels:
        revision = max(revision, 2)
    elif has_rotations or has_shims:
        revision = max(revision, 1)

    chunks: list[str] = []
    w = chunks.append

    w('# Pulseq sequence file\n')
    w('# Created by PyPulseq\n\n')

    w('[VERSION]\n')
    w(f'major {int(plain.version_major)}\n')
    w(f'minor {int(plain.version_minor)}\n')
    w(f'revision {revision}\n')
    w('\n')

    if len(plain.definitions) != 0:
        w('[DEFINITIONS]\n')
        for key in sorted(plain.definitions):
            value = plain.definitions[key]
            w(f'{key} ')
            if isinstance(value, str):
                w(value + ' ')
            elif isinstance(value, (int, float)):
                w(f'{value:0.9g} ')
            elif isinstance(value, (list, tuple, np.ndarray)):
                for item in value:
                    w(f'{item:0.9g} ' if isinstance(item, (int, float)) else f'{item} ')
            else:
                raise RuntimeError(f'Unsupported definition {key!r}')
            w('\n')
        w('\n')

    w('# Format of blocks:\n')
    w('# NUM DUR RF  GX  GY  GZ  ADC  EXT\n')
    w('[BLOCKS]\n')
    # The one section whose cost scales with the scan rather than with the
    # number of distinct events, so the raster division and its round-trip check
    # are done for the whole column at once.
    n_blocks = numbers.size
    if n_blocks:
        ticks = seconds / plain.block_duration_raster
        rounded = np.rint(ticks)
        if np.abs(rounded - ticks).max() >= 1e-6:
            worst = int(np.argmax(np.abs(rounded - ticks)))
            raise AssertionError(
                f'block {int(numbers[worst])} duration is not a multiple of '
                f'the block duration raster ({ticks[worst] * plain.block_duration_raster} s)'
            )
        printed = np.empty((n_blocks, 8), dtype=np.int64)
        printed[:, 0] = numbers
        printed[:, 1] = rounded
        printed[:, 2:] = table[:, 1:]

        widths = (len(str(n_blocks)), 3, 3, 3, 3, 3, 2, 2)
        rendered = _render_int_rows(printed, widths)
        if rendered is not None:
            w(rendered)
        else:
            row_fmt = f'%{len(str(n_blocks))}d %3d %3d %3d %3d %3d %2d %2d\n'
            for row in printed.tolist():
                w(row_fmt % tuple(row))
    w('\n')

    if len(plain.rf_library.data) != 0:
        w('# Format of RF events:\n')
        w('# id ampl. mag_id phase_id time_shape_id center delay freqPPm phasePPM freq phase use\n')
        w('# ..   Hz      ..       ..            ..     us    us     ppm  rad/MHz   Hz   rad  ..\n')
        w(f'# Field "use" is the initial of: {" ".join(get_supported_rf_uses()).strip()}\n')
        w('[RF]\n')
        rf_fmt = '%.0f %12g %.0f %.0f %.0f %g %g %g %g %g %g %s\n'
        raster = plain.rf_raster_time
        for k, data in plain.rf_library.data.items():
            center = data[4] * 1e6
            delay = round(data[5] / raster) * raster * 1e6
            w(rf_fmt % (k, *data[0:4], center, delay, *data[6:10], plain.rf_library.type.get(k, 'u')))
        w('\n')

    arbitrary = [k for k, kind in plain.grad_library.type.items() if kind == 'g']
    trapezoids = [k for k, kind in plain.grad_library.type.items() if kind == 't']

    if arbitrary:
        w('# Format of arbitrary gradients:\n')
        w('#   time_shape_id of 0 means default timing (stepping with grad_raster starting at 1/2 of grad_raster)\n')
        w('# id amplitude first last amp_shape_id time_shape_id delay\n')
        w('# ..      Hz/m  Hz/m Hz/m        ..         ..          us\n')
        w('[GRADIENTS]\n')
        arb_fmt = '%.0f %12g %12g %12g %.0f %.0f %.0f\n'
        for k in arbitrary:
            data = plain.grad_library.data[k]
            w(arb_fmt % (k, *data[:5], round(data[5] * 1e6)))
        w('\n')

    if trapezoids:
        w('# Format of trapezoid gradients:\n')
        w('# id amplitude rise flat fall delay\n')
        w('# ..      Hz/m   us   us   us    us\n')
        w('[TRAP]\n')
        trap_fmt = '%2.0f %12g %3.0f %4.0f %3.0f %3.0f\n'
        for k in trapezoids:
            amplitude, rise, flat, fall, delay = plain.grad_library.data[k]
            w(trap_fmt % (k, amplitude, 1e6 * rise, 1e6 * flat, 1e6 * fall, 1e6 * delay))
        w('\n')

    if len(plain.adc_library.data) != 0:
        w('# Format of ADC events:\n')
        w('# id num dwell delay freqPPM phasePPM freq phase phase_id\n')
        w('# ..  ..    ns    us     ppm  rad/MHz   Hz   rad       ..\n')
        w('[ADC]\n')
        adc_fmt = '%.0f %.0f %.0f %.0f %g %g %g %g %.0f\n'
        for k, data in plain.adc_library.data.items():
            num, dwell, delay, freq_ppm, phase_ppm, freq, phase, phase_id = data[0:8]
            w(adc_fmt % (k, num, 1e9 * dwell, 1e6 * delay, freq_ppm, phase_ppm, freq, phase, phase_id))
        w('\n')

    # The other section that grows with the scan rather than with the number of
    # distinct events: a block carrying two labels needs a chain of its own, so
    # a 3D scan has one row here per phase-encode pair. A sequence a
    # `PeriodicSequence` built still holds them as the array deduplication
    # produced and hands it over untouched, so the library is never built.
    deferred_chains = getattr(seq, '_deferred_chains', None)
    if deferred_chains is not None:
        chain_count = deferred_chains.shape[0]
    else:
        chain_count = len(plain.extensions_library.data)

    if chain_count != 0:
        w('# Format of extension lists:\n')
        w('# id type ref next_id\n')
        w('# next_id of 0 terminates the list\n')
        w('# Extension list is followed by extension specifications\n')
        w('[EXTENSIONS]\n')
        rows = np.empty((chain_count, 4), dtype=np.int64)
        rows[:, 0] = np.arange(1, chain_count + 1, dtype=np.int64)
        if deferred_chains is not None:
            rows[:, 1:] = deferred_chains
        else:
            chains = plain.extensions_library.data
            rows[:, 0] = np.fromiter(chains, dtype=np.int64, count=chain_count)
            rows[:, 1:] = np.rint(
                np.fromiter(
                    itertools.chain.from_iterable(chains.values()),
                    dtype=np.float64, count=chain_count * 3,
                ).reshape(-1, 3)
            )
        rendered = _render_int_rows(rows, (1, 1, 1, 1))
        if rendered is not None:
            w(rendered)
        else:
            ext_fmt = '%.0f %.0f %.0f %.0f\n'
            for row in rows.tolist():
                w(ext_fmt % tuple(row))
        w('\n')

    if len(plain.trigger_library.data) != 0:
        w('# Extension specification for digital output and input triggers:\n')
        w('# id type channel delay (us) duration (us)\n')
        w(f'extension TRIGGERS {plain.get_extension_type_ID("TRIGGERS")}\n')
        trigger_fmt = '%.0f %.0f %.0f %.0f %.0f\n'
        for k, data in plain.trigger_library.data.items():
            kind, channel, delay, duration = data
            w(trigger_fmt % (k, kind, channel, 1e6 * delay, 1e6 * duration))
        w('\n')

    for name, library in (
        ('LABELSET', plain.label_set_library),
        ('LABELINC', plain.label_inc_library),
    ):
        if len(library.data) == 0:
            continue
        w('# Extension specification for setting labels:\n')
        w('# id set labelstring\n')
        w(f'extension {name} {plain.get_extension_type_ID(name)}\n')
        label_fmt = '%.0f %.0f %s\n'
        for k, data in library.data.items():
            w(label_fmt % (k, data[0], _label_string(data[1])))
        w('\n')

    if has_shims:
        w('# Extension specification for RF shimming:\n')
        w('# id num_chan factor magn_c1 phase_c1 magn_c2 phase_c2 ...\n')
        w(f'extension RF_SHIMS {plain.get_extension_type_ID("RF_SHIMS")}\n')
        for k, data in shims.data.items():
            fmt = '{:d} {:d}' + ''.join(' {:g}' for _ in range(len(data))) + '\n'
            w(fmt.format(k, len(data) // 2, *data))
        w('\n')

    if has_rotations:
        w('# Extension specification for rotation events:\n')
        w('# id RotQuat0 RotQuatX RotQuatY RotQuatZ\n')
        w(f'extension ROTATIONS {plain.get_extension_type_ID("ROTATIONS")}\n')
        for k, data in rotations.data.items():
            w('{:.0f} {:12g} {:12g} {:12g} {:12g}\n'.format(k, *data))
        w('\n')

    if len(plain.soft_delay_library.data) != 0:
        w('# Extension specification for soft delays:\n')
        w('# id num offset factor hint\n')
        w('# ..  ..     us     ..   ..\n')
        w(f'extension DELAYS {plain.get_extension_type_ID("DELAYS")}\n')
        for k, data in plain.soft_delay_library.data.items():
            w('{:.0f} {:.0f} {:.0f} {:.0f} {}\n'.format(k, data[0], np.round(data[1] * 1e6), data[2], data[3]))
        w('\n')

    if len(plain.shape_library.data) != 0:
        w('# Sequence Shapes\n')
        w('[SHAPES]\n\n')
        for k, shape in plain.shape_library.data.items():
            w(f'shape_id {k:.0f}\n')
            w(f'num_samples {shape[0]:.0f}\n')
            w(('{:.9g}\n' * len(shape[1:])).format(*shape[1:]))
            w('\n')

    body = ''.join(chunks)
    signature = None
    if create_signature:
        signature = hashlib.md5(body.encode('utf-8')).hexdigest()
        body += (
            '\n[SIGNATURE]\n'
            '# This is the hash of the Pulseq file, calculated right before the [SIGNATURE] section was added\n'
            '# It can be reproduced/verified with md5sum if the file trimmed to the position right above [SIGNATURE]\n'
            '# The new line character preceding [SIGNATURE] BELONGS to the signature (and needs to be stripped away for recalculating/verification)\n'
            'Type md5\n'
            f'Hash {signature}\n'
        )

    payload = body.encode('utf-8')
    if output is None:
        return payload
    if hasattr(output, 'write'):
        try:
            output.write(payload)
        except TypeError:
            output.write(body)
        return signature

    path = Path(output)
    if path.suffix != '.seq':
        path = path.with_suffix(path.suffix + '.seq')
    path.write_bytes(payload)
    return signature


# =============================================================================
# Reading
# =============================================================================
#
# Upstream reads everything it knows and refuses the rest outright -- an
# `extension ROTATIONS` section is an unknown section code, and a label name
# outside its built-in list is a `ValueError` from `list.index`. So the file is
# split first, the sections upstream would refuse are held back and parsed here,
# and what is left is a file upstream reads without complaint.

#: Extension sections this module parses itself rather than handing to PyPulseq.
_OWN_SECTIONS = ('ROTATIONS', 'RF_SHIMS')


def _split_sections(text: str) -> list:
    """The file as `(header, body)` pairs, in order; the preamble has no header."""
    sections = [(None, [])]
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(('[', 'extension ')):
            sections.append((stripped, [line]))
        else:
            sections[-1][1].append(line)
    return sections


def _parse_rows(body: list, convert=float) -> dict:
    """The numbered rows of one extension specification section.

    `convert` is either one callable for every field or one per field; a shim
    row is as wide as the system has transmit channels, so only the first is
    any use there.
    """
    rows = {}
    for line in body[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        fields = stripped.split()
        values = fields[1:]
        if callable(convert):
            row = tuple(convert(field) for field in values)
        else:
            row = tuple(step(field) for step, field in zip(convert, values))
        rows[int(fields[0])] = row
    return rows


def read(source, *, system=None) -> Sequence:
    """Read a `.seq` file into a `Sequence`, at revision 1.5.0, 1.5.1 or 1.5.2.

    Everything PyPulseq understands is read by PyPulseq. What it does not -- the
    two extension specifications it has no library for, and label names outside
    its built-in set -- is parsed here and put back afterwards, so a file written
    by this module, or by MATLAB Pulseq, comes back whole.

    Parameters
    ----------
    source : str, pathlib.Path or bytes
        Path to a `.seq` file, or the payload itself.
    system : pypulseq.Opts, optional
        System limits to attach. The raster times the file records in
        `[DEFINITIONS]` are used by PyPulseq regardless.

    Returns
    -------
    Sequence
    """
    payload = source if isinstance(source, bytes) else Path(source).read_bytes()
    text = payload.decode('utf-8')

    kept, held = [], {}
    labels_seen = {}
    for header, body in _split_sections(text):
        if header is not None and header.startswith('extension '):
            _, name, type_id = header.split()
            if name in _OWN_SECTIONS:
                held[name] = (int(type_id), body)
                continue
            if name in ('LABELSET', 'LABELINC'):
                # Held back only when a name is one upstream cannot look up;
                # otherwise upstream reads them and this stays out of the way.
                names = {
                    line.split()[2] for line in body[1:]
                    if line.strip() and not line.strip().startswith('#')
                }
                if not names <= set(_BUILTIN_LABELS):
                    held[name] = (int(type_id), body)
                    continue
                labels_seen[name] = int(type_id)
        kept.append(body)

    seq = Sequence(system=system)
    plain = seq._seq

    # Upstream reads from a path, so what it is given is the file minus the
    # sections it would refuse.
    with tempfile.TemporaryDirectory() as scratch:
        readable = Path(scratch) / 'readable.seq'
        readable.write_text(''.join(''.join(body) for body in kept), encoding='utf-8')
        plain.read(str(readable))

    for name, (type_id, body) in held.items():
        plain.set_extension_string_ID(name, type_id)
        if name == 'ROTATIONS':
            for key, row in _parse_rows(body).items():
                seq.rotation_library.insert(key, row[:4])
        elif name == 'RF_SHIMS':
            for key, row in _parse_rows(body).items():
                # The first field counts the channels; the rest is the vector it
                # counts, magnitude and phase per channel.
                seq.rf_shim_library.insert(key, row[1 : 1 + 2 * int(row[0])])
        else:
            library = plain.label_set_library if name == 'LABELSET' else plain.label_inc_library
            for key, row in _parse_rows(body, (float, str)).items():
                library.insert(key, (row[0], label_id(row[1])))
    return seq
