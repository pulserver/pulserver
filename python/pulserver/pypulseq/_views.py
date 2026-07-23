"""Upstream-PyPulseq views of a Pulserver sequence.

Two views are derived from the same block list, and every analysis method on
:class:`pulserver.pypulseq.Sequence` runs on one of them:

``_seq``
    The plain view. Ordinary RF / gradient / ADC events only — what upstream
    PyPulseq can decode on its own.

``_seqplot``
    The plotting view. Same blocks, but each ``ROTATIONS`` extension has been
    *applied* to the block's gradients (via :func:`._rotate3d.rotate3D`) and
    each ``RF_SHIMS`` extension has been expanded into a multi-channel RF
    waveform. What you see is therefore what the scanner plays, not the
    unrotated base waveform stored in the file.

Both views share the sequence's :class:`pypulseq.Opts`, so raster times stay
consistent between them and any segment view sliced out of them.

Both are built by serialising the sequence and reading it back, which is what
lets upstream do the decoding. The consequence is that amplitudes carry the
``.seq`` format's six significant digits, not full double precision — fine for
looking at a sequence, not a basis for exact numerical comparison.
"""

from __future__ import annotations

__all__ = ["build_views", "slice_view", "time_bounds"]

import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pypulseq as pp

from ._extensions import BlockExtensions, parse_extensions, strip_extensions
from ._rotate3d import rotate3D


def _decode_plain(payload: bytes, system: pp.Opts) -> pp.Sequence:
    """Read a plain-Pulseq payload through upstream :meth:`pypulseq.Sequence.read`."""
    with TemporaryDirectory(prefix="pulserver-view-") as directory:
        path = Path(directory) / "sequence.seq"
        path.write_bytes(payload)
        decoded = pp.Sequence(system=system)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decoded.read(str(path))
    return decoded


def _expand_rf_shim(rf, shim_vector: np.ndarray):
    """Replicate an RF event across transmit channels weighted by ``shim_vector``.

    The per-channel time axes are concatenated rather than offset — ``t``
    becomes ``(0..t_end, 0..t_end, ...)`` — so every channel is drawn over the
    same interval and upstream plotting needs no pTx awareness.
    """
    shim = np.asarray(shim_vector, dtype=complex).ravel()
    if shim.size == 0:
        return rf

    rf = deepcopy(rf)
    signal = np.asarray(rf.signal).ravel()
    t = np.asarray(rf.t, dtype=float).ravel()

    rf.signal = (shim[:, None] * signal[None, :]).ravel()
    rf.t = np.tile(t, shim.size)
    return rf


def _apply_extensions(block, ext: BlockExtensions, system: pp.Opts) -> list:
    """Return the block's events with rotation and RF shim resolved into them."""
    events = []
    gradients = []
    for name in ("gx", "gy", "gz"):
        grad = getattr(block, name, None)
        if grad is not None:
            gradients.append(grad)

    rf = getattr(block, "rf", None)
    if rf is not None:
        events.append(_expand_rf_shim(rf, ext.rf_shim) if ext.rf_shim is not None else rf)

    if gradients and ext.rotation is not None:
        gradients = rotate3D(*gradients, rotation_matrix=ext.rotation, system=system)
    events.extend(gradients)

    adc = getattr(block, "adc", None)
    if adc is not None:
        events.append(adc)

    return events


def _append(target: pp.Sequence, events: list, duration: float, block_index: int) -> None:
    """Append ``events`` to ``target``, pinning the block to its original duration."""
    args = [*events]
    if duration > 0:
        args.append(pp.make_delay(duration))
    try:
        target.add_block(*args)
    except RuntimeError as exc:
        if "first block has to start at 0" not in str(exc):
            raise
        # A slice that starts mid-ramp is a legitimate thing to want to look
        # at; upstream only rejects it because it cannot know a preceding
        # block exists. Relax the leading edge and say so.
        warnings.warn(
            f"Block {block_index} starts on a non-zero gradient; its leading edge was "
            "zeroed so the view could be built. Amplitudes elsewhere are unaffected.",
            stacklevel=3,
        )
        for event in args:
            if getattr(event, "type", None) == "grad":
                event.first = 0.0
        target.add_block(*args)


def build_views(seq, payload: bytes) -> tuple[pp.Sequence, pp.Sequence, dict[int, BlockExtensions]]:
    """Build the plain and plotting views of ``seq`` from its ``.seq`` payload.

    Returns
    -------
    tuple
        ``(plain, plotting, extensions)``. When the sequence carries no
        rotation or RF shim extension the two views are the same object.
    """
    system = seq.system
    extensions = parse_extensions(payload)
    plain = _decode_plain(strip_extensions(payload), system)

    resolvable = {idx: ext for idx, ext in extensions.items() if ext.rotation is not None or ext.rf_shim is not None}
    if not resolvable:
        return plain, plain, extensions

    plotting = pp.Sequence(system=system)
    plotting.definitions = dict(plain.definitions)
    for block_index in plain.block_events:
        block = plain.get_block(block_index)
        ext = resolvable.get(block_index)
        events = _apply_extensions(block, ext, system) if ext is not None else _block_events(block)
        _append(plotting, events, plain.block_durations[block_index], block_index)

    return plain, plotting, extensions


def _block_events(block) -> list:
    """Ordinary RF / gradient / ADC events of a decoded block, in plot order."""
    events = []
    rf = getattr(block, "rf", None)
    if rf is not None:
        events.append(rf)
    for name in ("gx", "gy", "gz"):
        grad = getattr(block, name, None)
        if grad is not None:
            events.append(grad)
    adc = getattr(block, "adc", None)
    if adc is not None:
        events.append(adc)
    return events


def slice_view(source: pp.Sequence, first: int, last: int) -> pp.Sequence:
    """Build a standalone sequence from blocks ``first..last`` of ``source``.

    Parameters
    ----------
    source : pypulseq.Sequence
        View to slice, normally ``_seqplot``.
    first, last : int
        Inclusive 1-based block indices.

    Returns
    -------
    pypulseq.Sequence
        A new sequence sharing ``source.system``, so raster times match.
    """
    out = pp.Sequence(system=source.system)
    for block_index in range(first, last + 1):
        if block_index not in source.block_events:
            continue
        block = source.get_block(block_index)
        _append(out, _block_events(block), source.block_durations[block_index], block_index)
    return out


def time_bounds(source: pp.Sequence, first: int, last: int) -> tuple[float, float]:
    """Start and end time (seconds) of blocks ``first..last`` within ``source``."""
    start = 0.0
    for block_index in source.block_events:
        if block_index >= first:
            break
        start += source.block_durations[block_index]
    stop = start
    for block_index in range(first, last + 1):
        stop += source.block_durations.get(block_index, 0.0)
    return start, stop
