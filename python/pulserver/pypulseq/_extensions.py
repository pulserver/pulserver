"""Parsing of Pulserver's block extensions straight out of a ``.seq`` payload.

Upstream PyPulseq 1.5.0 decodes ``TRIGGERS``, ``LABELSET`` and ``LABELINC``
but not the two extensions Pulserver relies on — ``ROTATIONS`` and
``RF_SHIMS`` — and it rejects label strings outside its built-in set. So the
plain upstream view of a Pulserver sequence is produced from a payload with
the whole extension region stripped (:func:`strip_extensions`), and the
extensions themselves are read back from the *original* payload here.

The result is a mapping ``block_index -> BlockExtensions`` keyed by the
1-based Pulseq block index, which is what both the plotting views and
:func:`pulserver.io.read` consume.

Notes
-----
``RF_SHIMS`` phases are **radians**, matching what
:meth:`pulserver.pypulseq.Sequence._fast_register_rf_shim` writes and what the
pulseg C parser reads back. (The unofficial PyPulseq 1.5.1 port in
``refcode/`` reads that column as turns; it disagrees with both ends of
Pulserver's own pipeline, so it is not followed here.)
"""

from __future__ import annotations

__all__ = ["BlockExtensions", "parse_extensions", "strip_extensions"]

from dataclasses import dataclass, field

import numpy as np

_EXTENSION_SECTIONS = ("TRIGGERS", "LABELSET", "LABELINC", "RF_SHIMS", "ROTATIONS")

_TRIGGER_TYPES = ("output", "trigger")
_OUTPUT_CHANNELS = ("osc0", "osc1", "ext1")
_INPUT_CHANNELS = ("physio1", "physio2")


@dataclass
class BlockExtensions:
    """Every extension attached to a single block."""

    #: Rotation matrix (3x3) from the block's ``ROTATIONS`` entry, if any.
    rotation: np.ndarray | None = None
    #: Complex per-channel weights from the block's ``RF_SHIMS`` entry, if any.
    rf_shim: np.ndarray | None = None
    #: ``(type, channel, delay_s, duration_s)`` for each ``TRIGGERS`` entry.
    triggers: list[tuple[str, str, float, float]] = field(default_factory=list)
    #: ``(label, value)`` for each ``LABELSET`` entry.
    label_sets: list[tuple[str, float]] = field(default_factory=list)
    #: ``(label, value)`` for each ``LABELINC`` entry.
    label_incs: list[tuple[str, float]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(
            self.rotation is not None or self.rf_shim is not None or self.triggers or self.label_sets or self.label_incs
        )


def strip_extensions(payload: bytes) -> bytes:
    """Drop the extension region and zero the block EXT column.

    The remaining payload is plain Pulseq that upstream
    :meth:`pypulseq.Sequence.read` accepts unchanged.
    """
    lines: list[str] = []
    in_blocks = False
    skipping = False
    for line in payload.decode("utf-8").splitlines(keepends=True):
        section = line.strip()
        if section == "[BLOCKS]":
            in_blocks, skipping = True, False
        elif section == "[EXTENSIONS]":
            in_blocks, skipping = False, True
            continue
        elif skipping and section == "[SHAPES]":
            skipping = False
        elif skipping:
            continue
        elif section.startswith("["):
            in_blocks = False

        if in_blocks and line.lstrip()[:1].isdigit():
            fields = line.split()
            if len(fields) == 8:
                fields[-1] = "0"
                line = " ".join(fields) + "\n"
        lines.append(line)
    return "".join(lines).encode("utf-8")


def _quaternion_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Scalar-first quaternion to a 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation

    arr = np.asarray(q, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return np.eye(3)
    return Rotation.from_quat(arr / norm, scalar_first=True).as_matrix()


def parse_extensions(payload: bytes) -> dict[int, BlockExtensions]:
    """Read every block extension out of a ``.seq`` payload.

    Parameters
    ----------
    payload : bytes
        A complete ``.seq`` file, extension region included.

    Returns
    -------
    dict[int, BlockExtensions]
        Keyed by 1-based block index. Blocks without extensions are absent.
    """
    block_ext_head: dict[int, int] = {}
    ext_nodes: dict[int, tuple[int, int, int]] = {}  # id -> (type_id, ref, next)
    type_id_to_name: dict[int, str] = {}
    tables: dict[str, dict[int, list[str]]] = {name: {} for name in _EXTENSION_SECTIONS}

    section = ""
    for raw in payload.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line
            continue
        if line.startswith("extension "):
            parts = line.split()
            if len(parts) >= 3:
                name = parts[1]
                type_id_to_name[int(parts[2])] = name
                section = f"extension {name}"
            continue

        if section == "[BLOCKS]":
            fields = line.split()
            if len(fields) == 8:
                head = int(fields[7])
                if head:
                    block_ext_head[int(fields[0])] = head
        elif section == "[EXTENSIONS]":
            fields = line.split()
            if len(fields) == 4:
                ext_nodes[int(fields[0])] = (int(float(fields[1])), int(float(fields[2])), int(float(fields[3])))
        elif section.startswith("extension "):
            name = section.split()[1]
            if name in tables:
                fields = line.split()
                tables[name][int(fields[0])] = fields[1:]

    if not block_ext_head:
        return {}

    out: dict[int, BlockExtensions] = {}
    for block_index, head in block_ext_head.items():
        ext = BlockExtensions()
        cursor = head
        seen: set[int] = set()
        while cursor:
            if cursor in seen:  # defensive: malformed cyclic chain
                break
            seen.add(cursor)
            node = ext_nodes.get(cursor)
            if node is None:
                break
            type_id, ref, cursor = node
            name = type_id_to_name.get(type_id)
            row = tables.get(name, {}).get(ref) if name else None
            if row is None:
                continue

            if name == "ROTATIONS":
                ext.rotation = _quaternion_to_matrix(tuple(float(v) for v in row[:4]))
            elif name == "RF_SHIMS":
                num_chan = int(row[0])
                values = np.asarray([float(v) for v in row[1 : 1 + 2 * num_chan]], dtype=float)
                ext.rf_shim = values[0::2] * np.exp(1j * values[1::2])
            elif name == "TRIGGERS":
                type_idx, channel_idx = int(row[0]), int(row[1])
                channels = _OUTPUT_CHANNELS if type_idx == 1 else _INPUT_CHANNELS
                ext.triggers.append(
                    (
                        _TRIGGER_TYPES[type_idx - 1],
                        channels[channel_idx - 1],
                        float(row[2]) * 1e-6,
                        float(row[3]) * 1e-6,
                    )
                )
            elif name == "LABELSET":
                ext.label_sets.append((row[1], float(row[0])))
            elif name == "LABELINC":
                ext.label_incs.append((row[1], float(row[0])))

        if ext:
            out[block_index] = ext
    return out
