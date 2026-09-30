"""The pose a tracker leaves for a running scan, in a file both sides hold open.

Prospective motion correction needs the scan to know, between one segment and
the next, where the object now is. The tracker runs here and the scan runs on
the sequencer; what they share is a directory. So the pose travels through a
file of fixed size, created before the scan starts and written in place. It is
never renamed, never truncated and never extended, so a descriptor opened once
stays valid for the whole scan -- which is what the sequencer needs, having
only the gap between two segments to read in.

What travels is a pose, not a change of pose. A latest-value file drops
versions: the reader is not promised to see every write, so one accumulating
changes would silently lose the ones it missed. Reading an absolute pose, it
can miss any number and still be right.

Only the rotation crosses. A translation is corrected as a phase on the data,
which the reconstruction applies.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

#: The file's first four bytes, as the writer's byte order lays them.
MAGIC = 0x504D4342  # 'PMCB'

#: The layout below; a reader refuses one it does not know.
LAYOUT = 1

#: Slots the pose rings through, so a reader never reads one being written.
SLOTS = 2

#: The file's name inside the scan's directory.
FILENAME = "motion.buf"

_WORD = 4
_HEADER_WORDS = 4
_SLOT_WORDS = 12
_ECHO_WORDS = 3
_HEADER_BYTES = _HEADER_WORDS * _WORD
_SLOT_BYTES = _SLOT_WORDS * _WORD
_ECHO_OFFSET = _HEADER_BYTES + SLOTS * _SLOT_BYTES
_SIZE = _ECHO_OFFSET + _ECHO_WORDS * _WORD

#: Where the check word sits in a slot and in the echo.
_CHECK = 2


def _check_of(words, skip: int = _CHECK) -> int:
    """Return a record's check value: FNV-1a over its words, less the check itself."""
    total = 2166136261
    for at, word in enumerate(words):
        if at == skip:
            continue
        total = ((total ^ (word & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return total or 1  # 0 is reserved for a slot never written


class MotionWriter:
    """Publishes poses into the file a running scan reads.

    The file is created by :meth:`create` before the scan starts and held open
    for its length. Each pose goes into the slot the last one did not use,
    written whole with its check value, and is published by writing its version
    last: a reader takes the greater version whose check agrees, so a pose
    caught mid-write is passed over rather than played.

    Words are written in this machine's byte order. The reader compares the
    header's magic against its own spelling and swaps every word after it when
    they differ, so the two need not agree.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._fd = -1
        self._version = 0
        self._slot = SLOTS - 1

    @property
    def version(self) -> int:
        """The version of the last pose published; 0 before the first."""
        return self._version

    def create(self) -> MotionWriter:
        """Create the file at its full size and write its header.

        An existing file is replaced: a scan reads the poses of its own run.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = struct.pack("=4I", MAGIC, LAYOUT, SLOTS, _SLOT_BYTES)
        with self.path.open("wb") as f:
            f.write(header + bytes(_SIZE - _HEADER_BYTES))
        self._fd = os.open(self.path, os.O_RDWR)
        self._version = 0
        self._slot = SLOTS - 1
        return self

    def publish(self, rotation, *, rescan: bool = False) -> int:
        """Publish one pose and return its version.

        Parameters
        ----------
        rotation
            Nine values, row major, the physical frame from the logical one.
        rescan
            Whether the repetition that measured this pose should be played
            again.

        Raises
        ------
        ValueError
            If ``rotation`` is not nine values.
        RuntimeError
            If the file is not open.
        """
        values = [float(v) for v in rotation]
        if len(values) != 9:
            raise ValueError(f"a rotation is nine values, got {len(values)}")
        if self._fd < 0:
            raise RuntimeError(f"{self.path} is not open")

        version = self._version + 1
        slot = (self._slot + 1) % SLOTS
        turned = struct.unpack("=9I", struct.pack("=9f", *values))
        words = [version, 1 if rescan else 0, 0, *turned]
        words[_CHECK] = _check_of(words)

        at = _HEADER_BYTES + slot * _SLOT_BYTES
        # Everything but the version, then the version: writing it publishes
        # the slot, so it goes last and alone.
        os.pwrite(self._fd, struct.pack(f"={_SLOT_WORDS - 1}I", *words[1:]), at + _WORD)
        os.pwrite(self._fd, struct.pack("=I", version), at)
        self._version = version
        self._slot = slot
        return version

    def applied(self) -> tuple[int, int] | None:
        """Return the pose the scan played and the repetition it played it in.

        A rotation moves where the samples of a readout lie, so a
        reconstruction correcting the translation has to use the rotation the
        scan actually played rather than the one last published.

        Returns
        -------
        tuple of int, or None
            Version and repetition, or None where the scan has echoed nothing
            or the echo was caught mid-write.
        """
        if self._fd < 0:
            raise RuntimeError(f"{self.path} is not open")
        raw = os.pread(self._fd, _ECHO_WORDS * _WORD, _ECHO_OFFSET)
        if len(raw) != _ECHO_WORDS * _WORD:
            return None
        words = list(struct.unpack(f"={_ECHO_WORDS}I", raw))
        if words[_CHECK] != _check_of(words):
            return None
        return words[0], words[1]

    def close(self) -> None:
        """Close the file; safe on one that was never opened."""
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> MotionWriter:
        return self.create() if self._fd < 0 else self

    def __exit__(self, *_: object) -> None:
        self.close()
