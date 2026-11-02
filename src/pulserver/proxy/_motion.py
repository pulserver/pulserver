"""The pose a tracker leaves for a running scan, in a file both sides hold open.

The tracker runs on the host and the scan on the sequencer; they share a
file of fixed size, created before the scan starts and written in place. It is
never renamed, truncated or extended, so a descriptor opened once stays valid
for the whole scan.

The file holds an absolute pose, not a change of pose: a reader is not
promised to see every write, and an absolute pose stays correct when versions
are missed.

Only the rotation crosses. A translation is corrected as a phase on the data,
which the reconstruction applies.
"""

from __future__ import annotations

import logging
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger("pulserver.proxy")

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


#: The waveform a reconstruction states a pose in. Reserved: a stream carrying
#: it is not carrying a physiological trace.
POSE_WAVEFORM_ID = 1010


@dataclass(frozen=True)
class Pose:
    """Where a reconstruction has found the object, and what to do about it.

    ``rotation`` is row major, the physical frame from the logical one.
    ``translation_m`` is what the reconstruction applies to its own data; it
    does not cross to the sequencer, which corrects orientation alone.
    """

    rotation: tuple[float, ...]
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rescan: bool = False


def pose_of(waveform: Any) -> Pose | None:
    """Read a pose out of a waveform, or None where it holds none.

    Returns
    -------
    Pose or None
        None where the waveform is not a pose, or holds too few values to be
        one: a stream is not stopped over a malformed pose, the last one
        published simply stands.
    """
    if int(getattr(waveform, "waveform_id", -1)) != POSE_WAVEFORM_ID:
        return None
    words = np.asarray(waveform.data, dtype=np.uint32).ravel()
    if words.size < 13:
        return None
    values = words.view(np.float32)
    return Pose(
        rotation=tuple(float(v) for v in values[:9]),
        translation_m=(
            float(values[9]),
            float(values[10]),
            float(values[11]),
        ),
        rescan=bool(words[12]),
    )


def pose_waveform(pose: Pose) -> Any:
    """Return the waveform a reconstruction states ``pose`` in."""
    import ismrmrd

    values = np.zeros(13, dtype=np.float32)
    values[:9] = np.asarray(pose.rotation, dtype=np.float32)
    values[9:12] = np.asarray(pose.translation_m, dtype=np.float32)
    words = values.view(np.uint32).copy()
    words[12] = 1 if pose.rescan else 0
    waveform = ismrmrd.Waveform.from_array(words.reshape(1, -1))
    waveform.waveform_id = POSE_WAVEFORM_ID
    return waveform


def _blank(path: Path) -> int:
    """Write an empty pose file at ``path`` in place; return it open for writing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("=4I", MAGIC, LAYOUT, SLOTS, _SLOT_BYTES)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        os.pwrite(fd, header + bytes(_SIZE - _HEADER_BYTES), 0)
        os.ftruncate(fd, _SIZE)
    except BaseException:
        os.close(fd)
        raise
    return fd


#: The first bytes of a sequence file in the Pulseq binary format.
_BINARY_MAGIC = b"\x01pulseq"


def asks_for_motion_correction(entry: Path) -> bool:
    """Whether a sequence file sets ``EnablePmc``, read from its definitions.

    A file in the Pulseq binary format, as the design service writes, is read
    through pypulseqpp; a text file is scanned for its definitions alone.
    """
    try:
        with Path(entry).open("rb") as f:
            binary = f.read(len(_BINARY_MAGIC)) == _BINARY_MAGIC
    except OSError:
        return False
    if binary:
        import pypulseqpp as pp

        seq = pp.Sequence()
        seq.read(str(entry))
        value = seq.definitions.get("EnablePmc")
        if value is None:
            return False
        return bool(np.any(np.asarray(value, dtype=float) != 0))
    try:
        with Path(entry).open("r", errors="replace") as f:
            inside = False
            for line in f:
                stripped = line.strip()
                if stripped.startswith("["):
                    if inside:
                        return False
                    inside = stripped.startswith("[DEFINITIONS]")
                    continue
                if inside and stripped.startswith("EnablePmc"):
                    parts = stripped.split()
                    return len(parts) > 1 and parts[1] not in ("0", "0.0")
    except OSError:
        return False
    return False


def prepare(scan: Path | str, entry: Path | str) -> Path | None:
    """Create the empty pose file of a scan whose sequence asks for motion correction.

    The scan opens the file once, when it starts, so it has to exist before
    then: it is created with the design, empty and at version 0, and the
    series' reconstruction clears it again in place when it starts. A file
    already there is left as it is.

    Parameters
    ----------
    scan : Path | str
        The scan's directory of the store.
    entry : Path | str
        The first sequence file of the design.

    Returns
    -------
    Path or None
        The pose file, or None where the sequence does not ask for it.
    """
    if not asks_for_motion_correction(Path(entry)):
        return None
    path = Path(scan) / FILENAME
    if not path.is_file():
        os.close(_blank(path))
    return path


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
        """Create the file at its full size and write its header, at version 0.

        An existing file is cleared in place, never truncated or replaced: a
        scan reads the poses of its own run, through a descriptor it may
        already hold.
        """
        self._fd = _blank(self.path)
        self._version = 0
        self._slot = SLOTS - 1
        return self

    def publish(self, rotation, *, rescan: bool = False) -> int:
        """Publish one pose and return its version.

        Parameters
        ----------
        rotation : sequence of float
            Nine values, row major, the physical frame from the logical one.
        rescan : bool, default=False
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


#: The motion log beside the pose file, and its plot.
LOG_FILENAME = "motion.csv"
PLOT_FILENAME = "motion.png"

_LOG_COLUMNS = (
    "pose",
    "time_s",
    "rx_deg",
    "ry_deg",
    "rz_deg",
    "tx_mm",
    "ty_mm",
    "tz_mm",
    "rescan",
)


def euler_deg(rotation: tuple[float, ...]) -> tuple[float, float, float]:
    """Rotations about x, y and z (degrees) of a row-major rotation, ``R = Rz @ Ry @ Rx``."""
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    rx = math.atan2(r[2, 1], r[2, 2])
    ry = -math.asin(max(-1.0, min(1.0, r[2, 0])))
    rz = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


class MotionLog:
    """The poses a scan was given, as a table and a plot beside its pose file.

    Each pose is appended to ``motion.csv`` as it is published: its index, the
    seconds since the first, the rotations about x, y and z (degrees,
    ``R = Rz @ Ry @ Rx``), the translation (mm) and the rescan flag. The table
    is written as ``motion.csv.part`` while the scan runs, so that whatever
    exports the scan's directory waits for it, and on :meth:`close` the six
    parameters are plotted against time into ``motion.png`` and the table is
    renamed. Nothing raises: a table that cannot be written is logged.
    """

    def __init__(self, directory: Path | str) -> None:
        self.path = Path(directory) / LOG_FILENAME
        self._part = self.path.with_name(self.path.name + ".part")
        self._rows: list[tuple[float, ...]] = []
        self._start: float | None = None
        self._file = None
        try:
            self._file = self._part.open("w")
            self._file.write(",".join(_LOG_COLUMNS) + "\n")
            self._file.flush()
        except OSError as error:
            _log.error("the motion log %s cannot be written: %s", self._part, error)

    def add(self, pose: Pose) -> None:
        now = time.monotonic()
        if self._start is None:
            self._start = now
        row = (
            float(len(self._rows)),
            now - self._start,
            *euler_deg(pose.rotation),
            *(1e3 * float(value) for value in pose.translation_m),
            float(pose.rescan),
        )
        self._rows.append(row)
        if self._file is not None:
            self._file.write(
                f"{int(row[0])},{row[1]:.3f},"
                + ",".join(f"{value:.4f}" for value in row[2:8])
                + f",{int(row[8])}\n"
            )
            self._file.flush()

    def close(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None
        if self._rows:
            self._plot()
        try:
            self._part.replace(self.path)
        except OSError as error:
            _log.error("the motion log %s cannot be kept: %s", self.path, error)

    def _plot(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        rows = np.asarray(self._rows)
        fig, (rot, trans) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
        for column, name in zip(range(2, 5), "xyz", strict=True):
            rot.plot(rows[:, 1], rows[:, column], label=f"about {name}")
        for column, name in zip(range(5, 8), "xyz", strict=True):
            trans.plot(rows[:, 1], rows[:, column], label=name)
        rot.set_ylabel("rotation (deg)")
        trans.set_ylabel("translation (mm)")
        trans.set_xlabel("time since the first pose (s)")
        rot.legend(loc="upper left")
        trans.legend(loc="upper left")
        fig.tight_layout()
        try:
            fig.savefig(self.path.with_name(PLOT_FILENAME), dpi=100)
        except OSError as error:
            _log.error("the motion plot cannot be written: %s", error)
        plt.close(fig)

    def __enter__(self) -> MotionLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
