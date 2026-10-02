"""The waveform XML a scanner's plotter writes, read into arrays."""

from __future__ import annotations

__all__ = ["PlayedWaveforms", "read_waveform_xml"]

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Which channel each sequencer of the file carries, by its id.
_CHANNELS = {
    0: "gx",
    1: "gy",
    2: "gz",
    3: "ssp",
    4: "rho",
    5: "theta",
    6: "omega",
}

#: One gauss per centimetre in millitesla per metre.
_GAUSS_PER_CM_IN_MT_PER_M = 10.0

#: The transmit phase spans a turn over a signed 24-bit converter, so one of
#: its counts is this many radians.
_PHASE_COUNT_IN_RAD = np.pi / 2**23


@dataclass
class PlayedWaveforms:
    """What a scanner played, as its plotter recorded it.

    One pair of arrays per channel: the time of each sample in microseconds,
    and its amplitude in the unit that channel is stored in. The gradients are
    in gauss per centimetre, and :meth:`gradient_mt_per_m` converts them. The
    transmit channels are stored as the converter's own numbers.
    :meth:`phase_rad` converts the phase, whose converter spans a turn. The
    magnitude's scale is the peak transmit field of the scan being played,
    which is not in the file, so nothing here invents one: a magnitude is
    compared by its shape.

    Attributes
    ----------
    end_time_us
        The length of the recording, as the file states it.
    channels
        ``(time_us, amplitude)`` per channel: ``gx``, ``gy`` and ``gz``,
        ``ssp``, and the transmit ``rho``, ``theta`` and ``omega``.
    titles
        What the file calls each sequencer, which names its core and hardware.
    """

    end_time_us: float
    channels: dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]] = field(
        default_factory=dict
    )
    titles: dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return a channel, or a pair of empty arrays where it played nothing."""
        return self.channels.get(
            name, (np.array([], dtype=np.int64), np.array([], dtype=np.float64))
        )

    def gradient_mt_per_m(
        self, name: str
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return a gradient channel in millitesla per metre."""
        time_us, amplitude = self.get(name)
        return time_us, amplitude * _GAUSS_PER_CM_IN_MT_PER_M

    def phase_rad(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return the transmit phase in radians, wrapped to one turn.

        The recorded phase runs the other way from the one a sequence states,
        so it is negated here and the two are read in the same sense.
        """
        time_us, counts = self.get("theta")
        return time_us, np.angle(np.exp(-1j * counts * _PHASE_COUNT_IN_RAD))


def read_waveform_xml(path: Path | str) -> PlayedWaveforms:
    """Read one waveform XML file.

    The file holds one ``sequencer`` per channel, each carrying its samples as
    lines of a time in microseconds and an amplitude.

    Raises
    ------
    ValueError
        If the file is not a pulse sequence recording, or declares a document
        type.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    _refuse_document_type(text, path)
    # S314: _refuse_document_type has rejected the declaration that entity
    # expansion and external entity resolution both need.
    root = ET.fromstring(text)  # noqa: S314
    if root.tag != "PulseSequence":
        raise ValueError(
            f"{path} is a <{root.tag}>, not the <PulseSequence> a plotter writes"
        )
    played = PlayedWaveforms(end_time_us=float(root.get("endTime") or 0.0))
    for sequencer in root.iter("sequencer"):
        try:
            number = int(sequencer.get("id"))
        except (TypeError, ValueError):
            continue
        name = _CHANNELS.get(number, f"sequencer{number}")
        data = sequencer.find("data")
        played.channels[name] = _samples(data.text if data is not None else "")
        played.titles[name] = sequencer.get("title", "")
    return played


def _refuse_document_type(text: str, path: Path | str) -> None:
    """Refuse a document type declaration before anything parses it.

    A recording is elements and text. A declared document type is what both
    entity expansion and external entity resolution need, and it may only
    appear in the prolog, so reading as far as the root element settles it.
    """
    at = 0
    while at < len(text):
        at = text.find("<", at)
        if at < 0:
            return
        if text.startswith("<?", at):
            closed = text.find("?>", at)
        elif text.startswith("<!--", at):
            closed = text.find("-->", at)
        elif text.startswith("<!DOCTYPE", at):
            raise ValueError(
                f"{path} declares a document type, which a waveform recording does not"
            )
        else:
            return
        if closed < 0:
            return
        at = closed + 2


def _samples(text: str) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Read the sample lines of one channel: a time in microseconds, an amplitude."""
    if not text:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    times: list[int] = []
    amplitudes: list[float] = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        times.append(int(float(parts[0])))
        amplitudes.append(float(parts[1]))
    return np.asarray(times, dtype=np.int64), np.asarray(amplitudes, dtype=np.float64)
