"""A virtual scanner: an IR cache played, a phantom acquired along it, the series sent as the scanner sends it."""

from . import _coils
from ._bloch import simulate
from ._brainweb import BrainWeb
from ._client import record, send
from ._coils import Coil
from ._console import Console
from ._export import export
from ._localizer import localizer
from ._phantom import Ellipse, Phantom
from ._scanner import acquire, trajectory
from ._stream import SAMPLE_RATE, Chunk, Scan

#: The virtual scanner's coils, by name: a body coil, an 8-channel transmit and
#: receive head coil for parallel transmission, and 32- and 48-channel receive
#: head arrays transmitting on the body coil.
COILS = _coils.COILS

__all__ = [
    "COILS",
    "SAMPLE_RATE",
    "BrainWeb",
    "Chunk",
    "Coil",
    "Console",
    "Ellipse",
    "Phantom",
    "Scan",
    "acquire",
    "export",
    "localizer",
    "record",
    "send",
    "simulate",
    "trajectory",
]
