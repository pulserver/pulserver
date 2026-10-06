"""A virtual scanner: an IR cache played, a phantom acquired along it, the series sent as the scanner sends it."""

from . import _coils
from ._bloch import simulate
from ._brainweb import BrainWeb
from ._client import record, send
from ._coils import Coil
from ._console import Console
from ._export import export
from ._isochromats import Isochromats, Repetitions
from ._localizer import localizer
from ._motion import RigidMotion
from ._phantom import Ellipse, Phantom
from ._region import Slabs, excited
from ._scanner import acquire, trajectory
from ._stream import SAMPLE_RATE, Chunk, Scan
from ._tissue import Tissue

#: The virtual scanner's coils, by name, ``transmit/receive``: the body coil
#: both ways, the body coil transmitting to a 48-channel receive head array,
#: and an 8-channel head coil for parallel transmission with a 32-channel
#: receive head array.
COILS = _coils.COILS

__all__ = [
    "COILS",
    "SAMPLE_RATE",
    "BrainWeb",
    "Chunk",
    "Coil",
    "Console",
    "Ellipse",
    "FourierPlayer",
    "Isochromats",
    "Phantom",
    "Repetitions",
    "RigidMotion",
    "Scan",
    "Slabs",
    "Tissue",
    "acquire",
    "excited",
    "export",
    "localizer",
    "record",
    "send",
    "simulate",
    "trajectory",
]


def __getattr__(name: str):
    """Import the Fourier engine, which needs torch, the first time it is named."""
    if name == "FourierPlayer":
        from ._fourier import FourierPlayer

        return FourierPlayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
