"""A virtual scanner: an IR cache played on a phantom's tissue by the Fourier engine, the series sent as the scanner sends it."""

from . import _coils
from ._brainweb import BrainWeb
from ._client import record, send
from ._coils import Coil
from ._console import Console
from ._export import export
from ._localizer import localizer
from ._phantom import Ellipse, Phantom
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
    "Phantom",
    "Scan",
    "Tissue",
    "export",
    "localizer",
    "record",
    "send",
    "simulate",
]


def __getattr__(name: str):
    """Import the Fourier engine, and torch with it, the first time it is named."""
    if name in ("FourierPlayer", "simulate"):
        from . import _fourier

        return getattr(_fourier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
