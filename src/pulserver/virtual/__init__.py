"""A virtual scanner: an IR cache played, a phantom acquired along it, the series sent as the scanner sends it."""

from ._bloch import simulate
from ._brainweb import BrainWeb
from ._client import record, send
from ._export import export
from ._phantom import Ellipse, Phantom
from ._scanner import acquire, trajectory
from ._stream import SAMPLE_RATE, Chunk, Scan

__all__ = [
    "SAMPLE_RATE",
    "BrainWeb",
    "Chunk",
    "Ellipse",
    "Phantom",
    "Scan",
    "acquire",
    "export",
    "record",
    "send",
    "simulate",
    "trajectory",
]
