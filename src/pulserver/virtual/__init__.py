"""A virtual scanner: an IR cache played, a phantom acquired along it, the series sent as the scanner sends it."""

from ._bloch import simulate
from ._client import send
from ._export import export
from ._phantom import Ellipse, Phantom
from ._scanner import acquire, trajectory

__all__ = ["Ellipse", "Phantom", "acquire", "export", "send", "simulate", "trajectory"]
