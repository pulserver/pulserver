"""A virtual scanner: an IR cache played, a phantom acquired along it, the series sent as the scanner sends it."""

from ._client import send
from ._phantom import Ellipse, Phantom
from ._scanner import acquire, trajectory

__all__ = ["Ellipse", "Phantom", "acquire", "send", "trajectory"]
