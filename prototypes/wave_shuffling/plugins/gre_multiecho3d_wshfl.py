"""gre_multiecho3d loaded with wave-shuffling options."""
from pathlib import Path as _Path

from pulserver.design._options import loading_options as _loading

_SOURCE = _Path("/home/user/pulserver-7t/src/pulserver/_zoo/sequences/gre_multiecho3d.py")
with _loading({"wave": True}):
    exec(compile(_SOURCE.read_text(), str(_SOURCE), "exec"))
