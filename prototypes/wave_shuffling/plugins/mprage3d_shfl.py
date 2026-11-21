"""mprage3d loaded with shuffling only."""

from pathlib import Path as _Path

from pulserver.design._options import loading_options as _loading

_SOURCE = _Path("/home/user/pulserver-7t/src/pulserver/_zoo/sequences/mprage3d.py")
with _loading({"ordering": "shuffling"}):
    exec(compile(_SOURCE.read_text(), str(_SOURCE), "exec"))
