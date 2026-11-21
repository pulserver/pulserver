"""gre_multiecho3d loaded as shipped."""

from pathlib import Path as _Path

from pulserver.design._options import loading_options as _loading

_SOURCE = _Path(
    "/home/user/pulserver-7t/src/pulserver/_zoo/sequences/gre_multiecho3d.py"
)
with _loading({}):
    exec(compile(_SOURCE.read_text(), str(_SOURCE), "exec"))
