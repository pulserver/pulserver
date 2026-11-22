"""fse3d shuffled with centre-out wave-free calibration trains (pypulseqpp worktree claude/wave-shuffling-prototype)."""
import importlib.util as _u
import sys as _sys
from pathlib import Path as _Path

_NAME = "pypulseqpp.sequences.sequence.fse3D_sequence"
_spec = _u.spec_from_file_location(_NAME, "/home/user/pypulseqpp-wshfl/examples/sequence/fse3D_sequence.py")
_mod = _u.module_from_spec(_spec); _sys.modules[_NAME] = _mod; _spec.loader.exec_module(_mod)

from pulserver.design._options import loading_options as _loading

_SOURCE = _Path("/home/user/pulserver-7t/src/pulserver/_zoo/sequences/fse3d.py")
_text = _SOURCE.read_text().replace(
    'references = -(-calibrating // lengths[0]) if waved(a) else 0',
    'references = -(-calibrating // lengths[0]) if waved(a) or a["ordering"] == "shuffling" else 0',
)
assert "or a[\"ordering\"] == \"shuffling\"" in _text
with _loading({"ordering": "shuffling"}):
    exec(compile(_text, str(_SOURCE), "exec"))
