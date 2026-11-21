"""Scan one wave-shuffling case on the virtual scanner at 7 T: scan.py NAME PLUGIN key=value ..."""

import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/user/pulserver-7t")
name, plugin, *pairs = sys.argv[1:]
d = Path(name).resolve()
d.mkdir(exist_ok=True)
from pulserver.host._blocks import format_limits
from pulserver.protocol import PROTOCOL_BEGIN, PROTOCOL_END

LIMITS = {
    "max_grad": 40.0,
    "grad_unit": "mT/m",
    "max_slew": 150.0,
    "slew_unit": "T/m/s",
    "B0": 7.0,
    "design_max_grad": 40.0 / 3**0.5,
    "design_max_slew": 150.0 / 3**0.5,
}
(d / "limits.txt").write_text(format_limits(LIMITS))
(d / "protocol.txt").write_text(
    PROTOCOL_BEGIN
    + "\n"
    + "\n".join(p.replace("=", ": ", 1) for p in pairs)
    + "\n"
    + PROTOCOL_END
    + "\n"
)
cmd = [
    sys.executable,
    "-c",
    "import sys; from pulserver._cli import main; sys.exit(main(sys.argv[1:]))",
    "scan",
    "--plugins",
    str(Path(__file__).resolve().parent / "plugins"),
    "--plugins",
    str(ROOT / "src/pulserver/_zoo/sequences"),
    "--plugin",
    plugin,
    "--protocol",
    str(d / "protocol.txt"),
    "--limits",
    str(d / "limits.txt"),
    "--store",
    str(d / "designs"),
    "--mrd",
    str(d / "raw.h5"),
    "--coil",
    "head8/head32",
    "--phantom",
    "brainweb",
    "--spacing",
    "2",
]
import os

cmd += os.environ.get("SCANARGS", "").split()
sys.exit(subprocess.run(cmd, cwd=d).returncode)
