"""A host daemon in a subprocess, for tests that need a generated revision."""

import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pulserver.host import SessionKey
from pulserver.host.client import HostClient

PLUGINS = Path(__file__).parent / "plugins"
FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
# The limits the IR fixtures convert under.
FIXTURE_LIMITS = {
    "max_grad": 40.0,
    "grad_unit": "mT/m",
    "max_slew": 170.0,
    "slew_unit": "T/m/s",
    "B0": 3.0,
    "rf_raster_time": 1e-6,
    "grad_raster_time": 1e-5,
    "adc_raster_time": 1e-7,
    "block_duration_raster": 1e-5,
}
GE_IR = {"ir_vendor": 2, "ir_label_column_map": "8 0 6", "ir_cache_ext": ".pge"}
LIMITS = {
    "max_grad": 40.0,
    "grad_unit": "mT/m",
    "max_slew": 150.0,
    "slew_unit": "T/m/s",
}
DAY = 20711


class Daemon:
    def __init__(self, base: Path, plugins: Path = PLUGINS) -> None:
        self.base = base
        self.plugins = plugins
        self._socket_dir = Path(tempfile.mkdtemp(prefix="ps"))
        self.socket = self._socket_dir / "s"
        self._process = None

    def start(self) -> None:
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-X",
                "faulthandler",
                "-m",
                "pulserver.host",
                "--base",
                str(self.base),
                "--socket",
                str(self.socket),
                "--plugins",
                str(self.plugins),
                "--workers",
                "1",
            ]
        )
        deadline = time.monotonic() + 30
        while not self.socket.exists():
            if self._process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("the host daemon did not start")
            time.sleep(0.05)

    def stop(self) -> None:
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # faulthandler writes every thread's stack to stderr on SIGABRT.
            self._process.send_signal(signal.SIGABRT)
            self._process.wait(timeout=10)
            raise
        self.socket.unlink(missing_ok=True)

    def client(self, pid: int) -> HostClient:
        return HostClient(self.socket, SessionKey(pid=pid, day=DAY))

    def cleanup(self) -> None:
        if self._process.poll() is None:
            self.stop()
        shutil.rmtree(self._socket_dir, ignore_errors=True)
