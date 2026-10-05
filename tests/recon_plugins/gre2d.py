"""A Cartesian FFT of the lines in arrival order, one image per slice, held and timed where the client's config asks.

``gate`` names a file the reconstruction waits for before it starts, so a test
decides when a slot frees; it touches ``<gate>.waiting`` once it is holding.
``trace`` names a directory it writes ``<pid>.json`` into, with the interval it
held its slot for. ``delay`` is seconds the reconstruction sleeps before it
returns its image.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from pulserver.mrd import AcquisitionFlag, center_crop, coil_combine
from pulserver.recon import ReconPlugin, ReconResult

HOLD_SECONDS = 0.5
GATE_TIMEOUT = 60.0


class TracedFft(ReconPlugin):
    def __init__(self):
        super().__init__(
            triggers={"imaging": AcquisitionFlag.LAST_IN_SLICE},
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
            buffered=False,
        )

    def startup(self, context):
        super().startup(context)
        self.trace = _path(context.config, "trace")
        self.gate = _path(context.config, "gate")
        self.delay = float(_parameter(context.config, "delay") or 0.0)
        self.started = time.time()
        if self.gate is not None:
            _wait_for(self.gate)
        elif self.trace is not None:
            time.sleep(HOLD_SECONDS)

    def recon(self, context, branch, data):
        result = _image(data.acquisitions, context.header)
        time.sleep(self.delay)
        if self.trace is not None:
            path = self.trace / f"{os.getpid()}.json"
            path.write_text(json.dumps({"start": self.started, "end": time.time()}))
        return result


PLUGIN = TracedFft()


def _image(lines, header):
    if not lines:
        return None
    kspace = np.stack([line.data for line in lines], axis=-1)
    axes = (1, 2)
    image = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes), axes=axes
    )
    image = coil_combine(image, coil_axis=0)
    matrix = header.encoding[0].reconSpace.matrixSize
    shape = (
        min(int(matrix.x or image.shape[0]), image.shape[0]),
        min(int(matrix.y or image.shape[1]), image.shape[1]),
    )
    return ReconResult(np.array(center_crop(image, shape)).transpose())


def _parameter(config, name):
    parameters = config.get("parameters") if isinstance(config, dict) else None
    return parameters.get(name) if isinstance(parameters, dict) else None


def _path(config, name):
    value = _parameter(config, name)
    return None if value is None else Path(value)


def _wait_for(gate):
    gate.with_name(gate.name + ".waiting").touch()
    deadline = time.monotonic() + GATE_TIMEOUT
    while not gate.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
