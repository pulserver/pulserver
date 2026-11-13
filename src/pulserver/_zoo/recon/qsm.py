"""Multi-echo 3D gradient echo with fast-qsm: RSS, T2* and QSM, SWI outside a phantom and DECOMPOSE for the brain, each its own series; the echoes are not sent.

Hot-fix 13 Nov: every echo after the first takes the coil sensitivities the
first echo estimated, instead of estimating its own from its calibration
readouts. One set of maps keeps the echo-to-echo phase evolution intact (maps
estimated per echo absorb part of each echo's phase) and avoids fitting maps to
the low-SNR late echoes; it also saves one NLINV per echo.
"""

import logging

from pulserver.recon.handlers import pics as _pics
from pulserver.recon.handlers.qsm import PLUGIN  # noqa: F401

_log = logging.getLogger("pulserver.recon")
_estimated_maps = _pics.coil_maps


def _first_echo_maps(context, data, *, estimate, required=True):
    """Return the maps the slice's first echo stored for a later echo; estimate them otherwise."""
    import torch

    buffer = data.data
    location = int(data.counters.get("slice", 0))
    polarity = int(data.counters.get("set", 0))
    own = (location, polarity) if polarity else location
    echo = int(buffer.reference.idx.contrast) if buffer is not None else 0
    if echo > 0 and own in context.coil_maps:
        _log.info(
            "echo %d of slice %d takes the coil sensitivities of the first echo",
            echo,
            location,
        )
        return torch.tensor(context.coil_maps[own].maps)
    return _estimated_maps(context, data, estimate=estimate, required=required)


_pics.coil_maps = _first_echo_maps


# The wave-CAIPI solve fits its maps by calling nlinv_estimate on each echo's
# wave-free calibration readouts directly; the first echo's are kept on the
# series' context and given to the later echoes.
import threading  # noqa: E402

from pulserver.recon.handlers.qsm import QsmRecon  # noqa: E402

_unit = threading.local()
_wave_image = QsmRecon.wave_image
_nlinv_estimate = _pics.nlinv_estimate


def _first_echo_wave_image(self, context, data):
    _unit.current = (context, int(data.data.reference.idx.contrast))
    try:
        return _wave_image(self, context, data)
    finally:
        _unit.current = None


def _first_echo_nlinv(kspace, traj=None, shape=None):
    current = getattr(_unit, "current", None)
    if current is None:
        return _nlinv_estimate(kspace, traj, shape)
    context, echo = current
    kept = getattr(context, "_first_echo_wave_maps", None)
    if (
        echo > 0
        and kept is not None
        and tuple(kept.shape[1:]) == tuple(kspace.shape[1:])
    ):
        _log.info(
            "echo %d takes the wave-CAIPI coil sensitivities of the first echo", echo
        )
        return kept.clone()
    maps = _nlinv_estimate(kspace, traj, shape)
    if echo == 0:
        context._first_echo_wave_maps = maps
    return maps


QsmRecon.wave_image = _first_echo_wave_image
_pics.nlinv_estimate = _first_echo_nlinv
