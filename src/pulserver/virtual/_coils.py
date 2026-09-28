"""RF coils of the virtual scanner, with the sensitivities of BART's coil models."""

from __future__ import annotations

__all__ = ["COILS", "Coil"]

import functools
import tempfile
from dataclasses import dataclass

import numpy as np

#: Side, in m, of the cube centred on the isocentre that the unit field of view
#: of BART's coil models is taken to span.
MODEL_FOV = 0.256

#: Samples per axis of the grid a model is evaluated on and interpolated from.
_SAMPLES = 32


@dataclass(frozen=True)
class Coil:
    """The RF coils of an exam, fixed in the physical frame: the coil pulses are transmitted on and the coil the signal is received by.

    ``name`` is ``transmit/receive``, or one coil's name where it does both.
    Each side is BART's coil model (``HEAD_2D_8CH`` or ``HEAD_3D_64CH``) and
    the number of its channels the coil takes, the first ones, as ``phantom
    -S`` selects them; a side without a model is one channel of unit
    sensitivity everywhere, as a body coil is taken to be. A model is sampled
    by bartorch on first use, which the ``coils`` extra installs, over
    :data:`MODEL_FOV` along the physical axes, and interpolated linearly;
    ``HEAD_2D_8CH`` is constant along z.

    Receive sensitivities are scaled to a root sum of squares of 1 at the
    isocentre. Transmit sensitivities are the complex conjugates of the
    receive sensitivities of the transmit side's model, the quasi-static limit
    of reciprocity, scaled so that the :attr:`default_shim` plays a pulse at
    its nominal amplitude at the isocentre.
    """

    name: str
    transmit_model: tuple[str, int] | None = None
    receive_model: tuple[str, int] | None = None

    @property
    def transmit_channels(self) -> int:
        """Channels a pulse is played on."""
        return 1 if self.transmit_model is None else self.transmit_model[1]

    @property
    def receive_channels(self) -> int:
        """Coils a readout is received by."""
        return 1 if self.receive_model is None else self.receive_model[1]

    @property
    def default_shim(self) -> np.ndarray | None:
        """Unit channel weights that bring every transmit channel into phase at the isocentre; None for one channel.

        A single-channel pulse played without an RF shim is played on every
        channel with these weights. They are an RF shim's weights, which scale
        the waveform in Pulseq's convention, the conjugate of the field.
        """
        if self.transmit_model is None:
            return None
        return np.exp(1j * np.angle(_isocentre(_transmit(*self.transmit_model))))

    def transmit(self, points: np.ndarray) -> np.ndarray | None:
        """Return each channel's transmit sensitivity at ``(n, 3)`` physical points, in m: ``(n, channels)``; None for one uniform channel."""
        if self.transmit_model is None:
            return None
        return _interpolated(_transmit(*self.transmit_model), points)

    def receive(self, points: np.ndarray) -> np.ndarray | None:
        """Return each coil's receive sensitivity at ``(n, 3)`` physical points, in m: ``(n, coils)``; None for one uniform coil."""
        if self.receive_model is None:
            return None
        return _interpolated(_receive(*self.receive_model), points)


COILS = {
    coil.name: coil
    for coil in (
        Coil("body"),
        Coil("body/head48", receive_model=("HEAD_3D_64CH", 48)),
        Coil("head8/head32", ("HEAD_2D_8CH", 8), ("HEAD_3D_64CH", 32)),
    )
}


@functools.cache
def _receive(model: str, channels: int) -> np.ndarray:
    maps = _sampled(model, channels)
    return maps / np.sqrt(np.sum(np.abs(_isocentre(maps)) ** 2))


@functools.cache
def _transmit(model: str, channels: int) -> np.ndarray:
    maps = np.conj(_sampled(model, channels))
    return maps / np.sum(np.abs(_isocentre(maps)))


def _sampled(model: str, channels: int) -> np.ndarray:
    """Return the first ``channels`` sensitivities of BART's ``model``: ``(channels, z, y, x)``, z of size 1 for a 2D model.

    Sample ``i`` along an axis lies at ``(i - n // 2) / n`` of the model's
    field of view, as BART's ``phantom`` places it.
    """
    try:
        from bartorch.tools import simulate
    except ImportError as error:
        raise ImportError(
            "BART's coil models are sampled by bartorch: pip install 'pulserver[coils]'"
        ) from error
    three = model == "HEAD_3D_64CH"
    maps = simulate.phantom((_SAMPLES,) * (3 if three else 2), S=channels, coil=model)
    return np.asarray(maps.cpu().numpy(), dtype=np.complex64).reshape(
        channels, -1, _SAMPLES, _SAMPLES
    )


def _isocentre(maps: np.ndarray) -> np.ndarray:
    return maps[(slice(None), *(size // 2 for size in maps.shape[1:]))]


def _interpolated(
    maps: np.ndarray, points: np.ndarray, chunk: int = 1 << 16
) -> np.ndarray:
    """Return ``maps`` ``(channels, z, y, x)`` interpolated trilinearly at ``(n, 3)`` physical points; the edge value beyond the grid.

    The result is complex128, as :class:`pypulseqpp.Isochromats` takes it,
    in a temporary file mapped into memory: its pages belong to the file,
    which the operating system writes back rather than holding in the
    process's memory. ``TMPDIR`` names where the file is made.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    table = maps.reshape(maps.shape[0], -1).T
    out = _mapped((len(points), maps.shape[0]))
    for start in range(0, len(points), chunk):
        part = points[start : start + chunk]
        corners, weights = [], []
        for axis, size in zip((2, 1, 0), maps.shape[1:], strict=True):
            index = np.clip(part[:, axis] / MODEL_FOV * size + size // 2, 0, size - 1)
            low = np.minimum(np.floor(index).astype(np.intp), max(size - 2, 0))
            corners.append((low, np.minimum(low + 1, size - 1)))
            weights.append((1.0 - (index - low), index - low))
        total = np.zeros((len(part), maps.shape[0]), dtype=np.complex64)
        for z in (0, 1):
            for y in (0, 1):
                for x in (0, 1):
                    flat = np.ravel_multi_index(
                        (corners[0][z], corners[1][y], corners[2][x]), maps.shape[1:]
                    )
                    weight = weights[0][z] * weights[1][y] * weights[2][x]
                    total += weight.astype(np.float32)[:, None] * table[flat]
        out[start : start + chunk] = total
    return out


def _mapped(shape: tuple[int, int]) -> np.ndarray:
    """Return a complex128 array of ``shape`` in an unlinked temporary file, which is freed with the array."""
    if 0 in shape:
        return np.empty(shape, dtype=np.complex128)
    with tempfile.TemporaryFile() as file:
        return np.memmap(file, dtype=np.complex128, mode="w+", shape=shape)
