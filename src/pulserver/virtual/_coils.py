"""RF coils of the virtual scanner, with the sensitivities of BART's coil models or of each coil's field maps."""

from __future__ import annotations

__all__ = ["COILS", "Coil", "coils"]

import functools
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .._accelerators import require

#: Side, in m, of the cube centred on the isocentre that the unit field of view
#: of BART's coil models is taken to span.
MODEL_FOV = 0.256

#: Samples per axis of the grid a model is evaluated on and interpolated from.
_SAMPLES = 32

#: Largest relative difference allowed between the frequency a coil's maps
#: were solved at and the Larmor frequency of the magnet they are used in.
_FREQUENCY_TOLERANCE = 0.01

#: Head SAR and local SAR limits of a transmit coil, W/kg: the IEC 60601-2-33
#: normal operating mode's for head SAR and for local SAR in head and trunk.
HEAD_SAR_LIMIT = 3.2
LOCAL_SAR_LIMIT = 10.0


class _Grid(NamedTuple):
    """Sensitivities ``(channels, z, y, x)``; sample ``i`` along an axis lies ``(i - centre) * step`` m from the isocentre."""

    values: np.ndarray
    centre: np.ndarray
    step: np.ndarray


@dataclass(frozen=True)
class Coil:
    """The RF coils of an exam, fixed in the physical frame: the coil pulses are transmitted on and the coil the signal is received by.

    ``name`` is ``transmit/receive``, or one coil's name where it does both.
    Each side is BART's coil model (``HEAD_2D_8CH`` or ``HEAD_3D_64CH``) and
    the number of its channels the coil takes, the first ones, as ``phantom
    -S`` selects them, or the file of the coil's field maps :func:`coils`
    reads; a side without either is one channel of unit sensitivity
    everywhere, as a body coil is taken to be. A model is sampled by bartorch
    on first use, which the ``coils`` extra installs, over :data:`MODEL_FOV`
    along the physical axes; ``HEAD_2D_8CH`` is constant along z. Either is
    interpolated linearly.

    Receive sensitivities are scaled to a root sum of squares of 1 at the
    isocentre, and transmit sensitivities so that the :attr:`default_shim`
    plays a pulse at its nominal amplitude at the isocentre. A model's
    transmit sensitivities are the complex conjugates of its receive
    sensitivities, the quasi-static limit of reciprocity.
    """

    name: str
    transmit_model: tuple[str, int] | Path | None = None
    receive_model: tuple[str, int] | Path | None = None

    @property
    def transmit_channels(self) -> int:
        """Channels a pulse is played on."""
        return _channels(self.transmit_model)

    @property
    def receive_channels(self) -> int:
        """Coils a readout is received by."""
        return _channels(self.receive_model)

    @property
    def default_shim(self) -> np.ndarray | None:
        """Unit channel weights that bring every transmit channel into phase at the isocentre; None for one channel.

        A single-channel pulse played without an RF shim is played on every
        channel with these weights. They are an RF shim's weights, which scale
        the waveform in Pulseq's convention, the conjugate of the field.
        """
        if self.transmit_model is None:
            return None
        return np.exp(1j * np.angle(_isocentre(_transmit(self.transmit_model))))

    def transmit(self, points: np.ndarray) -> np.ndarray | None:
        """Return each channel's transmit sensitivity at ``(n, 3)`` physical points, in m: ``(n, channels)``; None for one uniform channel."""
        if self.transmit_model is None:
            return None
        return _interpolated(_transmit(self.transmit_model), points)

    def receive(self, points: np.ndarray) -> np.ndarray | None:
        """Return each coil's receive sensitivity at ``(n, 3)`` physical points, in m: ``(n, coils)``; None for one uniform coil."""
        if self.receive_model is None:
            return None
        return _interpolated(_receive(self.receive_model), points)

    def limits(self) -> dict[str, str]:
        """Return the VOP entries of the check limits a design is made under in this coil; none unless its transmit maps have VOPs beside them.

        ``vop_file`` is ``<coil>_vops.npz`` beside the transmit maps
        ``<coil>.npz``; ``vop_drive_per_hz`` the drive of every channel, in
        the unit of the maps and the VOPs, per Hz of a pulse's amplitude,
        with which the channels' rotating fields, half their ``minus``, add
        up to the pulse's nominal amplitude at the isocentre through the
        :attr:`default_shim`; ``vop_default_shim`` that shim, a magnitude
        and a phase per channel; and ``vop_head_limit`` and
        ``vop_local_limit``, :data:`HEAD_SAR_LIMIT` and
        :data:`LOCAL_SAR_LIMIT`.
        """
        maps = self.transmit_model
        if not isinstance(maps, Path):
            return {}
        vops = maps.with_name(f"{maps.stem}_vops.npz")
        if not vops.is_file():
            return {}
        phases = np.angle(self.default_shim).tolist()
        return {
            "vop_file": str(vops),
            "vop_drive_per_hz": repr(_drive_per_hz(maps)),
            "vop_default_shim": " ".join(f"1.0 {phase!r}" for phase in phases),
            "vop_head_limit": repr(HEAD_SAR_LIMIT),
            "vop_local_limit": repr(LOCAL_SAR_LIMIT),
        }


COILS = {
    coil.name: coil
    for coil in (
        Coil("body"),
        Coil("body/head48", receive_model=("HEAD_3D_64CH", 48)),
        Coil("head8/head32", ("HEAD_2D_8CH", 8), ("HEAD_3D_64CH", 32)),
    )
}


def coils(
    fields: Path | str | None = None, *, field_t: float | None = None
) -> dict[str, Coil]:
    """Return the virtual scanner's coils by name: :data:`COILS`, or the same coils with the field maps in ``fields``.

    Each coil of a name has its maps in ``<coil>.npz``, as mariepy's
    ``maps.write`` writes them: the circular components ``plus`` and
    ``minus``, mu0 (Hx + j Hy) and mu0 (Hx - j Hy) with the time dependence
    exp(+j omega t), of every channel's field over a body in the physical
    frame, zero outside the body's mask. In a static field along +z, the
    field a channel transmits is the complex conjugate of its ``minus``, and
    what it receives is weighted by the complex conjugate of its ``plus``.
    Voxels outside the mask take the value of the nearest voxel inside.

    Raises
    ------
    FileNotFoundError
        If ``fields`` lacks a coil's maps.
    ValueError
        If maps were solved at a frequency other than the Larmor frequency of
        ``field_t``, in T.
    """
    if fields is None:
        return COILS
    directory = Path(fields)
    mapped = {}
    for name in COILS:
        transmit, _, receive = name.partition("/")
        mapped[name] = Coil(
            name,
            directory / f"{transmit}.npz",
            directory / f"{receive or transmit}.npz",
        )
    if field_t is not None:
        import pypulseqpp as pp

        larmor = pp.Opts().gamma * field_t
        sides = {side for name in COILS for side in name.split("/")}
        for path in sorted(directory / f"{side}.npz" for side in sides):
            solved = float(_metadata(path)["frequency_hz"])
            if abs(solved - larmor) > _FREQUENCY_TOLERANCE * larmor:
                raise ValueError(
                    f"{path} was solved at {solved / 1e6:.2f} MHz, and the magnet's "
                    f"Larmor frequency is {larmor / 1e6:.2f} MHz"
                )
    return mapped


def _channels(side: tuple[str, int] | Path | None) -> int:
    if side is None:
        return 1
    if isinstance(side, tuple):
        return side[1]
    return len(_metadata(side)["channels"])


@functools.cache
def _receive(side: tuple[str, int] | Path) -> _Grid:
    grid = _model(*side) if isinstance(side, tuple) else _map(side, "plus")
    return _scaled(grid, np.sqrt(np.sum(np.abs(_isocentre(grid)) ** 2)))


@functools.cache
def _transmit(side: tuple[str, int] | Path) -> _Grid:
    if isinstance(side, tuple):
        grid = _model(*side)
        grid = grid._replace(values=np.conj(grid.values))
    else:
        grid = _map(side, "minus")
    return _scaled(grid, np.sum(np.abs(_isocentre(grid))))


@functools.cache
def _drive_per_hz(maps: Path) -> float:
    import pypulseqpp as pp

    minus = np.sum(np.abs(_isocentre(_map(maps, "minus"))))
    return 2.0 / (pp.Opts().gamma * float(minus))


def _scaled(grid: _Grid, norm: float) -> _Grid:
    return grid._replace(values=grid.values / np.float32(norm))


def _model(model: str, channels: int) -> _Grid:
    maps = _sampled(model, channels)
    sizes = np.array(maps.shape[:0:-1])
    return _Grid(maps, sizes // 2, MODEL_FOV / sizes)


def _sampled(model: str, channels: int) -> np.ndarray:
    """Return the first ``channels`` sensitivities of BART's ``model``: ``(channels, z, y, x)``, z of size 1 for a 2D model.

    Sample ``i`` along an axis lies at ``(i - n // 2) / n`` of the model's
    field of view, as BART's ``phantom`` places it.
    """
    try:
        from bartorch.tools import phantom
    except ImportError as error:
        raise ImportError(
            "BART's coil models are sampled by bartorch: pip install 'pulserver[coils]'"
        ) from error
    three = model == "HEAD_3D_64CH"
    maps = phantom((_SAMPLES,) * (3 if three else 2), S=channels, coil=model)
    return np.asarray(maps.cpu().numpy(), dtype=np.complex64).reshape(
        channels, -1, _SAMPLES, _SAMPLES
    )


def _map(path: Path, component: str) -> _Grid:
    """Return the complex conjugate of a map file's ``plus`` or ``minus``, each voxel outside its mask given the value of the nearest voxel inside."""
    from scipy import ndimage

    with np.load(path, allow_pickle=False) as archive:
        values, mask = archive[component], archive["mask"]
    nearest = ndimage.distance_transform_edt(
        ~mask, return_distances=False, return_indices=True
    )
    values = np.conj(values[(slice(None), *nearest)])
    metadata = _metadata(path)
    step = np.full(3, float(metadata["resolution"]))
    return _Grid(
        np.ascontiguousarray(values.transpose(0, 3, 2, 1)),
        -np.asarray(metadata["origin"], dtype=float) / step,
        step,
    )


@functools.cache
def _metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"]))


def _isocentre(grid: _Grid) -> np.ndarray:
    return _trilinear(grid, np.zeros((1, 3)))[0]


def _interpolated(grid: _Grid, points: np.ndarray) -> np.ndarray:
    """Return ``grid`` interpolated trilinearly at ``(n, 3)`` physical points, as :func:`_trilinear` interpolates it; the edge value beyond it.

    The result is complex128,
    in a temporary file mapped into memory: its pages belong to the file,
    which the operating system writes back rather than holding in the
    process's memory. ``TMPDIR`` names where the file is made.
    """
    points = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 3))
    out = _mapped((len(points), grid.values.shape[0]))
    require("fourier").trilinear(
        grid.values,
        np.asarray(grid.centre, dtype=float),
        np.asarray(grid.step, dtype=float),
        points,
        out,
    )
    return out


def _trilinear(grid: _Grid, points: np.ndarray) -> np.ndarray:
    table = grid.values.reshape(grid.values.shape[0], -1).T
    corners, weights = [], []
    for axis, size in zip((2, 1, 0), grid.values.shape[1:], strict=True):
        index = np.clip(
            points[:, axis] / grid.step[axis] + grid.centre[axis], 0, size - 1
        )
        low = np.minimum(np.floor(index).astype(np.intp), max(size - 2, 0))
        corners.append((low, np.minimum(low + 1, size - 1)))
        weights.append((1.0 - (index - low), index - low))
    total = np.zeros((len(points), grid.values.shape[0]), dtype=np.complex64)
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                flat = np.ravel_multi_index(
                    (corners[0][z], corners[1][y], corners[2][x]), grid.values.shape[1:]
                )
                weight = weights[0][z] * weights[1][y] * weights[2][x]
                total += weight.astype(np.float32)[:, None] * table[flat]
    return total


def _mapped(shape: tuple[int, int]) -> np.ndarray:
    """Return a complex128 array of ``shape`` in an unlinked temporary file, which is freed with the array."""
    if 0 in shape:
        return np.empty(shape, dtype=np.complex128)
    with tempfile.TemporaryFile() as file:
        return np.memmap(file, dtype=np.complex128, mode="w+", shape=shape)
