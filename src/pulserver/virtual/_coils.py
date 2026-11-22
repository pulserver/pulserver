"""RF coils of the virtual scanner, with the sensitivities of coil models or of each coil's field maps."""

from __future__ import annotations

__all__ = ["COILS", "Coil", "coils"]

import functools
import itertools
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

#: Model name of a receive array of circular loops on a helmet about the head,
#: whose sensitivities are the loops' quasi-static Biot-Savart fields.
HELMET = "HELMET"

#: Semi-axes, in m, left-right, anterior-posterior and head-foot, of the
#: ellipsoid the loop centres of :data:`HELMET` lie on, centred on the
#: isocentre; loops sit no lower than ``HELMET_LOWEST`` of the head-foot
#: semi-axis below the centre.
HELMET_SEMI_AXES = (0.095, 0.115, 0.12)
HELMET_LOWEST = 0.45

#: Radius, in m, of each loop of :data:`HELMET`.
HELMET_LOOP_RADIUS = 0.035

#: Samples per axis of the grid :data:`HELMET` is evaluated on: a pitch of
#: 4 mm over :data:`MODEL_FOV`, fine against the centimetre scale on which a
#: loop's field varies inside the head.
_HELMET_SAMPLES = 64

#: Straight segments each loop of :data:`HELMET` is integrated over.
_HELMET_SEGMENTS = 48

#: Distance, in m, below which the field of a segment is evaluated as at this
#: distance; the grid points this reaches lie outside the head.
_HELMET_NEAREST = 0.005

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
    """The transmit and receive RF coils of an exam, fixed in the physical frame.

    ``name`` is ``transmit/receive``, or one coil's name where it does both.
    Each side is a coil model and the number of its channels the coil takes,
    or the file of the coil's field maps :func:`coils` reads; a side without
    either is one channel of unit sensitivity everywhere, as a body coil is
    taken to be. A model is BART's ``HEAD_2D_8CH`` or ``HEAD_3D_64CH``, of
    which the coil takes the first channels, as ``phantom -S`` selects them,
    or :data:`HELMET`, an array of that many loops close about the head. A
    model is sampled on first use over :data:`MODEL_FOV` along the physical
    axes and interpolated linearly; ``HEAD_2D_8CH`` is constant along z.

    Receive sensitivities are scaled to a root sum of squares of 1 at the
    isocentre, and transmit sensitivities so that the :attr:`default_shim`
    plays a pulse at its nominal amplitude at the isocentre. Its weights have a
    root sum of squares of 1, so the channels driven by it carry together the
    power of one channel at that amplitude, as an RF shim of the same
    normalisation does. A model's
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
        """Channel weights of magnitude 1/sqrt(channels) that bring every transmit channel into phase at the isocentre; None for one channel.

        A single-channel pulse played without an RF shim is played on every
        channel with these weights. They are an RF shim's weights, which scale
        the waveform in Pulseq's convention, the conjugate of the field.
        """
        if self.transmit_model is None:
            return None
        phases = np.angle(_isocentre(_transmit(self.transmit_model)))
        return np.exp(1j * phases) / np.sqrt(phases.size)

    def transmit(self, points: np.ndarray) -> np.ndarray | None:
        """Return each channel's transmit sensitivity at ``(n, 3)`` physical points, in m.

        ``(n, channels)``; None for one uniform channel.
        """
        if self.transmit_model is None:
            return None
        return _interpolated(_transmit(self.transmit_model), points)

    def receive(self, points: np.ndarray) -> np.ndarray | None:
        """Return each coil's receive sensitivity at ``(n, 3)`` physical points, in m.

        ``(n, coils)``; None for one uniform coil.
        """
        if self.receive_model is None:
            return None
        return _interpolated(_receive(self.receive_model), points)

    def limits(self) -> dict[str, str]:
        """Return the VOP entries of the check limits a design is made under in this coil.

        Returns
        -------
        dict of str to str
            Empty unless the transmit maps ``<coil>.npz`` have
            ``<coil>_vops.npz`` beside them. Otherwise: ``vop_file``, the VOP
            file; ``vop_drive_per_hz``, the drive of every channel, in the unit
            of the maps and the VOPs, per Hz of a pulse's amplitude, with which
            the channels' rotating fields, half their ``minus``, add up to the
            pulse's nominal amplitude at the isocentre through the
            :attr:`default_shim`; ``vop_default_shim``, that shim, a magnitude
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
        shim = self.default_shim
        return {
            "vop_file": str(vops),
            "vop_drive_per_hz": repr(_drive_per_hz(maps)),
            "vop_default_shim": " ".join(
                f"{abs(w)!r} {float(np.angle(w))!r}" for w in shim.tolist()
            ),
            "vop_head_limit": repr(HEAD_SAR_LIMIT),
            "vop_local_limit": repr(LOCAL_SAR_LIMIT),
        }


COILS = {
    coil.name: coil
    for coil in (
        Coil("body"),
        Coil("body/head48", receive_model=(HELMET, 48)),
        Coil("head8/head32", ("HEAD_2D_8CH", 8), (HELMET, 32)),
    )
}


def coils(
    fields: Path | str | None = None, *, field_t: float | None = None
) -> dict[str, Coil]:
    """Return the virtual scanner's coils by name.

    :data:`COILS`, or the same coils with the field maps in ``fields``.

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
    magnitudes = np.abs(_isocentre(grid))
    return _scaled(grid, np.sum(magnitudes) / np.sqrt(magnitudes.size))


@functools.cache
def _drive_per_hz(maps: Path) -> float:
    import pypulseqpp as pp

    minus = np.abs(_isocentre(_map(maps, "minus")))
    return 2.0 * float(np.sqrt(minus.size)) / (pp.Opts().gamma * float(np.sum(minus)))


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
    if model == HELMET:
        return _helmet(channels)
    from bartorch.tools import phantom

    three = model == "HEAD_3D_64CH"
    maps = phantom((_SAMPLES,) * (3 if three else 2), S=channels, coil=model)
    return np.asarray(maps.cpu().numpy(), dtype=np.complex64).reshape(
        channels, -1, _SAMPLES, _SAMPLES
    )


def _helmet_centres(channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(channels, 3)`` centres and outward unit normals of the loops of :data:`HELMET`.

    The centres are spread over the helmet's cap by farthest-point sampling of
    a dense golden-angle spiral, starting from its vertex.
    """
    semi = np.asarray(HELMET_SEMI_AXES)
    count = 4000
    height = 1.0 - (np.arange(count) + 0.5) * 2.0 / count
    height = height[height >= -HELMET_LOWEST]
    turn = np.pi * (1.0 + np.sqrt(5.0)) * (np.arange(height.size) + 0.5)
    ring = np.sqrt(1.0 - height**2)
    dense = np.stack([ring * np.cos(turn), ring * np.sin(turn), height], 1) * semi
    chosen = [int(np.argmax(dense[:, 2]))]
    distance = np.linalg.norm(dense - dense[chosen[0]], axis=1)
    for _ in range(channels - 1):
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(
            distance, np.linalg.norm(dense - dense[chosen[-1]], axis=1)
        )
    centres = dense[chosen]
    normals = centres / semi**2
    return centres, normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _helmet(channels: int) -> np.ndarray:
    """Return the receive sensitivities of :data:`HELMET` of ``channels`` loops: ``(channels, z, y, x)``.

    Each is ``Bx + j By`` of a unit current around its loop, by the
    Biot-Savart law over straight segments, at the points of a grid of
    :data:`_HELMET_SAMPLES` per axis placed as BART's models are.
    """
    n = _HELMET_SAMPLES
    axis = (np.arange(n) - n // 2) * (MODEL_FOV / n)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
    angles = np.linspace(0.0, 2.0 * np.pi, _HELMET_SEGMENTS + 1)
    maps = np.empty((channels, points.shape[0]), dtype=np.complex64)
    centres, normals = _helmet_centres(channels)
    for channel, (centre, normal) in enumerate(zip(centres, normals, strict=True)):
        across = np.cross(
            normal, [0.0, 0.0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0.0, 0.0]
        )
        across /= np.linalg.norm(across)
        wire = centre + HELMET_LOOP_RADIUS * (
            np.cos(angles)[:, None] * across
            + np.sin(angles)[:, None] * np.cross(normal, across)
        )
        bx = np.zeros(points.shape[0])
        by = np.zeros(points.shape[0])
        for start, end in itertools.pairwise(wire):
            dx, dy, dz = (points - 0.5 * (start + end)).T
            scale = (
                np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), _HELMET_NEAREST) ** -3
            )
            lx, ly, lz = end - start
            bx += (ly * dz - lz * dy) * scale
            by += (lz * dx - lx * dz) * scale
        maps[channel] = bx + 1j * by
    return maps.reshape(channels, n, n, n)


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
    """Return ``grid`` interpolated trilinearly at ``(n, 3)`` physical points.

    The edge value is taken beyond the grid. The result is complex128, held in
    a temporary file mapped into memory, so that the operating system writes
    its pages back rather than the process holding them. ``TMPDIR`` names
    where the file is made.
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
