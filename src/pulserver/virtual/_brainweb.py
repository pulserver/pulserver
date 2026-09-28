"""BrainWeb's normal brain, sampled as isochromats of the virtual scanner."""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
import pypulseqpp as pp

from ._coils import Coil
from ._phantom import Phantom

#: The tissue classes of BrainWeb's normal brain, in the order of the fuzzy
#: model brainweb-dl returns for its subject 0, each with the T1 and T2, in s,
#: and the proton density BrainWeb's MRI simulator gives it at 1.5 T.
TISSUES = {
    "background": (0.0, 0.0, 0.0),
    "CSF": (2.569, 0.329, 1.0),
    "grey matter": (0.833, 0.083, 0.86),
    "white matter": (0.5, 0.07, 0.77),
    "fat": (0.35, 0.07, 1.0),
    "muscle and skin": (0.9, 0.047, 1.0),
    "skin": (2.569, 0.329, 1.0),
    "skull": (0.0, 0.0, 0.0),
    "glial matter": (0.833, 0.083, 0.86),
    "connective tissue": (0.5, 0.07, 0.77),
}

#: Volume magnetic susceptibility, in ppm (SI), of air, and of water, which
#: every tissue class of the head is taken to have (Schenck, Med Phys 23:815,
#: 1996).
AIR_PPM, WATER_PPM = 0.36, -9.05

#: MNI coordinates, in mm, of the first voxel of the model, whose 1 mm voxels
#: run along z, y and x, x fastest.
_FIRST_VOXEL_MM = {"x": -90.0, "y": -126.0, "z": -72.0}


class BrainWeb:
    """BrainWeb's normal brain, received by one coil or several.

    The fuzzy tissue model of BrainWeb's normal brain (Collins et al., IEEE
    Trans Med Imaging 17:463, 1998) gives each 1 mm voxel the fraction of it
    each tissue class fills. brainweb-dl downloads it on first use into its
    cache, which the ``brainweb`` extra installs. The brain lies as a subject
    lying head first and supine, with the origin of its MNI coordinates, the
    anterior commissure, at the isocentre: the physical x axis points to the
    subject's left, y posterior and z superior. Each tissue class relaxes with
    the T1 and T2 and has the proton density BrainWeb's simulator gives it at
    1.5 T (Kwan et al., IEEE Trans Med Imaging 18:1085, 1999), at any field; fat
    precesses at pypulseqpp's fat shift, and every isochromat at the field
    :attr:`field_ppm` its head adds. The coils are those of
    :class:`~pulserver.virtual.Phantom`, fixed in the physical frame.

    Parameters
    ----------
    coils
        Receive channels.
    period
        Spatial period of the sensitivities, in metres.
    depth
        Their modulation depth.
    directory
        brainweb-dl's cache; ``BRAINWEB_DIR``, or ``~/.cache/brainweb``,
        without one.
    susceptibility
        Whether the field the head's susceptibility adds acts on it.
    """

    def __init__(
        self,
        *,
        coils: int = 1,
        period: float = 0.5,
        depth: float = 0.5,
        directory: Path | str | None = None,
        susceptibility: bool = True,
    ) -> None:
        self._coils = Phantom((), coils=coils, period=period, depth=depth)
        self.coils = coils
        self.directory = directory
        self.susceptibility = susceptibility

    @functools.cached_property
    def fractions(self) -> np.ndarray:
        """The fraction of each voxel each tissue fills, ``(z, y, x, tissue)``, downloaded on first use.

        Raises
        ------
        ImportError
            If brainweb-dl is not installed.
        ValueError
            If the model does not hold one fraction per tissue class.
        """
        try:
            import brainweb_dl
        except ImportError as error:
            raise ImportError(
                "BrainWeb is downloaded by brainweb-dl: pip install 'pulserver[brainweb]'"
            ) from error
        fractions = np.asarray(
            brainweb_dl.get_mri(0, "fuzzy", brainweb_dir=self.directory),
            dtype=np.float32,
        )
        if fractions.ndim != 4 or fractions.shape[3] != len(TISSUES):
            raise ValueError(
                f"BrainWeb's fuzzy model holds {len(TISSUES)} tissues per voxel, "
                f"not the shape {fractions.shape}"
            )
        return fractions

    @functools.cached_property
    def field_ppm(self) -> np.ndarray:
        """The field the head adds to B0 at each voxel, ``(z, y, x)``, in ppm of B0, as a first-order shim leaves it.

        The head is water, and the background air, each voxel in proportion to
        the fraction of it the background fills. The field along B0, the
        physical z axis, is the susceptibility convolved with the dipole
        kernel, the Lorentz sphere's third included (Marques and Bowtell,
        Concepts Magn Reson B 25:65, 2005), whose constant and linear terms
        over the voxels at least half head are then removed.
        """
        return _susceptibility_field(self.fractions[..., 0])

    def proton_density(
        self, points: np.ndarray, *, normal: np.ndarray, thickness: float
    ) -> np.ndarray:
        """Return the proton density of a slab ``thickness`` thick along ``normal`` at each of ``(n, 3)`` physical points, in metres.

        The density is each tissue's times the fraction of the nearest voxel it
        fills, averaged in 1 mm steps across the slab; zero outside the model.
        """
        return _slab_density(
            self.fractions, np.asarray(points, dtype=float), normal, thickness
        )

    def isochromats(
        self,
        spacing: float = 1e-3,
        *,
        field_t: float | None = None,
        off_resonance_hz: float = 0.0,
        region: np.ndarray | None = None,
        coil: Coil | None = None,
        threads: int = 0,
    ) -> pp.Isochromats:
        """Return the brain sampled as isochromats, for :func:`~pulserver.virtual.simulate`.

        The voxels are averaged in cubes ``spacing`` wide. Each tissue a cube
        holds is an isochromat at the cube's centre, of proton density the
        tissue's times the fraction of the cube it fills times the cube's
        volume in m³, precessing at the cube's mean :attr:`field_ppm`.

        Parameters
        ----------
        spacing
            Width of a cube, in metres: a whole number of millimetres.
        field_t
            The magnet's field, in T, at which fat's chemical shift is resolved
            into a frequency, with pypulseqpp's default gamma.
        off_resonance_hz
            Frequency of every isochromat from the scanner's centre frequency,
            in Hz, beside its chemical shift.
        region
            ``(3, 2)`` lower and upper bounds along the physical axes, in
            metres, outside which no isochromat is kept; the whole head
            without one.
        coil
            The scanner's coil the brain is scanned with, in place of the
            phantom's coils.
        threads
            Worker threads of the simulation; 0 for every core.

        Raises
        ------
        ValueError
            If ``spacing`` is not a whole number of millimetres, ``field_t``
            is not given, or the brain has coils of its own and ``coil`` is
            given.
        """
        from pypulseqpp.sequences.preparation.fatsat import FAT_SHIFT_PPM

        step = round(spacing / 1e-3)
        if step < 1 or not math.isclose(step * 1e-3, spacing, rel_tol=1e-6):
            raise ValueError(
                f"BrainWeb is sampled in whole millimetres, not {spacing} m"
            )
        if field_t is None:
            raise ValueError(
                "BrainWeb's fat has a chemical shift: scan it at a field_t"
            )
        if coil is not None and self.coils > 1:
            raise ValueError(
                "a phantom received by coils of its own is not scanned with a coil"
            )
        cubes = _cubes(self.fractions, step)
        field = np.zeros(cubes.shape[:3], dtype=np.float32)
        if self.susceptibility:
            field = _cubes(self.field_ppm[..., None], step)[..., 0]
        index = np.indices(cubes.shape[:3]).reshape(3, -1).T
        voxel = step * index + 0.5 * (step - 1)
        z, y, x = (
            voxel[:, axis] + _FIRST_VOXEL_MM[name] for axis, name in enumerate("zyx")
        )
        positions = 1e-3 * np.column_stack([-x, -y, z])
        inside = np.ones(len(positions), dtype=bool)
        if region is not None:
            low, high = np.asarray(region, dtype=float).T
            inside = np.all((positions >= low) & (positions <= high), axis=1)
        per_ppm = 1e-6 * pp.Opts().gamma * field_t
        inhomogeneity = per_ppm * field.reshape(-1)
        points, rows = [], []
        for tissue, (name, (t1, t2, density)) in enumerate(TISSUES.items()):
            fraction = cubes[..., tissue].reshape(-1)
            kept = inside & (fraction > 0.0) & (density > 0.0)
            shift = per_ppm * FAT_SHIFT_PPM if name == "fat" else 0.0
            points.append(positions[kept])
            rows.append(
                np.column_stack(
                    [
                        density * fraction[kept] * (step * 1e-3) ** 3,
                        np.full(kept.sum(), t1),
                        np.full(kept.sum(), t2),
                        shift + off_resonance_hz + inhomogeneity[kept],
                    ]
                )
            )
        own = np.concatenate(points)
        proton_density, t1, t2, frequency = np.concatenate(rows).T
        return pp.Isochromats(
            own,
            proton_density=proton_density,
            t1=t1,
            t2=t2,
            off_resonance=frequency,
            transmit=None if coil is None else coil.transmit(own),
            receive=self._coils._received(own) if coil is None else coil.receive(own),
            threads=threads,
        )


def _slab_density(
    fractions: np.ndarray, points: np.ndarray, normal: np.ndarray, thickness: float
) -> np.ndarray:
    """Return the proton density at ``(n, 3)`` physical points, averaged over 1 mm steps across the slab."""
    densities = np.array([density for _, _, density in TISSUES.values()])
    first = np.array([_FIRST_VOXEL_MM[axis] for axis in "xyz"])
    size = np.array(fractions.shape[2::-1])
    steps = max(1, round(thickness / 1e-3))
    total = np.zeros(len(points))
    for offset in (np.arange(steps) - 0.5 * (steps - 1)) * thickness / steps:
        shifted = points + offset * np.asarray(normal, dtype=float)
        mni = 1e3 * shifted * np.array([-1.0, -1.0, 1.0])
        voxel = np.rint(mni - first).astype(int)
        inside = np.all((voxel >= 0) & (voxel < size), axis=1)
        x, y, z = voxel[inside].T
        total[inside] += fractions[z, y, x] @ densities
    return total / steps


def _susceptibility_field(background: np.ndarray) -> np.ndarray:
    """Return the shimmed field, in ppm, of a head whose voxels ``(z, y, x)`` the background fills the fraction ``background`` of."""
    from scipy import fft

    contrast = ((1.0 - background) * (WATER_PPM - AIR_PPM)).astype(np.float32)
    shape = [fft.next_fast_len(3 * size // 2, real=True) for size in contrast.shape]
    kz = np.fft.fftfreq(shape[0]).astype(np.float32)[:, None, None]
    ky = np.fft.fftfreq(shape[1]).astype(np.float32)[None, :, None]
    kx = np.fft.rfftfreq(shape[2]).astype(np.float32)[None, None, :]
    squared = kz**2 + ky**2 + kx**2
    squared[0, 0, 0] = 1.0
    kernel = np.float32(1.0 / 3.0) - kz**2 / squared
    kernel[0, 0, 0] = 0.0
    spectrum = fft.rfftn(contrast, shape, workers=-1)
    spectrum *= kernel
    field = fft.irfftn(spectrum, shape, workers=-1)
    field = field[tuple(slice(0, size) for size in contrast.shape)]
    grid = np.indices(field.shape, dtype=np.float32)
    head = background <= 0.5
    basis = np.column_stack([np.ones(head.sum()), *(axis[head] for axis in grid)])
    shim, *_ = np.linalg.lstsq(basis, field[head], rcond=None)
    return (field - shim[0] - np.tensordot(shim[1:], grid, axes=1)).astype(np.float32)


def _cubes(fractions: np.ndarray, step: int) -> np.ndarray:
    """Return the fractions averaged in cubes of ``step`` voxels, the voxels past the last whole cube left out."""
    if step == 1:
        return fractions
    whole = [size // step for size in fractions.shape[:3]]
    kept = fractions[: whole[0] * step, : whole[1] * step, : whole[2] * step]
    blocks = kept.reshape(whole[0], step, whole[1], step, whole[2], step, -1)
    return blocks.mean(axis=(1, 3, 5))
