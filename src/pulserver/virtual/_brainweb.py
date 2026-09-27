"""BrainWeb's normal brain, sampled as isochromats of the virtual scanner."""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
import pypulseqpp as pp

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
    precesses at pypulseqpp's fat shift. The coils are those of
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
    """

    def __init__(
        self,
        *,
        coils: int = 1,
        period: float = 0.5,
        depth: float = 0.5,
        directory: Path | str | None = None,
    ) -> None:
        self._coils = Phantom((), coils=coils, period=period, depth=depth)
        self.coils = coils
        self.directory = directory

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

    def isochromats(
        self,
        spacing: float = 1e-3,
        *,
        field_t: float | None = None,
        off_resonance_hz: float = 0.0,
        region: np.ndarray | None = None,
        threads: int = 0,
    ) -> pp.Isochromats:
        """Return the brain sampled as isochromats, for :func:`~pulserver.virtual.simulate`.

        The voxels are averaged in cubes ``spacing`` wide. Each tissue a cube
        holds is an isochromat at the cube's centre, of proton density the
        tissue's times the fraction of the cube it fills times the cube's
        volume in m³.

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
        threads
            Worker threads of the simulation; 0 for every core.

        Raises
        ------
        ValueError
            If ``spacing`` is not a whole number of millimetres, or
            ``field_t`` is not given.
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
        cubes = _cubes(self.fractions, step)
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
                        np.full(kept.sum(), shift + off_resonance_hz),
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
            receive=self._coils._received(own),
            threads=threads,
        )


def _cubes(fractions: np.ndarray, step: int) -> np.ndarray:
    """Return the fractions averaged in cubes of ``step`` voxels, the voxels past the last whole cube left out."""
    if step == 1:
        return fractions
    whole = [size // step for size in fractions.shape[:3]]
    kept = fractions[: whole[0] * step, : whole[1] * step, : whole[2] * step]
    blocks = kept.reshape(whole[0], step, whole[1], step, whole[2], step, -1)
    return blocks.mean(axis=(1, 3, 5))
