"""BrainWeb's normal brain, sampled as the virtual scanner's tissue."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pypulseqpp as pp

from ._coils import Coil

if TYPE_CHECKING:
    from ._tissue import Tissue
from ._phantom import Phantom

#: The tissue classes of BrainWeb's normal brain, in the order of the fuzzy
#: model brainweb-dl returns for its subject 0, each with the T1, T2 and T2*,
#: in s, and the proton density BrainWeb's MRI simulator gives it at 1.5 T, as
#: its tissue MR parameters list them
#: (https://brainweb.bic.mni.mcgill.ca/brainweb/tissue_mr_parameters.txt).
TISSUES = {
    "background": (0.0, 0.0, 0.0, 0.0),
    "CSF": (2.569, 0.329, 0.058, 1.0),
    "grey matter": (0.833, 0.083, 0.069, 0.86),
    "white matter": (0.5, 0.07, 0.061, 0.77),
    "fat": (0.35, 0.07, 0.058, 1.0),
    "muscle and skin": (0.9, 0.047, 0.030, 1.0),
    "skin": (2.569, 0.329, 0.058, 1.0),
    "skull": (0.0, 0.0, 0.0, 0.0),
    "glial matter": (0.833, 0.083, 0.069, 0.86),
    "connective tissue": (0.5, 0.07, 0.061, 0.77),
}

#: The field, in T, at which BrainWeb's simulator gives :data:`TISSUES`.
TISSUES_FIELD_T = 1.5

#: Exponents ``b`` of the power laws ``T1 ~ B0**b``, by the tissue classes
#: BrainWeb gives their relaxation: those white matter and grey matter follow
#: from 0.2 T to 7 T (Rooney et al., Magn Reson Med 57:308, 2007), and those
#: Bottomley et al. fit to skeletal muscle and adipose tissue from 1 MHz to
#: 100 MHz (Med Phys 11:425, 1984). CSF's T1 does not change with the field
#: from 0.2 T to 7 T, and neither does that of skin, to which BrainWeb gives
#: CSF's relaxation; skull and background hold no protons.
T1_EXPONENTS = {
    "white matter": 0.382,
    "connective tissue": 0.382,
    "grey matter": 0.376,
    "glial matter": 0.376,
    "muscle and skin": 0.4203,
    "fat": 0.1743,
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
    cache. The brain is oriented as a subject
    lying head first and supine, with the origin of its MNI coordinates, the
    anterior commissure, at the isocentre: the physical x axis points to the
    subject's left, y posterior and z superior. Each tissue class relaxes with
    the T1, T2 and T2' :meth:`relaxation` derives at the field it is scanned
    at from the parameters BrainWeb's simulator gives it at 1.5 T (Kwan et
    al., IEEE Trans Med Imaging 18:1085, 1999), and has the proton density
    the simulator gives it. Fat precesses at pypulseqpp's fat shift, or at
    each peak of ``fat_peaks``, and every
    voxel at the field :attr:`field_ppm` its head adds. The coils are those of
    :class:`~pulserver.virtual.Phantom`, fixed in the physical frame.

    Parameters
    ----------
    coils : int, default=1
        Receive channels.
    period : float, default=0.5
        Spatial period of the sensitivities, in metres.
    depth : float, default=0.5
        Their modulation depth.
    directory : Path | str, default=None
        brainweb-dl's cache; ``BRAINWEB_DIR``, or ``~/.cache/brainweb``,
        without one.
    susceptibility : bool, default=True
        Whether the field the head's susceptibility adds acts on it.
    t2_prime : Mapping[str, float], default=None
        T2', in s, by tissue class, in place of the one :meth:`relaxation`
        derives, at any field.
    diffusion : Mapping[str, float], default=None
        Isotropic diffusion coefficient, in m²/s, by tissue class, as
        :attr:`DIFFUSION` gives them; a class without one does not diffuse.
    fat_peaks : Sequence[tuple[float, float]], default=None
        Fat's spectrum, ``(shift_ppm, fraction)`` per peak, the shift
        relative to water: each fat entry is split into one entry per peak
        holding that fraction of its density (the fractions scaled to sum to
        one), as :attr:`FAT_PEAKS` gives them;
        a single peak at pypulseqpp's fat shift without one.
    """

    #: The six-peak spectrum of subcutaneous fat measured at 7 T (Ren et al.,
    #: J Lipid Res 49:2055, 2008), ``(shift_ppm, fraction)`` relative to
    #: water at 4.7 ppm: the methyl, methylene, allylic, diallylic, glycerol
    #: and olefinic resonances. The olefinic peak lies on the water side, out
    #: of reach of a fat-saturation pulse centred on the methylene peak.
    FAT_PEAKS = (
        (-3.80, 0.087),
        (-3.40, 0.693),
        (-2.60, 0.128),
        (-1.94, 0.004),
        (-0.39, 0.039),
        (0.60, 0.048),
    )

    #: Isotropic diffusion coefficients, in m²/s, of the tissue classes whose
    #: water diffusion is measured: the apparent diffusion coefficients of
    #: cortical grey matter and of white matter in adults (Helenius et al.,
    #: AJNR Am J Neuroradiol 23:194, 2002), which glial matter and connective
    #: tissue take as BrainWeb gives them those tissues' relaxation, and that
    #: of free water at 37 degrees C for CSF (Holz et al., Phys Chem Chem Phys
    #: 2:4740, 2000).
    DIFFUSION = MappingProxyType(
        {
            "CSF": 3.0e-9,
            "grey matter": 0.89e-9,
            "white matter": 0.70e-9,
            "glial matter": 0.89e-9,
            "connective tissue": 0.70e-9,
        }
    )

    def __init__(
        self,
        *,
        coils: int = 1,
        period: float = 0.5,
        depth: float = 0.5,
        directory: Path | str | None = None,
        susceptibility: bool = True,
        t2_prime: Mapping[str, float] | None = None,
        diffusion: Mapping[str, float] | None = None,
        fat_peaks: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        for given in (t2_prime, diffusion):
            unknown = set(given or ()) - set(TISSUES)
            if unknown:
                raise ValueError(f"BrainWeb has no tissue class {sorted(unknown)}")
        self._coils = Phantom((), coils=coils, period=period, depth=depth)
        self.coils = coils
        self.directory = directory
        self.susceptibility = susceptibility
        self.t2_prime = dict(t2_prime or {})
        self.diffusion = dict(diffusion or {})
        self.fat_peaks = None
        if fat_peaks is not None:
            total = sum(fraction for _, fraction in fat_peaks)
            if not total > 0.0:
                raise ValueError("fat needs at least one peak of positive fraction")
            self.fat_peaks = tuple(
                (ppm, fraction / total) for ppm, fraction in fat_peaks
            )

    @functools.cached_property
    def fractions(self) -> np.ndarray:
        """The fraction of each voxel each tissue fills, ``(z, y, x, tissue)``, downloaded on first use.

        Raises
        ------
        ValueError
            If the model does not hold one fraction per tissue class.
        """
        import brainweb_dl

        # C order, as every sampling of the model reshapes it: brainweb-dl
        # returns the NIfTI file's Fortran order.
        fractions = np.ascontiguousarray(
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

        The head is water, and the background air where the head ends: at a
        voxel mostly background and at its neighbours, each in proportion to
        the fraction of it the background fills. The fuzzy model's small
        background fractions inside the scalp and the skull are no air. The
        field along B0, the physical z axis, is the susceptibility convolved
        with the dipole kernel, the Lorentz sphere's third included (Marques
        and Bowtell, Concepts Magn Reson B 25:65, 2005). The field steps
        across the surface of the head, so a voxel the surface crosses, and
        the voxel beneath it, take the field of the tissue deeper in, which
        is the field their tissue is in. The constant and linear terms over
        the voxels at least half head are then removed.
        """
        return _susceptibility_field(self.fractions[..., 0])

    def relaxation(self, field_t: float) -> dict[str, tuple[float, float, float]]:
        """Return the T1, T2 and T2', in s, of each tissue class at ``field_t`` T.

        T1 is BrainWeb's, at 1.5 T, times ``(field_t / 1.5)**b`` with the
        class's exponent in :data:`T1_EXPONENTS`, or BrainWeb's without one; T2
        is BrainWeb's. T2' is the one BrainWeb's T2 and T2* leave at 1.5 T,
        ``1 / (1/T2* - 1/T2)``, times ``1.5 / field_t``, as the static
        dephasing regime makes R2' proportional to the field (Yablonskiy and
        Haacke, Magn Reson Med 32:749, 1994), or the one :attr:`t2_prime`
        gives the class; infinite for a class without T2 or T2*.

        Raises
        ------
        ValueError
            If ``field_t`` is not above zero.
        """
        if not field_t > 0.0:
            raise ValueError(f"tissues relax at a field above zero, not {field_t} T")
        ratio = field_t / TISSUES_FIELD_T
        relaxed = {}
        for name, (t1, t2, t2_star, _) in TISSUES.items():
            t2_prime = self.t2_prime.get(name, _t2_prime(t2, t2_star) / ratio)
            relaxed[name] = (t1 * ratio ** T1_EXPONENTS.get(name, 0.0), t2, t2_prime)
        return relaxed

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

    def tissue(
        self,
        spacing: float = 1e-3,
        *,
        field_t: float | None = None,
        off_resonance_hz: float = 0.0,
        region: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        coil: Coil | None = None,
    ) -> Tissue:
        """Return the brain sampled for the Fourier engine: each tissue of each cube ``spacing`` wide, at the cube's centre.

        Raises
        ------
        ValueError
            If ``spacing`` is not a whole number of millimetres, ``field_t``
            is not given, or the brain has coils of its own and ``coil`` is
            given.
        """
        from ._tissue import Tissue, transmitted

        _whole_millimetres(spacing)
        if coil is not None and self.coils > 1:
            raise ValueError(
                "a phantom received by coils of its own is not scanned with a coil"
            )
        positions, density, t1, t2, frequency, t2_prime, diffusion = self._sampled(
            spacing, field_t, off_resonance_hz, region
        )
        return Tissue(
            positions=positions,
            density=density,
            t1=t1,
            t2=t2,
            t2_prime=t2_prime,
            frequency=frequency,
            spacing=spacing,
            axes=np.eye(3),
            transmit=None if coil is None else transmitted(coil, positions),
            receive=self._coils._received if coil is None else coil.receive,
            coils=self.coils if coil is None else coil.receive_channels,
            diffusion=diffusion if np.any(diffusion) else None,
        )

    def _sampled(
        self,
        spacing: float,
        field_t: float | None,
        off_resonance_hz: float,
        region: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    ) -> tuple[np.ndarray, ...]:
        """Return the centres, proton densities, T1, T2, frequencies, T2' and diffusion coefficients of the tissues of the cubes ``spacing`` wide."""
        from pypulseqpp.sequences.preparation.fatsat import FAT_SHIFT_PPM

        if field_t is None:
            raise ValueError(
                "BrainWeb's fat has a chemical shift: scan it at a field_t"
            )
        step = round(spacing / 1e-3)
        cubes = _cubes(self.fractions, step)
        field = np.zeros(cubes.shape[:3], dtype=np.float32)
        if self.susceptibility:
            field = _cubes(self.field_ppm[..., None], step)[..., 0]
        per_ppm = 1e-6 * pp.Opts().gamma * field_t
        fat = self.fat_peaks or ((FAT_SHIFT_PPM, 1.0),)
        spectra = [
            [(per_ppm * ppm, weight) for ppm, weight in fat]
            if name == "fat"
            else [(0.0, 1.0)]
            for name in TISSUES
        ]
        fractions = cubes.reshape(-1, len(TISSUES))
        inhomogeneity = per_ppm * field.reshape(-1)
        index = np.indices(cubes.shape[:3]).reshape(3, -1).T
        voxel = step * index + 0.5 * (step - 1)
        z, y, x = (
            voxel[:, axis] + _FIRST_VOXEL_MM[name] for axis, name in enumerate("zyx")
        )
        positions = 1e-3 * np.column_stack([-x, -y, z])
        relaxed = self.relaxation(field_t)
        points, rows = [], []
        for tissue, (name, (*_, density)) in enumerate(TISSUES.items()):
            t1, t2, t2_prime = relaxed[name]
            fraction = fractions[:, tissue]
            kept = np.flatnonzero((fraction > 0.0) & (density > 0.0))
            for shift, weight in spectra[tissue]:
                frequency = shift + off_resonance_hz + inhomogeneity[kept]
                held = kept
                if region is not None:
                    inside = region(positions[held], frequency)
                    held, frequency = held[inside], frequency[inside]
                points.append(positions[held])
                rows.append(
                    np.column_stack(
                        [
                            weight * density * fraction[held] * (step * 1e-3) ** 3,
                            np.full(held.size, t1),
                            np.full(held.size, t2),
                            frequency,
                            np.full(held.size, t2_prime),
                            np.full(held.size, self.diffusion.get(name, 0.0)),
                        ]
                    )
                )
        return (np.concatenate(points), *np.concatenate(rows).T)


def _t2_prime(t2: float, t2_star: float) -> float:
    """Return the T2' that T2 and T2*, in s, leave, R2' = R2* - R2; infinite for a tissue without either."""
    if not 0.0 < t2_star < t2:
        return math.inf
    return 1.0 / (1.0 / t2_star - 1.0 / t2)


def _whole_millimetres(spacing: float) -> None:
    step = round(spacing / 1e-3)
    if step < 1 or not math.isclose(step * 1e-3, spacing, rel_tol=1e-6):
        raise ValueError(f"BrainWeb is sampled in whole millimetres, not {spacing} m")


def _slab_density(
    fractions: np.ndarray, points: np.ndarray, normal: np.ndarray, thickness: float
) -> np.ndarray:
    """Return the proton density at ``(n, 3)`` physical points, averaged over 1 mm steps across the slab."""
    densities = np.array([density for *_, density in TISSUES.values()])
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

    air = _air(background)
    contrast = ((1.0 - air) * (WATER_PPM - AIR_PPM)).astype(np.float32)
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
    field = _tissue_side(field, air)
    grid = np.indices(field.shape, dtype=np.float32)
    head = background <= 0.5
    basis = np.column_stack([np.ones(head.sum()), *(axis[head] for axis in grid)])
    shim, *_ = np.linalg.lstsq(basis, field[head], rcond=None)
    return (field - shim[0] - np.tensordot(shim[1:], grid, axes=1)).astype(np.float32)


#: Background fraction above which a voxel is air, and the width, in voxels,
#: of the Gaussian weights with which a voxel at the surface of the head
#: takes the field of the tissue beneath it.
_AIR_FRACTION, _SURFACE_REACH = 0.5, 1.5


def _air(background: np.ndarray) -> np.ndarray:
    """Return the fraction of each voxel that is air: the background's at a voxel mostly background and at the 26 around it, none elsewhere."""
    from scipy import ndimage

    beside = ndimage.binary_dilation(
        background > _AIR_FRACTION, structure=np.ones((3, 3, 3), dtype=bool)
    )
    return np.where(beside, background, 0.0).astype(np.float32)


def _tissue_side(field: np.ndarray, air: np.ndarray) -> np.ndarray:
    """Return ``field`` with every voxel that holds both tissue and air, and every voxel of tissue beside air, at the field of the tissue beneath it.

    The field of a voxel of tissue not beside air is kept; the others take
    the mean of those within reach, weighted by a Gaussian ``_SURFACE_REACH``
    voxels wide, or the nearest one where none is within reach. A voxel of
    air alone keeps its field.
    """
    from scipy import ndimage

    cube = np.ones((3, 3, 3), dtype=bool)
    deep = ~ndimage.binary_dilation(air > 0.0, structure=cube)
    if not deep.any():
        return field
    weight = ndimage.gaussian_filter(deep.astype(np.float32), _SURFACE_REACH)
    spread = ndimage.gaussian_filter(np.where(deep, field, 0.0), _SURFACE_REACH)
    beneath = spread / np.maximum(weight, 1e-6)
    out_of_reach = weight < 1e-3
    if out_of_reach.any():
        nearest = ndimage.distance_transform_edt(
            ~deep, return_distances=False, return_indices=True
        )
        beneath = np.where(out_of_reach, field[tuple(nearest)], beneath)
    surface = ~deep & (air < 1.0)
    return np.where(surface, beneath, field).astype(np.float32)


def _cubes(fractions: np.ndarray, step: int) -> np.ndarray:
    """Return the fractions averaged in cubes of ``step`` voxels, the voxels past the last whole cube left out."""
    if step == 1:
        return fractions
    whole = [size // step for size in fractions.shape[:3]]
    kept = fractions[: whole[0] * step, : whole[1] * step, : whole[2] * step]
    blocks = kept.reshape(whole[0], step, whole[1], step, whole[2], step, -1)
    return blocks.mean(axis=(1, 3, 5))
