"""Vendor-neutral gradient nonlinearity correction.

The spherical-harmonic conversion and coordinate conventions in this module
are adapted from the MIT-licensed HCP ``gradunwarp`` package and its older
``.dat`` converter. Cubic interpolation uses native SimpleITK B-spline
resampling for both 2D and 3D images.

No vendor coefficient table is bundled or persisted. Scanner-specific
coefficient transport belongs to the private MRD integration layer.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np

__all__ = [
    "CoefficientAccessor",
    "GradientCoefficients",
    "Gradunwarp",
    "ImageGeometry",
]


HarmonicBasis = Literal["unnormalized", "normalized"]
CoefficientFormat = Literal["auto", "dat", "grad", "coef"]
CoefficientPayload: TypeAlias = str | bytes | bytearray

_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_DAT_SCALE_RE = re.compile(
    rf"^\s*SCALE(?P<axis>[XYZ])(?P<order>\d+)\s+(?P<value>{_FLOAT_PATTERN})",
    re.IGNORECASE | re.MULTILINE,
)
_SIEMENS_GRAD_RE = re.compile(
    rf"^\s*\d+\s+(?P<kind>[AB])\s*\(\s*(?P<n>\d+)\s*,\s*(?P<m>\d+)"
    rf"\s*\)\s+(?P<value>{_FLOAT_PATTERN})\s+(?P<axis>[xyz])\b",
    re.IGNORECASE,
)
_SIEMENS_R0_RE = re.compile(
    rf"(?P<radius>{_FLOAT_PATTERN})\s*m\s*=\s*R0\b",
    re.IGNORECASE,
)
_NEUTRAL_COEF_RE = re.compile(
    rf"^\s*(?P<kind>Alpha|Beta)_(?P<axis>[xyz])\s+"
    rf"(?P<n>\d+)\s+(?P<m>\d+)\s+(?P<value>{_FLOAT_PATTERN})",
    re.IGNORECASE,
)

# The original converter divides stored values by the multiplicative factors
# documented in the source table. The 6th--9th order factors extend the beta
# converter's original 1st--5th order table. SCALE10 has no documented
# conversion and is therefore accepted only when it is zero.
_DAT_XY_FACTORS = np.asarray(
    [
        0.0,
        1.0,
        1.0 / 3.0,
        2.0 / 3.0,
        2.0 / 5.0,
        8.0 / 15.0,
        8.0 / 21.0,
        16.0 / 7.0,
        16.0 / 9.0,
        128.0 / 45.0,
        0.0,
    ],
    dtype=np.float64,
)
_DAT_Z_FACTORS = np.asarray(
    [0.0, 1.0, -2.0, -2.0, -8.0, -8.0, -16.0, -16.0, -128.0, -128.0, 0.0],
    dtype=np.float64,
)


@runtime_checkable
class CoefficientAccessor(Protocol):
    """Interface used by scanner/LiveSDK integrations for private tables.

    Examples
    --------
    Reads a vendor gradient coefficient file into the harmonic basis the unwarping
    evaluates::

        import pulserver.recon as recon

        coefficients = recon.CoefficientAccessor.from_file("coeff.grad")
        unwarp = recon.Gradunwarp(coefficients)
    """

    def read_coefficients(self) -> CoefficientPayload:
        """Return coefficient-file contents without writing them to disk."""


CoefficientSource: TypeAlias = (
    CoefficientPayload
    | str
    | Path
    | io.TextIOBase
    | io.BufferedIOBase
    | CoefficientAccessor
)


def _source_text(source: CoefficientSource) -> tuple[str, str | None]:
    """Resolve a coefficient source to text and a non-sensitive source label."""
    if isinstance(source, CoefficientAccessor):
        payload = source.read_coefficients()
        label = getattr(source, "name", type(source).__name__)
    elif hasattr(source, "read"):
        payload = source.read()
        label = getattr(source, "name", type(source).__name__)
    elif isinstance(source, Path):
        payload = source.read_bytes()
        label = source.name
    elif isinstance(source, str):
        if "\n" not in source and "\r" not in source:
            candidate = Path(source)
            if candidate.is_file():
                payload = candidate.read_bytes()
                label = candidate.name
            else:
                payload = source
                label = None
        else:
            payload = source
            label = None
    else:
        payload = source
        label = None

    if isinstance(payload, (bytes, bytearray, np.bytes_)):
        text = bytes(payload).decode("utf-8")
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError(
            "Coefficient accessors must return str or bytes, not "
            f"{type(payload).__name__}."
        )
    return text, label


@dataclass(frozen=True)
class GradientCoefficients:
    """Neutral spherical-harmonic coefficient representation.

    Arrays have shape ``(3, order + 1, order + 1)`` for scanner X/Y/Z and
    contain the real cosine (``alpha``) and sine (``beta``) terms.  Coefficient
    arrays are deliberately excluded from ``repr`` so private scanner tables
    are not accidentally logged.

    Examples
    --------
    The spherical-harmonic description of a gradient coil, as the vendor ships it.
    :class:`~pulserver.recon.Gradunwarp` evaluates these over an image's geometry
    to find where each voxel really was::

        import pulserver.recon as recon

        coefficients = recon.CoefficientAccessor.from_file("coeff.grad")
    """

    basis: HarmonicBasis
    alpha: np.ndarray = field(repr=False)
    beta: np.ndarray = field(repr=False)
    reference_radius_mm: float | None = None
    source_name: str | None = None

    def __post_init__(self) -> None:
        alpha = np.array(self.alpha, dtype=np.float64, copy=True)
        beta = np.array(self.beta, dtype=np.float64, copy=True)
        if alpha.ndim != 3 or alpha.shape[0] != 3 or alpha.shape != beta.shape:
            raise ValueError(
                "alpha and beta must have matching shape (3, order + 1, order + 1)."
            )
        if alpha.shape[1] != alpha.shape[2]:
            raise ValueError("Harmonic coefficient matrices must be square.")
        if self.basis not in ("unnormalized", "normalized"):
            raise ValueError("basis must be 'unnormalized' or 'normalized'.")
        if self.basis == "normalized" and (
            self.reference_radius_mm is None or self.reference_radius_mm <= 0
        ):
            raise ValueError(
                "Normalized coefficients require a positive reference_radius_mm."
            )
        alpha.setflags(write=False)
        beta.setflags(write=False)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @property
    def max_order(self) -> int:
        """Largest represented spherical-harmonic order."""
        return self.alpha.shape[1] - 1

    @property
    def mapping_sign(self) -> float:
        """Sign used by HCP gradunwarp to map output to source coordinates."""
        return -1.0 if self.basis == "unnormalized" else 1.0

    @classmethod
    def from_file(
        cls,
        source: CoefficientSource,
        *,
        coefficient_format: CoefficientFormat = "auto",
        reference_radius_mm: float | None = None,
    ) -> GradientCoefficients:
        """Parse a coefficient path, serialized payload, or MRD accessor.

        ``coefficient_format="auto"`` detects the supported ``.dat``,
        ``.grad``, and neutral ``.coef`` syntaxes from their contents.
        """
        text, label = _source_text(source)
        if coefficient_format == "auto":
            coefficient_format = "dat" if _DAT_SCALE_RE.search(text) else "grad"
        if coefficient_format == "dat":
            return cls._from_dat_text(text, label)
        if coefficient_format in ("grad", "coef"):
            return cls._from_normalized_text(text, label, reference_radius_mm)
        raise ValueError("coefficient_format must be 'auto', 'dat', 'grad', or 'coef'.")

    @classmethod
    def _from_dat_text(
        cls,
        text: str,
        label: str | None,
    ) -> GradientCoefficients:
        gradwarp_type: int | None = None
        delta = 0.0
        scales: dict[tuple[int, int], float] = {}

        for line in text.splitlines():
            fields = line.split("#", 1)[0].strip().split()
            if not fields:
                continue
            key = fields[0].upper()
            if key == "GRADWARPTYPE" and len(fields) >= 2:
                gradwarp_type = int(fields[1])
                continue
            if key == "DELTA" and len(fields) >= 2:
                delta = float(fields[1].replace("D", "E").replace("d", "e"))
                continue
            match = _DAT_SCALE_RE.match(line)
            if match:
                axis = "XYZ".index(match.group("axis").upper())
                order = int(match.group("order"))
                value = float(match.group("value").replace("D", "E").replace("d", "e"))
                scales[(axis, order)] = value

        if gradwarp_type not in (None, 1):
            raise ValueError(f"Only GRADWARPTYPE 1 is supported, got {gradwarp_type}.")
        if not scales:
            raise ValueError("No SCALE{X,Y,Z}{N} entries found in .dat payload.")
        if not math.isclose(delta, 0.0, abs_tol=1e-5):
            raise ValueError(f".dat DELTA must be zero, got {delta}.")

        max_order = max(order for _, order in scales)
        if max_order > 10:
            raise ValueError(f".dat harmonic order {max_order} is not supported.")
        for axis in range(3):
            value = scales.get((axis, 10), 0.0)
            if not math.isclose(value, 0.0, abs_tol=0.0):
                raise ValueError(
                    "Non-zero SCALE{X,Y,Z}10 cannot be converted because its "
                    "neutral normalization is undocumented."
                )

        represented_order = min(max(max_order, 1), 9)
        alpha = np.zeros((3, represented_order + 1, represented_order + 1))
        beta = np.zeros_like(alpha)
        for order in range(1, represented_order + 1):
            alpha[0, order, 1] = scales.get((0, order), 0.0) * _DAT_XY_FACTORS[order]
            beta[1, order, 1] = scales.get((1, order), 0.0) * _DAT_XY_FACTORS[order]
            alpha[2, order, 0] = scales.get((2, order), 0.0) * _DAT_Z_FACTORS[order]
        return cls("unnormalized", alpha, beta, source_name=label)

    @classmethod
    def _from_normalized_text(
        cls,
        text: str,
        label: str | None,
        reference_radius_mm: float | None = None,
    ) -> GradientCoefficients:
        entries: list[tuple[str, int, int, int, float]] = []
        parsed_radius = reference_radius_mm

        for line in text.splitlines():
            radius_match = _SIEMENS_R0_RE.search(line)
            if radius_match:
                parsed_radius = 1000.0 * float(
                    radius_match.group("radius").replace("D", "E").replace("d", "e")
                )
            grad_match = _SIEMENS_GRAD_RE.match(line)
            if grad_match:
                entries.append(
                    (
                        grad_match.group("kind").upper(),
                        "xyz".index(grad_match.group("axis").lower()),
                        int(grad_match.group("n")),
                        int(grad_match.group("m")),
                        float(
                            grad_match.group("value")
                            .replace("D", "E")
                            .replace("d", "e")
                        ),
                    )
                )
                continue
            neutral_match = _NEUTRAL_COEF_RE.match(line)
            if neutral_match:
                entries.append(
                    (
                        "A" if neutral_match.group("kind").lower() == "alpha" else "B",
                        "xyz".index(neutral_match.group("axis").lower()),
                        int(neutral_match.group("n")),
                        int(neutral_match.group("m")),
                        float(
                            neutral_match.group("value")
                            .replace("D", "E")
                            .replace("d", "e")
                        ),
                    )
                )

        if not entries:
            raise ValueError("No spherical-harmonic entries found.")
        if parsed_radius is None or parsed_radius <= 0:
            raise ValueError(
                "Neutral .coef data requires reference_radius_mm; .grad data "
                "normally contains '<R0> m = R0'."
            )

        max_order = max(n for _, _, n, _, _ in entries)
        alpha = np.zeros((3, max_order + 1, max_order + 1))
        beta = np.zeros_like(alpha)
        for kind, axis, order, degree, value in entries:
            if degree > order:
                raise ValueError(f"Invalid harmonic degree m={degree} for n={order}.")
            (alpha if kind == "A" else beta)[axis, order, degree] = value
        return cls(
            "normalized",
            alpha,
            beta,
            reference_radius_mm=float(parsed_radius),
            source_name=label,
        )


@dataclass(frozen=True)
class ImageGeometry:
    """Physical geometry for an image's trailing two or three array axes.

    ``direction[:, axis]`` is the unit scanner-coordinate vector followed when
    the corresponding NumPy image index increases.  ``fov_mm`` is independent
    of matrix size, so a zero-filled reconstruction must use its *actual*
    output ``shape`` with the unchanged physical FOV.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.recon as recon
    >>> geometry = recon.ImageGeometry(
    ...     shape=(4, 4, 4),
    ...     fov_mm=(240.0, 240.0, 240.0),
    ...     direction=np.eye(3),
    ...     center_mm=(0.0, 0.0, 0.0),
    ... )
    >>> geometry.shape, geometry.fov_mm[0]
    ((4, 4, 4), 240.0)
    """

    shape: tuple[int, ...]
    fov_mm: tuple[float, ...]
    direction: np.ndarray = field(repr=False)
    center_mm: np.ndarray

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        fov = tuple(float(value) for value in self.fov_mm)
        direction = np.array(self.direction, dtype=np.float64, copy=True)
        center = np.array(self.center_mm, dtype=np.float64, copy=True)
        if len(shape) not in (2, 3):
            raise ValueError("ImageGeometry supports two or three spatial axes.")
        if len(fov) != len(shape) or any(value <= 0 for value in fov):
            raise ValueError("fov_mm must contain one positive value per axis.")
        if any(value <= 0 for value in shape):
            raise ValueError("shape entries must be positive.")
        if direction.shape != (3, len(shape)):
            raise ValueError(
                f"direction must have shape (3, {len(shape)}), got {direction.shape}."
            )
        if center.shape != (3,):
            raise ValueError("center_mm must have shape (3,).")
        gram = direction.T @ direction
        if not np.allclose(gram, np.eye(len(shape)), atol=1e-6):
            raise ValueError("direction columns must be orthonormal.")
        direction.setflags(write=False)
        center.setflags(write=False)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "fov_mm", fov)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "center_mm", center)

    @property
    def ndim(self) -> int:
        """Number of spatial image axes."""
        return len(self.shape)

    @property
    def voxel_size_mm(self) -> np.ndarray:
        """Physical voxel spacing based on the actual matrix size."""
        return np.asarray(self.fov_mm) / np.asarray(self.shape)

    def indices_to_scanner(self, indices: np.ndarray) -> np.ndarray:
        """Map array indices to scanner coordinates in millimetres."""
        indices = np.asarray(indices, dtype=np.float64)
        if indices.shape[-1] != self.ndim:
            raise ValueError(
                f"Expected indices[..., {self.ndim}], got {indices.shape}."
            )
        centered = indices - (np.asarray(self.shape) - 1.0) / 2.0
        offsets = centered * self.voxel_size_mm
        return self.center_mm + offsets @ self.direction.T

    def scanner_to_indices(self, coordinates_mm: np.ndarray) -> np.ndarray:
        """Map scanner coordinates in millimetres to floating image indices."""
        coordinates = np.asarray(coordinates_mm, dtype=np.float64)
        if coordinates.shape[-1] != 3:
            raise ValueError("Scanner coordinates must have a final size-3 axis.")
        offsets = (coordinates - self.center_mm) @ self.direction
        return offsets / self.voxel_size_mm + (np.asarray(self.shape) - 1.0) / 2.0

    def scanner_grid(self) -> np.ndarray:
        """Return the physical coordinate of every voxel centre."""
        indices = np.moveaxis(np.indices(self.shape, dtype=np.float64), 0, -1)
        return self.indices_to_scanner(indices)

    @classmethod
    def from_affine(
        cls,
        affine: np.ndarray,
        shape: tuple[int, ...],
    ) -> ImageGeometry:
        """Create geometry from a voxel-centre-to-scanner affine."""
        shape = tuple(int(value) for value in shape)
        if len(shape) not in (2, 3):
            raise ValueError("shape must have two or three entries.")
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape != (4, 4):
            raise ValueError("affine must have shape (4, 4).")
        basis = affine[:3, : len(shape)]
        spacing = np.linalg.norm(basis, axis=0)
        if np.any(spacing <= 0):
            raise ValueError("affine contains a zero-length spatial axis.")
        direction = basis / spacing
        center_index = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
        center_h = np.concatenate((center_index, np.zeros(3 - len(shape)), [1.0]))
        center = (affine @ center_h)[:3]
        return cls(
            shape,
            tuple((spacing * np.asarray(shape)).tolist()),
            direction,
            center,
        )

    @classmethod
    def from_mrd(
        cls,
        image_header: Any,
        shape: tuple[int, ...],
        *,
        field_of_view_mm: tuple[float, ...] | None = None,
    ) -> ImageGeometry:
        """Create geometry from a vendor-neutral MRD image header.

        MRD image arrays conventionally store spatial axes as ``(y, x)`` or
        ``(z, y, x)``. The corresponding direction columns are therefore
        ``(line, column)`` or ``(slice, line, column)``. ``shape`` is required
        explicitly so zero-filled output matrices never fall back to the
        acquired matrix size.

        ``field_of_view_mm`` may override the image-header FOV, for example
        when geometry is assembled before an output image header exists.
        """
        shape = tuple(int(value) for value in shape)
        if len(shape) not in (2, 3):
            raise ValueError("shape must have two or three entries.")

        def vector(*names: str) -> np.ndarray:
            for name in names:
                value = getattr(image_header, name, None)
                if value is not None:
                    array = np.asarray(value, dtype=np.float64)
                    if array.shape == (3,):
                        return array
            raise ValueError(f"MRD image header is missing {names[0]!r}.")

        column = vector("read_dir", "col_dir")
        line = vector("phase_dir", "line_dir")
        directions = [line, column]
        if len(shape) == 3:
            directions.insert(0, vector("slice_dir"))

        if field_of_view_mm is None:
            raw_fov = getattr(image_header, "field_of_view", None)
            if raw_fov is None:
                raise ValueError(
                    "MRD image header has no field_of_view; pass field_of_view_mm."
                )
            xyz_fov = tuple(float(value) for value in raw_fov)
            field_of_view_mm = (
                (xyz_fov[1], xyz_fov[0])
                if len(shape) == 2
                else (xyz_fov[2], xyz_fov[1], xyz_fov[0])
            )
        center = vector("position")
        return cls(shape, field_of_view_mm, np.stack(directions, axis=1), center)


def _associated_legendre(
    order: int,
    degree: int,
    cosine: np.ndarray,
) -> np.ndarray:
    """Unnormalised associated Legendre polynomial with Condon-Shortley phase."""
    if degree < 0 or degree > order:
        raise ValueError(f"Invalid associated Legendre P_{order}^{degree}.")
    p_mm = np.ones_like(cosine)
    if degree:
        root = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
        factor = 1.0
        for _ in range(1, degree + 1):
            p_mm *= -factor * root
            factor += 2.0
    if order == degree:
        return p_mm
    p_m1m = (2 * degree + 1) * cosine * p_mm
    if order == degree + 1:
        return p_m1m
    previous, current = p_mm, p_m1m
    for n in range(degree + 2, order + 1):
        following = ((2 * n - 1) * cosine * current - (n + degree - 1) * previous) / (
            n - degree
        )
        previous, current = current, following
    return current


def _evaluate_harmonics(
    coefficients: GradientCoefficients,
    coordinates_mm: np.ndarray,
) -> np.ndarray:
    """Evaluate displacement/error fields directly on scanner coordinates."""
    coordinates = np.asarray(coordinates_mm, dtype=np.float64)
    if coefficients.basis == "unnormalized" and _has_dat_structure(coefficients):
        return _evaluate_dat_harmonics(coefficients, coordinates)

    x, y, z = np.moveaxis(coordinates, -1, 0)
    radius_mm = np.sqrt(x * x + y * y + z * z)
    cosine = np.divide(
        z,
        radius_mm,
        out=np.ones_like(radius_mm),
        where=radius_mm != 0,
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    azimuth = np.arctan2(y, x)
    result = np.zeros(coordinates.shape, dtype=np.float64)

    if coefficients.basis == "unnormalized":
        radial = radius_mm / 10.0  # Unnormalised .dat coefficients use centimetres.
        output_scale = 10.0  # centimetres back to millimetres.
    else:
        assert coefficients.reference_radius_mm is not None
        radial = radius_mm / coefficients.reference_radius_mm
        output_scale = coefficients.reference_radius_mm

    for n in range(coefficients.max_order + 1):
        radial_term = np.power(radial, n)
        for m in range(n + 1):
            alpha = coefficients.alpha[:, n, m]
            beta = coefficients.beta[:, n, m]
            if not np.any(alpha) and not np.any(beta):
                continue
            legendre = _associated_legendre(n, m, cosine)
            if coefficients.basis == "normalized" and m > 0:
                normalization = ((-1.0) ** m) * math.sqrt(
                    (2 * n + 1) * math.factorial(n - m) / (2.0 * math.factorial(n + m))
                )
                legendre = legendre * normalization
            angular = (
                alpha.reshape((1,) * radius_mm.ndim + (3,))
                * np.cos(m * azimuth)[..., None]
                + beta.reshape((1,) * radius_mm.ndim + (3,))
                * np.sin(m * azimuth)[..., None]
            )
            result += (
                output_scale * radial_term[..., None] * legendre[..., None] * angular
            )
    return result


def _has_dat_structure(coefficients: GradientCoefficients) -> bool:
    """Return whether only the three harmonic families stored by .dat are set."""
    alpha = coefficients.alpha.copy()
    beta = coefficients.beta.copy()
    for order in range(1, coefficients.max_order + 1):
        alpha[0, order, 1] = 0.0
        beta[1, order, 1] = 0.0
        alpha[2, order, 0] = 0.0
    return not (np.any(alpha) or np.any(beta))


def _evaluate_dat_harmonics(
    coefficients: GradientCoefficients,
    coordinates_mm: np.ndarray,
) -> np.ndarray:
    """Evaluate .dat solid harmonics using Cartesian recurrences.

    This avoids spherical coordinates, trigonometric functions, and repeated
    associated-Legendre evaluation. It is algebraically equivalent for the
    three coefficient families represented by the source format.
    """
    x, y, z = np.moveaxis(coordinates_mm / 10.0, -1, 0)
    radius_squared = x * x + y * y + z * z
    result = np.zeros_like(coordinates_mm)

    zonal_previous_2 = np.ones_like(x)
    zonal_previous_1 = z
    order_one_previous_2 = np.zeros_like(x)
    order_one_previous_1 = np.ones_like(x)

    for order in range(1, coefficients.max_order + 1):
        if order == 1:
            zonal = zonal_previous_1
            order_one = order_one_previous_1
        else:
            zonal = (
                (2 * order - 1) * z * zonal_previous_1
                - (order - 1) * radius_squared * zonal_previous_2
            ) / order
            order_one = (
                (2 * order - 1) * z * order_one_previous_1
                - order * radius_squared * order_one_previous_2
            ) / (order - 1)
            zonal_previous_2, zonal_previous_1 = zonal_previous_1, zonal
            order_one_previous_2, order_one_previous_1 = (
                order_one_previous_1,
                order_one,
            )

        result[..., 0] -= coefficients.alpha[0, order, 1] * x * order_one
        result[..., 1] -= coefficients.beta[1, order, 1] * y * order_one
        result[..., 2] += coefficients.alpha[2, order, 0] * zonal
    return 10.0 * result


class Gradunwarp:
    """Callable gradient-nonlinearity correction for 2D or 3D images.

    Parameters
    ----------
    coefficients
        Parsed neutral spherical-harmonic coefficients.
    input_geometry
        Physical geometry of the trailing image axes.
    output_geometry
        Desired corrected grid.  Defaults to ``input_geometry``.  Supplying a
        different geometry supports simultaneous correction and resampling.
    jacobian
        Apply the full 3D Jacobian intensity multiplier, including when
        sampling a single 2D plane.

    Examples
    --------
    What a real gradient does to the edge of a large field of view, and what
    undoing it moves. The shift grows with distance from isocentre, which is
    why a head-sized acquisition sees almost none of it:

    .. plot::

       import numpy as np
       import pulserver.recon as recon
       from _figures import images, phantom

       alpha = np.zeros((3, 4, 4))
       beta = np.zeros_like(alpha)
       alpha[0, 3, 1] = 8e-5
       beta[1, 3, 1] = 8e-5
       geometry = recon.ImageGeometry(
           (64, 64),
           (400.0, 400.0),
           np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
           np.zeros(3),
       )
       correct = recon.Gradunwarp(
           recon.GradientCoefficients("unnormalized", alpha, beta), geometry
       )

       acquired = phantom(64)[0][0].numpy().real.astype(np.float32)
       grid = np.stack(np.meshgrid(np.arange(64), np.arange(64), indexing="ij"), -1)
       shift = np.linalg.norm(correct.sampling_grid() - grid, axis=-1)
       images(
           [("as acquired", acquired), ("unwarped", correct(acquired)),
            ("shift [px]", shift)],
           title="gradient nonlinearity over a 400 mm field of view",
       )
    """

    def __init__(
        self,
        coefficients: GradientCoefficients,
        input_geometry: ImageGeometry,
        output_geometry: ImageGeometry | None = None,
        *,
        jacobian: bool = True,
    ) -> None:
        output_geometry = output_geometry or input_geometry
        if input_geometry.ndim != output_geometry.ndim:
            raise ValueError("Input and output geometry dimensionality must match.")
        self.coefficients = coefficients
        self.input_geometry = input_geometry
        self.output_geometry = output_geometry
        self.jacobian = bool(jacobian)
        self._source_indices: np.ndarray | None = None
        self._jacobian_multiplier: np.ndarray | None = None
        self._simpleitk_cache: tuple[object, object] | None = None

    @classmethod
    def from_file(
        cls,
        coefficient_source: CoefficientSource,
        input_geometry: ImageGeometry,
        output_geometry: ImageGeometry | None = None,
        *,
        coefficient_format: CoefficientFormat = "auto",
        reference_radius_mm: float | None = None,
        **kwargs: object,
    ) -> Gradunwarp:
        """Construct from a coefficient path, serialized text, or MRD accessor."""
        return cls(
            GradientCoefficients.from_file(
                coefficient_source,
                coefficient_format=coefficient_format,
                reference_radius_mm=reference_radius_mm,
            ),
            input_geometry,
            output_geometry,
            **kwargs,
        )

    def clear_cache(self) -> None:
        """Drop the cached physical sampling grid and Jacobian."""
        self._source_indices = None
        self._jacobian_multiplier = None
        self._simpleitk_cache = None

    def sampling_grid(self, *, copy: bool = True) -> np.ndarray:
        """Return floating input indices sampled for every output voxel."""
        self._prepare_grid()
        assert self._source_indices is not None
        return self._source_indices.copy() if copy else self._source_indices

    def jacobian_grid(self, *, copy: bool = True) -> np.ndarray:
        """Return the cached full-3D Jacobian intensity multiplier."""
        self._prepare_grid()
        if self._jacobian_multiplier is None:
            value = np.ones(self.output_geometry.shape, dtype=np.float64)
        else:
            value = self._jacobian_multiplier
        return value.copy() if copy else value

    def _prepare_grid(self) -> None:
        if self._source_indices is not None:
            return
        coordinates = self.output_geometry.scanner_grid()
        field = _evaluate_harmonics(self.coefficients, coordinates)
        displacement = self.coefficients.mapping_sign * field
        source_coordinates = coordinates + displacement
        source_indices = self.input_geometry.scanner_to_indices(source_coordinates)
        source_indices.setflags(write=False)
        self._source_indices = source_indices

        if self.jacobian:
            multiplier = self._full_jacobian(coordinates, displacement)
            multiplier.setflags(write=False)
            self._jacobian_multiplier = multiplier

    def _full_jacobian(
        self,
        coordinates: np.ndarray,
        displacement: np.ndarray,
    ) -> np.ndarray:
        geometry = self.output_geometry
        if geometry.ndim == 3:
            return self._simpleitk_jacobian(displacement)

        directional = np.empty((*geometry.shape, 3, 3), dtype=np.float64)
        basis = np.empty((3, 3), dtype=np.float64)

        for axis in range(geometry.ndim):
            basis[:, axis] = geometry.direction[:, axis]
            if geometry.shape[axis] > 1:
                edge_order = 2 if geometry.shape[axis] > 2 else 1
                for component in range(3):
                    directional[..., component, axis] = np.gradient(
                        displacement[..., component],
                        geometry.voxel_size_mm[axis],
                        axis=axis,
                        edge_order=edge_order,
                    )
            else:
                epsilon = max(geometry.voxel_size_mm[axis] * 1e-3, 1e-3)
                offset = geometry.direction[:, axis] * epsilon
                plus = self.coefficients.mapping_sign * _evaluate_harmonics(
                    self.coefficients, coordinates + offset
                )
                minus = self.coefficients.mapping_sign * _evaluate_harmonics(
                    self.coefficients, coordinates - offset
                )
                directional[..., :, axis] = (plus - minus) / (2.0 * epsilon)

        if geometry.ndim == 2:
            normal = np.cross(geometry.direction[:, 0], geometry.direction[:, 1])
            normal /= np.linalg.norm(normal)
            basis[:, 2] = normal
            epsilon = max(float(np.min(geometry.voxel_size_mm)) * 1e-3, 1e-3)
            plus = self.coefficients.mapping_sign * _evaluate_harmonics(
                self.coefficients, coordinates + normal * epsilon
            )
            minus = self.coefficients.mapping_sign * _evaluate_harmonics(
                self.coefficients, coordinates - normal * epsilon
            )
            directional[..., :, 2] = (plus - minus) / (2.0 * epsilon)

        derivative_scanner = np.einsum(
            "...ob,sb->...os",
            directional,
            basis,
        )
        mapping_jacobian = derivative_scanner + np.eye(3)
        return np.abs(np.linalg.det(mapping_jacobian))

    def _simpleitk_jacobian(self, displacement: np.ndarray) -> np.ndarray:
        """Compute a 3D physical Jacobian with SimpleITK's compiled filter."""
        import SimpleITK as sitk

        geometry = self.output_geometry
        displacement_in_image_basis = displacement @ geometry.direction
        field = sitk.GetImageFromArray(
            np.ascontiguousarray(displacement_in_image_basis[..., ::-1]),
            isVector=True,
        )
        field.SetSpacing(tuple(geometry.voxel_size_mm[::-1]))
        determinant = sitk.DisplacementFieldJacobianDeterminantFilter().Execute(field)
        return np.abs(sitk.GetArrayFromImage(determinant))

    def __call__(self, image: object) -> np.ndarray:
        """Apply cubic B-spline gradient nonlinearity correction to ``image``."""
        self._prepare_grid()
        return self._call_simpleitk(image)

    def _validate_shape(self, shape: tuple[int, ...]) -> None:
        ndim = self.input_geometry.ndim
        if len(shape) < ndim or tuple(shape[-ndim:]) != self.input_geometry.shape:
            raise ValueError(
                "Image trailing shape must match input_geometry.shape "
                f"{self.input_geometry.shape}, got {shape}."
            )

    def _call_simpleitk(self, image: object) -> np.ndarray:
        import SimpleITK as sitk

        array = np.asarray(image)
        self._validate_shape(array.shape)
        if not (np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array)):
            array = array.astype(np.float32)
        ndim = self.input_geometry.ndim
        leading_shape = array.shape[:-ndim]
        batch = math.prod(leading_shape) if leading_shape else 1
        values = array.reshape((batch, *self.input_geometry.shape))
        assert self._source_indices is not None

        if self._simpleitk_cache is None:
            output_indices = np.moveaxis(
                np.indices(self.output_geometry.shape, dtype=np.float64),
                0,
                -1,
            )
            displacement = self._source_indices - output_indices
            field = sitk.GetImageFromArray(
                np.ascontiguousarray(displacement[..., ::-1]),
                isVector=True,
            )
            transform = sitk.DisplacementFieldTransform(field)
            reference = sitk.Image(
                list(reversed(self.output_geometry.shape)),
                sitk.sitkFloat32,
            )
            self._simpleitk_cache = (transform, reference)
        transform, reference = self._simpleitk_cache

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(0.0)

        def sample(value: np.ndarray) -> np.ndarray:
            source = sitk.GetImageFromArray(np.ascontiguousarray(value))
            return sitk.GetArrayFromImage(resampler.Execute(source))

        output = []
        for value in values:
            if np.iscomplexobj(value):
                output.append(sample(value.real) + 1j * sample(value.imag))
            else:
                output.append(sample(value))
        result = np.stack(output).reshape(leading_shape + self.output_geometry.shape)
        if self._jacobian_multiplier is not None:
            result *= self._jacobian_multiplier.astype(result.dtype, copy=False)
        return result
