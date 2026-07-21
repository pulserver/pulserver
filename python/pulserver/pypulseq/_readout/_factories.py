"""Factories for public readout modules."""

from __future__ import annotations

from collections.abc import Sequence

from .epi import Epi2D, Epi2DFlyback, Epi3D, Epi3DFlyback
from .fse import Fse2D, Fse3D
from .line import Line2D, Line3D
from .noncartesian import NonCartesian2D, NonCartesian3D, Radial, Rosette, Spiral
from .zte import Zte


def _dimensions(matrix) -> int:
    if isinstance(matrix, int):
        return 1
    if not isinstance(matrix, Sequence):
        raise TypeError("matrix must be an integer or a sequence")
    return len(matrix)


def make_line_readout(system, fov, matrix, num_echoes: int = 1, **kwargs):
    """Create a 2D or 3D Cartesian line readout module."""
    dimensions = _dimensions(matrix)
    if dimensions == 2:
        return Line2D(system, fov, matrix, num_echoes, **kwargs)
    if dimensions == 3:
        return Line3D(system, fov, matrix, num_echoes, **kwargs)
    raise ValueError("line readout matrix must have two or three dimensions")


def make_epi_readout(system, fov, matrix, num_shots, sampling_mask, *, flyback: bool = False, **kwargs):
    """Create a 2D or 3D blipped or flyback EPI readout module."""
    dimensions = _dimensions(matrix)
    classes = {(2, False): Epi2D, (2, True): Epi2DFlyback, (3, False): Epi3D, (3, True): Epi3DFlyback}
    try:
        factory = classes[(dimensions, bool(flyback))]
    except KeyError as error:
        raise ValueError("EPI matrix must have two or three dimensions") from error
    return factory(system, fov, matrix, num_shots, sampling_mask, **kwargs)


def make_fse_readout(system, fov, matrix, echo_train_length, refocusing, **kwargs):
    """Create a 2D or 3D FSE readout from a refocusing RF module."""
    try:
        rf = refocusing.rf
        gradient = refocusing.gradients[0]
    except (AttributeError, IndexError):
        try:
            rf, gradient = refocusing
        except (TypeError, ValueError) as error:
            raise TypeError("refocusing must be an RF module or an (rf, gradient) pair") from error
    dimensions = _dimensions(matrix)
    if dimensions == 2:
        return Fse2D(system, fov, matrix, echo_train_length, rf, gradient, **kwargs)
    if dimensions == 3:
        return Fse3D(system, fov, matrix, echo_train_length, rf, gradient, **kwargs)
    raise ValueError("FSE matrix must have two or three dimensions")


def make_radial_readout(system, fov, matrix, **kwargs):
    """Create a reusable 2D radial readout module."""
    return NonCartesian2D(Radial(system, fov, matrix, **kwargs))


def make_spiral_readout(system, fov, matrix, design_interleaves, **kwargs):
    """Create a reusable 2D spiral readout module."""
    return NonCartesian2D(Spiral(system, fov, matrix, design_interleaves, **kwargs))


def make_rosette_readout(system, fov, matrix, **kwargs):
    """Create a reusable 2D rosette readout module."""
    return NonCartesian2D(Rosette(system, fov, matrix, **kwargs))


def make_noncartesian_3d_readout(readout, **kwargs):
    """Promote a native 3D non-Cartesian gradient to a readout module."""
    return NonCartesian3D(readout, **kwargs)


def make_zte_readout(system, fov, matrix, view_order, excitation, **kwargs):
    """Create a continuous-gradient ZTE readout module."""
    rf = getattr(excitation, "rf", excitation)
    return Zte(system, fov, matrix, view_order, rf, **kwargs)
