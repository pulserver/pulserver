"""MRI physics classes with a uniform DeepInverse-facing interface.

The public classes own the mri-nufft/DeepInverse integration boundary.
Callers never need to construct an mri-nufft autodiff wrapper themselves.
Subspace, off-resonance, and Toeplitz behavior are composed as decorators so
that the API does not grow one class for every possible combination.
"""

from __future__ import annotations

__all__ = [
    "SMS",
    "Cartesian2D",
    "Cartesian3D",
    "MRIPhysics",
    "NonCartesian2D",
    "NonCartesian3D",
    "OffResonance",
    "Subspace",
    "Toeplitz",
    "WaveEncoding",
    "WaveShuffling",
    "available_nufft_backends",
]

from ._base import MRIPhysics
from ._cartesian import SMS, Cartesian2D, Cartesian3D
from ._common import available_nufft_backends
from ._kernel import Toeplitz
from ._noncartesian import NonCartesian2D, NonCartesian3D
from ._offresonance import OffResonance
from ._subspace import Subspace
from ._waveenc import WaveEncoding, WaveShuffling
