"""Shot-tilt angle strategies, vendored from mri-nufft (see NOTICE.md).

Copied from ``mrinufft.trajectories.utils`` (``Tilts`` enum, ``initialize_tilt``,
and the ``CaseInsensitiveEnumMeta``/``StrEnum`` helpers they depend on).
"""

from __future__ import annotations

from enum import Enum, EnumMeta
from numbers import Real

import numpy as np

__all__ = ["Tilts", "initialize_tilt"]


class CaseInsensitiveEnumMeta(EnumMeta):
    """A case-insensitive EnumMeta."""

    def __getitem__(cls, name: str):
        """Allow ``MyEnum['Member'] == MyEnum['MEMBER']``."""
        return super().__getitem__(name.upper())

    def __getattr__(cls, name: str):
        """Allow ``MyEnum.Member == MyEnum.MEMBER``."""
        return super().__getattribute__(name.upper())


class StrEnum(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """An Enum for str that is case insensitive for its attributes."""


class Tilts(StrEnum):
    r"""Enumerate available tilts.

    Notes
    -----
    The following values are accepted for the tilt name, with :math:`N` the
    number of shots/partitions:

    - "none": no tilt
    - "uniform": uniform tilt: :math:`2\pi / N`
    - "intergaps": :math:`\pi/2N`
    - "inverted": inverted tilt :math:`\pi/N + \pi`
    - "golden": tilt of the golden angle :math:`\pi(3-\sqrt{5})`
    - "mri-golden": tilt of the golden angle used in MRI :math:`\pi(\sqrt{5}-1)/2`
    """

    UNIFORM = "uniform"
    NONE = "none"
    INTERGAPS = "intergaps"
    INVERTED = "inverted"
    GOLDEN = "golden"
    MRI_GOLDEN = "mri-golden"
    MRI = MRI_GOLDEN


def initialize_tilt(tilt: str | float | None, nb_partitions: int = 1) -> float:
    r"""Initialize the tilt angle increment.

    Parameters
    ----------
    tilt : str | float | None
        Tilt angle in rad, or a name from :class:`Tilts`.
    nb_partitions : int, optional
        Number of partitions/shots. The default is 1.

    Returns
    -------
    float
        Tilt (per-shot angle increment) in rad.

    Raises
    ------
    NotImplementedError
        If the tilt name is unknown.

    See Also
    --------
    Tilts
    """
    if isinstance(tilt, Real):
        return float(tilt)
    elif tilt is None or tilt == Tilts.NONE:
        return 0
    elif tilt == Tilts.UNIFORM:
        return 2 * np.pi / nb_partitions
    elif tilt == Tilts.INTERGAPS:
        return np.pi / nb_partitions / 2
    elif tilt == Tilts.INVERTED:
        return np.pi / nb_partitions + np.pi
    elif tilt == Tilts.GOLDEN:
        return np.pi * (3 - np.sqrt(5))
    elif tilt == Tilts.MRI_GOLDEN:
        return np.pi * (np.sqrt(5) - 1) / 2
    else:
        raise NotImplementedError(f"Unknown tilt name: {tilt}")
