"""Private EPI acquisition-role helpers for MRD streams.

The helpers only partition an MRD stream.  They deliberately do not estimate
an odd/even phase ramp: that numerical operation belongs to a selected
reconstruction backend, not to Pulserver.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .metadata import acquisition_label, has_acquisition_flag

__all__ = ["EpiAcquisitionGroups", "partition_epi_acquisitions"]


@dataclass(frozen=True)
class EpiAcquisitionGroups:
    """MRD acquisitions partitioned by the EPI labels emitted by the zoo.

    ``phase_correction`` contains the blip-nulled navigator (``NAV``/``REF``),
    ``reverse_polarity`` the optional ``SET=1`` reference volume, and
    ``single_band_reference`` the SMS ``REF`` data.  All remaining samples
    are ``imaging`` data.
    """

    phase_correction: list[Any]
    single_band_reference: list[Any]
    reverse_polarity: list[Any]
    imaging: list[Any]


def _has_any_flag(acquisition: Any, flags: tuple[str, ...]) -> bool:
    return any(has_acquisition_flag(acquisition, flag) for flag in flags)


def partition_epi_acquisitions(
    acquisitions: Iterable[Any], *, reverse_polarity_set: int = 1
) -> EpiAcquisitionGroups:
    """Partition EPI acquisitions using standard MRD flags and ``idx.set``.

    Parameters
    ----------
    acquisitions
        MRD acquisition stream produced by the EPI zoo sequence.
    reverse_polarity_set
        ``idx.set`` value reserved for the reverse phase-encode reference.
        The EPI zoo uses ``1`` and uses ``0`` for imaging data.

    Returns
    -------
    EpiAcquisitionGroups
        Ordered lists for phase correction, SMS reference, reverse-polarity,
        and image data.
    """
    phase_correction: list[Any] = []
    single_band_reference: list[Any] = []
    reverse_polarity: list[Any] = []
    imaging: list[Any] = []
    for acquisition in acquisitions:
        if _has_any_flag(
            acquisition, ("ACQ_IS_PHASECORR_DATA", "ACQ_IS_NAVIGATION_DATA")
        ):
            phase_correction.append(acquisition)
        elif acquisition_label(acquisition, "set", 0) == reverse_polarity_set:
            reverse_polarity.append(acquisition)
        elif _has_any_flag(
            acquisition,
            ("ACQ_IS_PARALLEL_CALIBRATION", "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING"),
        ):
            single_band_reference.append(acquisition)
        else:
            imaging.append(acquisition)
    return EpiAcquisitionGroups(
        phase_correction, single_band_reference, reverse_polarity, imaging
    )
