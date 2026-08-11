"""Private EPI MRD adapter example for an upstream reconstruction backend.

The EPI sequence labels a blip-nulled odd/even navigator with ``NAV``/``REF``
and writes an optional opposite phase-encode b=0 reference as ``SET=1``.
This example exposes those groups without implementing phase fitting or field
map estimation locally.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pulserver.recon.corrections import run_pyhysco
from pulserver.recon._mrd.epi import EpiAcquisitionGroups, partition_epi_acquisitions

__all__ = ["prepare_epi", "run_epi_distortion_correction"]


def prepare_epi(acquisitions: Iterable[Any]) -> EpiAcquisitionGroups:
    """Return navigator, SMS-reference, reverse-PE, and image acquisitions.

    Send ``phase_correction`` and ``imaging`` to the chosen EPI-capable
    backend, then reconstruct the image and ``reverse_polarity`` b=0 volumes
    before calling :func:`run_epi_distortion_correction` if desired.
    """
    return partition_epi_acquisitions(acquisitions)


def run_epi_distortion_correction(
    positive_phase_encode: str | Path,
    negative_phase_encode: str | Path,
    phase_encode_direction: int,
):
    """Delegate reverse-polarity distortion correction to optional PyHySCO."""
    return run_pyhysco(
        positive_phase_encode, negative_phase_encode, phase_encode_direction
    )
