"""SMS-EPI data contracts for upstream SENSE or Slice-GRAPPA reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["SmsEpiInputs"]


@dataclass(frozen=True)
class SmsEpiInputs:
    """Inputs required by an SMS-EPI reconstruction backend.

    Parameters
    ----------
    imaging
        Multiband EPI data with the acquisition's known CAIPI encoding.
    coil_maps
        Per-slice coil maps for model-based MRPro/DeepInverse SENSE.  Keep
        this tensor on the selected Torch CPU/CUDA device.
    single_band_reference
        Individual-slice reference data required by Slice-GRAPPA.  It may be
        ``None`` for model-based SENSE.
    caipi_encoding
        The known CAIPI phase/modulation model.  It must be supplied when the
        multiband factor is greater than one.

    Notes
    -----
    This class intentionally does not implement Slice-GRAPPA or SMS unaliasing.
    It makes the required acquired data explicit and leaves the numerical
    reconstruction to the selected upstream backend.
    """

    imaging: Any
    coil_maps: Any | None = None
    single_band_reference: Any | None = None
    caipi_encoding: Any | None = None

    def validate(self, multiband_factor: int) -> None:
        """Raise ``ValueError`` when an SMS backend lacks essential inputs."""
        if multiband_factor < 1:
            raise ValueError("multiband_factor must be at least one")
        if multiband_factor > 1 and self.caipi_encoding is None:
            raise ValueError("SMS reconstruction requires the acquired CAIPI encoding model")
        if multiband_factor > 1 and self.coil_maps is None and self.single_band_reference is None:
            raise ValueError("SMS reconstruction requires coil maps or a single-band reference")
