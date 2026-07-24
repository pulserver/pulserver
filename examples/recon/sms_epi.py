"""Prepare SMS-EPI inputs without reimplementing Slice-GRAPPA or SENSE."""

from __future__ import annotations

from typing import Any

from pulserver.recon.sms import SmsEpiInputs

__all__ = ["prepare_sms_epi"]


def prepare_sms_epi(
    imaging: Any,
    *,
    multiband_factor: int,
    caipi_encoding: Any,
    coil_maps: Any | None = None,
    single_band_reference: Any | None = None,
) -> SmsEpiInputs:
    """Validate the acquired SMS inputs for MRPro/DeepInverse or Slice-GRAPPA.

    Model-based reconstruction supplies ``coil_maps``; Slice-GRAPPA supplies
    ``single_band_reference``.  Both require the known acquired CAIPI model.
    """
    inputs = SmsEpiInputs(imaging, coil_maps, single_band_reference, caipi_encoding)
    inputs.validate(multiband_factor)
    return inputs
