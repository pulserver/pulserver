"""Package the inputs a model-based SMS reconstruction needs.

Slice separation is done by :class:`pulserver.recon.physics.SMS`, which wraps
whatever physics encodes the plane underneath the multiband pulse. That is
what makes it independent of the in-plane trajectory: the same call serves a
Cartesian EPI and a spiral, where a k-space method such as Slice-GRAPPA would
serve only the Cartesian one.
"""

from __future__ import annotations

from typing import Any

from pulserver.recon.preprocessing import SmsEpiInputs

__all__ = ["prepare_sms_epi"]


def prepare_sms_epi(
    imaging: Any,
    *,
    multiband_factor: int,
    caipi_encoding: Any,
    coil_maps: Any | None = None,
    single_band_reference: Any | None = None,
) -> SmsEpiInputs:
    """Validate the acquired SMS inputs against the CAIPI model that was played.

    Supply ``coil_maps`` for the model-based separation, or
    ``single_band_reference`` to derive them from the single-band prescan.
    Either way the acquired CAIPI encoding has to be known, because it is what
    distinguishes the simultaneously excited slices.
    """
    inputs = SmsEpiInputs(imaging, coil_maps, single_band_reference, caipi_encoding)
    inputs.validate(multiband_factor)
    return inputs
