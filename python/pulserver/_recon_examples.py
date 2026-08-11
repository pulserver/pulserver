"""Shared public callbacks for the importable reconstruction examples."""

from __future__ import annotations

__all__ = ["prepare_sms_epi", "run_epi_distortion_correction"]


def prepare_sms_epi(*args, **kwargs):
    """Validate and package SMS-EPI inputs for SENSE or Slice-GRAPPA."""
    from .recon.sms import SmsEpiInputs

    multiband_factor = kwargs.pop("multiband_factor")
    inputs = SmsEpiInputs(*args, **kwargs)
    inputs.validate(multiband_factor)
    return inputs


def run_epi_distortion_correction(*args, **kwargs):
    """Run the explicitly installed PyHySCO CLI on an opposite-PE NIfTI pair."""
    from .recon.corrections import run_pyhysco

    return run_pyhysco(*args, **kwargs)
