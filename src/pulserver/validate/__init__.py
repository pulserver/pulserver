"""Checking that a sequence plays as it was written."""

from ._compare import ChannelAgreement, Comparison, gradient_tolerance_mt_per_m
from ._validate import VENDORS, validate
from ._xml import CHANNELS, PlayedWaveforms, read_waveform_xml

__all__ = [
    "CHANNELS",
    "VENDORS",
    "ChannelAgreement",
    "Comparison",
    "PlayedWaveforms",
    "gradient_tolerance_mt_per_m",
    "read_waveform_xml",
    "validate",
]
