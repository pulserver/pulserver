"""Reusable sequence-design building blocks for pulserver plugins.

Requires the optional ``pypulseq`` dependency (same tier as
``pulserver.pulseq`` / ``pulserver.io``).
"""

from __future__ import annotations

from .system import apply_system_derates, quantize_readout_timing, copy_event, round_to_raster
from .params import (
    param_float,
    param_int,
    param_float_optional,
    param_int_optional,
    param_bool_optional,
    user_float,
    phase_fov_mm_from_protocol,
    acs_lines_from_protocol,
    resolve_readout_phase_axes,
    set_protocol_value,
)
from . import cli, encoding, excitation, preparations, pulses, readout, sampling

__all__ = [
    "cli",
    "encoding",
    "excitation",
    "preparations",
    "pulses",
    "readout",
    "sampling",
    "apply_system_derates",
    "quantize_readout_timing",
    "copy_event",
    "round_to_raster",
    "param_float",
    "param_int",
    "param_float_optional",
    "param_int_optional",
    "param_bool_optional",
    "user_float",
    "phase_fov_mm_from_protocol",
    "acs_lines_from_protocol",
    "resolve_readout_phase_axes",
    "set_protocol_value",
]
