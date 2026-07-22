"""Pulserver's drop-in replacement for :mod:`pypulseq`.

The complete public upstream namespace is re-exported, then Pulserver's
compatible overrides are layered on top. Sequence authoring therefore needs
one import only::

    import pulserver.pypulseq as pp

    sequence = pp.Sequence()
    delay = pp.make_delay(1e-3)
"""

from __future__ import annotations

# ruff: noqa: I001

import pypulseq as _pypulseq
from functools import wraps as _wraps

from . import _arbgrad as arbgrad  # noqa: F401
from ._gradients import make_crusher, make_phase_blip, make_phase_encoding  # noqa: F401
from ._make_label import get_supported_labels as _get_supported_labels
from ._make_label import make_label as _make_label
from ._make_rf_shim import make_rf_shim as _make_rf_shim
from ._make_rotation import make_rotation as _make_rotation
from ._opts import Opts as _Opts
from ._readout._factories import (  # noqa: F401
    make_bssfp_readout,
    make_epi_readout,
    make_fse_readout,
    make_line_readout,
    make_noncartesian_2d_readout,
    make_noncartesian_3d_readout,
    make_radial_projection_readout,
    make_radial_readout,
    make_radial_stack_readout,
    make_rosette_projection_readout,
    make_rosette_readout,
    make_rosette_stack_readout,
    make_spiral_projection_readout,
    make_spiral_readout,
    make_spiral_stack_readout,
    make_zte_readout,
)
from ._rf import (  # noqa: F401
    make_bloch_siegert_pulse,
    make_diffusion_prep,
    make_fat_saturation_pulse,
    make_frequency_selective_pulse,
    make_hard_pulse,
    make_ihmt_pulse,
    make_inversion_pulse,
    make_mt_pulse,
    make_multiband,
    make_multiband_frequency_selective_pulse,
    make_pins,
    make_pins_slice_selective_pulse,
    make_plane_selective_pulse,
    make_refocusing_pulse,
    make_sigpy_pulse,
    make_slr_pulse,
    make_slice_selective_pulse,
    make_spsp_pulse,
    make_t1t2_prep_pulse,
    make_t2prep_pulse,
)
from ._sampling import (  # noqa: F401
    Acquisition,
    AcquisitionPlan,
    SamplingPattern,
    SliceGroup,
    SliceSampling,
    calc_golden_angles,
    calc_raga_angles,
    calc_sampled_lines,
    calc_tiny_golden_angles,
    calc_uniform_angles,
    make_caipirinha_mask,
    make_cartesian_sampling,
    make_centric_order,
    make_epi_sampling,
    make_golden_means_3d_tilt,
    make_linear_order,
    make_noncartesian_2d_sampling,
    make_noncartesian_projection_sampling,
    make_noncartesian_stack_sampling,
    make_poisson_disc_mask,
    make_radial_adaptive_order,
    make_radial_order,
    make_radial_tilt,
    make_random_mask,
    make_shuffling_order,
    make_skipped_caipi_order,
    make_slice_sampling,
    make_spiral_phyllotaxis_tilt,
)
from ._schedules import make_phase_cycling_schedule, make_rf_spoiling_schedule, make_traps_schedule  # noqa: F401
from ._sequence import Sequence as _Sequence
from ._timing import calc_adc_timing  # noqa: F401
from ._traj2grad import traj2grad as _traj2grad

_pulserver_extensions = {name: value for name, value in globals().items() if not name.startswith("_")}
_excluded_upstream = {"make_adiabatic_pulse"}
for _name in dir(_pypulseq):
    if not _name.startswith("_") and _name not in _excluded_upstream:
        globals()[_name] = getattr(_pypulseq, _name)
globals().update(_pulserver_extensions)


def _promote(upstream):
    """Wrap an upstream callable so introspection reports this namespace."""

    @_wraps(upstream, assigned=("__name__", "__qualname__", "__doc__", "__annotations__"))
    def wrapper(*args, **kwargs):
        return upstream(*args, **kwargs)

    return wrapper


Sequence = _Sequence
Opts = _Opts
compress_shape = _promote(_pypulseq.compress_shape.compress_shape)
convert = _promote(_pypulseq.convert.convert)
decompress_shape = _promote(_pypulseq.decompress_shape.decompress_shape)
get_supported_labels = _get_supported_labels
make_label = _make_label
make_rf_shim = _make_rf_shim
make_rotation = _make_rotation
traj2grad = _traj2grad

# Public surface, grouped exactly as the API reference presents it.  The
# groups are the single source of truth for both ``__all__`` and the
# documentation's section membership test.
_UPSTREAM = {_name for _name in dir(_pypulseq) if not _name.startswith("_")} - _excluded_upstream
_OVERRIDES = {
    "Sequence",
    "Opts",
    "calc_adc_timing",
    "compress_shape",
    "convert",
    "decompress_shape",
    "get_supported_labels",
    "make_label",
    "make_rf_shim",
    "make_rotation",
    "traj2grad",
}
_GRADIENTS = {
    "make_crusher",
    "make_phase_blip",
    "make_phase_encoding",
}
_RF_EXCITATION = {
    "make_frequency_selective_pulse",
    "make_hard_pulse",
    "make_multiband",
    "make_multiband_frequency_selective_pulse",
    "make_pins",
    "make_pins_slice_selective_pulse",
    "make_refocusing_pulse",
    "make_slr_pulse",
    "make_slice_selective_pulse",
}
_RF_MULTIDIMENSIONAL = {
    "make_plane_selective_pulse",
    "make_spsp_pulse",
}
_PREPARATION = {
    "make_bloch_siegert_pulse",
    "make_diffusion_prep",
    "make_fat_saturation_pulse",
    "make_ihmt_pulse",
    "make_inversion_pulse",
    "make_mt_pulse",
    "make_t1t2_prep_pulse",
    "make_t2prep_pulse",
}
_READOUTS = {
    "make_bssfp_readout",
    "make_epi_readout",
    "make_fse_readout",
    "make_line_readout",
    "make_noncartesian_2d_readout",
    "make_noncartesian_3d_readout",
    "make_radial_projection_readout",
    "make_radial_readout",
    "make_radial_stack_readout",
    "make_rosette_projection_readout",
    "make_rosette_readout",
    "make_rosette_stack_readout",
    "make_spiral_projection_readout",
    "make_spiral_readout",
    "make_spiral_stack_readout",
    "make_zte_readout",
}
_SAMPLING = {
    "Acquisition",
    "AcquisitionPlan",
    "calc_golden_angles",
    "calc_raga_angles",
    "calc_sampled_lines",
    "calc_tiny_golden_angles",
    "calc_uniform_angles",
    "make_caipirinha_mask",
    "make_cartesian_sampling",
    "make_centric_order",
    "make_epi_sampling",
    "make_golden_means_3d_tilt",
    "make_linear_order",
    "make_noncartesian_2d_sampling",
    "make_noncartesian_projection_sampling",
    "make_noncartesian_stack_sampling",
    "make_poisson_disc_mask",
    "make_radial_adaptive_order",
    "make_radial_order",
    "make_radial_tilt",
    "make_random_mask",
    "make_shuffling_order",
    "make_skipped_caipi_order",
    "make_slice_sampling",
    "make_spiral_phyllotaxis_tilt",
}
_SCHEDULES = {
    "make_phase_cycling_schedule",
    "make_rf_spoiling_schedule",
    "make_traps_schedule",
}

__all__ = sorted(
    _UPSTREAM
    | _OVERRIDES
    | _GRADIENTS
    | _RF_EXCITATION
    | _RF_MULTIDIMENSIONAL
    | _PREPARATION
    | _READOUTS
    | _SAMPLING
    | _SCHEDULES
)

del _name, _pulserver_extensions
