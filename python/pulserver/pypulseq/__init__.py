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

from . import _arbgrad as arbgrad  # noqa: F401
from ._gradients import make_crusher, make_phase_blip, make_phase_encoding, make_spoiler  # noqa: F401
from ._make_label import make_label as _make_label
from ._make_rf_shim import make_rf_shim as _make_rf_shim
from ._make_rotation import make_rotation as _make_rotation
from ._readout._factories import (  # noqa: F401
    make_epi_readout,
    make_fse_readout,
    make_line_readout,
    make_noncartesian_3d_readout,
    make_radial_readout,
    make_rosette_readout,
    make_spiral_readout,
    make_zte_readout,
)
from ._rf import (  # noqa: F401
    make_2d_selective_pulse,
    make_3d_selective_pulse,
    make_adiabatic_pulse,
    make_bloch_siegert_pulse,
    make_diffusion_prep,
    make_fat_saturation_pulse,
    make_frequency_selective_pulse,
    make_hard_pulse,
    make_ihmt_pulse,
    make_inversion_pulse,
    make_mt_pulse,
    make_refocusing_pulse,
    make_sigpy_pulse,
    make_slice_selective_pulse,
    make_slr_pulse,
    make_spatially_selective_pulse,
    make_spiral_selective_pulse,
    make_spsp_pulse,
    make_t1t2_prep_pulse,
    make_t2prep_pulse,
)
from ._sampling import (  # noqa: F401
    SamplingPattern,
    SliceGroup,
    caipirinha_mask,
    chunk_indices,
    directions_to_rotations,
    from_mask,
    from_relative_shifts,
    fse_linear_order,
    fse_radial_adaptive_order,
    fse_radial_order,
    fse_shuffling_order,
    golden_angles,
    golden_means_3d,
    linear_order,
    outer_inner_order,
    outer_product,
    poisson_disc_mask,
    radial_2d,
    random_mask,
    sampled_lines,
    skipped_caipi,
    slice_groups,
    spiral_phyllotaxis,
    uniform_angles,
)
from ._schedules import make_phase_cycling_schedule, make_rf_spoiling_schedule, make_traps_schedule  # noqa: F401
from ._sequence import Sequence as _Sequence
from ._timing import calc_adc_timing  # noqa: F401
from ._traj2grad import traj2grad as _traj2grad

_pulserver_extensions = {name: value for name, value in globals().items() if not name.startswith("_")}
for _name in dir(_pypulseq):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_pypulseq, _name)
globals().update(_pulserver_extensions)

Sequence = _Sequence
make_label = _make_label
make_rf_shim = _make_rf_shim
make_rotation = _make_rotation
traj2grad = _traj2grad

__all__ = sorted(
    {
        *(_name for _name in dir(_pypulseq) if not _name.startswith("_")),
        "Sequence",
        "make_label",
        "make_rf_shim",
        "make_rotation",
        "traj2grad",
        "arbgrad",
        "calc_adc_timing",
        "make_crusher",
        "make_phase_encoding",
        "make_phase_blip",
        "make_spoiler",
        "make_rf_spoiling_schedule",
        "make_phase_cycling_schedule",
        "make_traps_schedule",
        "make_hard_pulse",
        "make_adiabatic_pulse",
        "make_sigpy_pulse",
        "make_slr_pulse",
        "make_frequency_selective_pulse",
        "make_slice_selective_pulse",
        "make_spsp_pulse",
        "make_spatially_selective_pulse",
        "make_spiral_selective_pulse",
        "make_2d_selective_pulse",
        "make_3d_selective_pulse",
        "make_inversion_pulse",
        "make_refocusing_pulse",
        "make_mt_pulse",
        "make_ihmt_pulse",
        "make_bloch_siegert_pulse",
        "make_t2prep_pulse",
        "make_t1t2_prep_pulse",
        "make_diffusion_prep",
        "make_fat_saturation_pulse",
        "make_line_readout",
        "make_epi_readout",
        "make_fse_readout",
        "make_radial_readout",
        "make_spiral_readout",
        "make_rosette_readout",
        "make_noncartesian_3d_readout",
        "make_zte_readout",
        "SamplingPattern",
        "SliceGroup",
        "chunk_indices",
        "linear_order",
        "outer_inner_order",
        "outer_product",
        "sampled_lines",
        "fse_linear_order",
        "fse_radial_order",
        "fse_radial_adaptive_order",
        "fse_shuffling_order",
        "random_mask",
        "caipirinha_mask",
        "poisson_disc_mask",
        "from_mask",
        "from_relative_shifts",
        "skipped_caipi",
        "radial_2d",
        "golden_angles",
        "uniform_angles",
        "golden_means_3d",
        "spiral_phyllotaxis",
        "directions_to_rotations",
        "slice_groups",
    }
)

del _name, _pulserver_extensions
