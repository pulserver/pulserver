"""Pulserver's design toolbox: factories for sequence modules and scan loops.

Everything here builds one of the two authoring types the plugin contract
exchanges — a :class:`~pulserver.SequenceModule` (a stateful, reusable
multi-block fragment: an excitation, a preparation, a whole readout shot) or a
:class:`~pulserver.ScanLoop` (a table of encoding positions grouped into
shots: k-space, slices, frames, contrasts).

The split from :mod:`pulserver.pypulseq` is by *role*, not by implementation.
``pulserver.pypulseq`` is the event layer — PyPulseq itself, plus Pulserver's
replacements for a handful of its objects. ``pulserver.design`` is the layer
above it::

    import pulserver.pypulseq as pp     # events, Sequence, Opts
    import pulserver.design as design  # modules and loops

    system = pp.Opts()
    excitation = design.make_slice_selective_pulse(0.35, 5e-3, system=system)
    readout = design.make_line_readout(system, (0.22, 0.22), (128, 128))
    loop = design.make_cartesian_sampling((128, 128))

    # The sequence is told what repeats and how often, up front: one shot of
    # this readout, once per entry of the loop. Every library row is claimed from
    # that, so the loop below writes numbers into rows that already exist.
    readout.set_state(lin_idx=0)
    seq = pp.Sequence(system, len(loop), readout)
    for shot in loop:
        readout.set_state(lin_idx=shot.lin[0])
        for block in readout:
            seq.add_block(*block)

Nothing here iterates the sequence for you: the loop nesting stays the
plugin's own ``for`` statements.

The public surface is deliberately the factory layer only. The ordering,
mask, tilt and angle generators these factories are built from are private
(:mod:`pulserver.design._lowlevel`): a plugin composes shipped factories and
writes its own ``for`` statements, and a custom ``ScanLoop`` is built from
:class:`~pulserver.ScanLoop` and its ``from_mask`` /
``from_relative_shifts`` constructors — see
:doc:`the custom-module how-to <../how-to/write_a_new_module>`.
"""

from __future__ import annotations

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
    make_slice_selective_pulse,
    make_t1t2_prep_pulse,
    make_t2prep_pulse,
)
from ._sampling import (  # noqa: F401
    calc_encoding_scales,
    make_cartesian_sampling,
    make_counter_loop,
    make_epi_sampling,
    make_noncartesian_2d_sampling,
    make_noncartesian_projection_sampling,
    make_rotated_projection_sampling,
    make_slice_loop,
)
from ._schedules import make_phase_cycling_schedule, make_rf_spoiling_schedule, make_traps_schedule  # noqa: F401

#: Slice-, slab-, frequency-selective and non-selective excitation modules.
_RF_EXCITATION = {
    "make_frequency_selective_pulse",
    "make_hard_pulse",
    "make_multiband",
    "make_multiband_frequency_selective_pulse",
    "make_pins",
    "make_pins_slice_selective_pulse",
    "make_refocusing_pulse",
    "make_slice_selective_pulse",
}
#: Multidimensional and spectral-spatial excitation modules.
_RF_MULTIDIMENSIONAL = {
    "make_plane_selective_pulse",
}
#: Magnetization-preparation modules.
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
#: One factory per readout family; each returns a whole shot.
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
#: Matrix-driven scan-loop factories, plus the loops for every other axis.
_SAMPLING = {
    "calc_encoding_scales",
    "make_cartesian_sampling",
    "make_counter_loop",
    "make_epi_sampling",
    "make_noncartesian_2d_sampling",
    "make_noncartesian_projection_sampling",
    "make_rotated_projection_sampling",
    "make_slice_loop",
}
#: Per-repetition RF phase and flip-angle lists.
_SCHEDULES = {
    "make_phase_cycling_schedule",
    "make_rf_spoiling_schedule",
    "make_traps_schedule",
}

__all__ = sorted(
    _RF_EXCITATION
    | _RF_MULTIDIMENSIONAL
    | _PREPARATION
    | _READOUTS
    | _SAMPLING
    | _SCHEDULES
)
