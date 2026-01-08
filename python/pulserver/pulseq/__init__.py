"""
Pulseq sub-package.

This sub-package contains all low-level routines extending Pulseq design toolbox.

"""

__all__ = [
    'DUMMY_OPTS',
    'Opts',
    'Sequence',
    'add_gradients',
    'align',
    'calc_duration',
    'calc_kspace_band_jump',
    'calc_kspace_line_jump',
    'calc_kspace_readout_params',
    'calc_rf_bandwidth',
    'calc_rf_center',
    'calc_spoil_area',
    'calc_waveform_area',
    'convert',
    'make_adc',
    'make_adiabatic_pulse',
    'make_arbitrary_grad',
    # "make_adiabatic_t2prep",
    'make_arbitrary_rf',
    'make_block_pulse',
    # "make_wave",
    'make_caipirinha_sampling',
    'make_centerout_ordering_1d',
    'make_centerout_ordering_2d',
    'make_delay',
    'make_digital_output_pulse',
    # "make_epi",
    'make_extended_trapezoid',
    'make_extended_trapezoid_area',
    'make_interleaved_ordering_1d',
    'make_label',
    'make_partial_fourier_sampling',
    'make_poisson_disk_sampling',
    'make_radial_ordering_2d',
    'make_random_ordering_1d',
    'make_random_ordering_2d',
    'make_regular_sampling',
    'make_slr_pulse',
    'make_sms_pulse',
    'make_spiral_ordering_2d',
    'make_spsp_pulse',
    # "make_spiral",
    'make_trapezoid',
    'make_trigger',
    'points_to_waveform',
    'scale_grad',
    'split_waveform',
    'split_waveform_at',
    'time_revert_waveform',
    'traj_to_grad',
]

# %% Core
from pypulseq import (
    Opts,  # pragma: no cover
    Sequence,  # pragma: no cover
)
from pypulseq.add_gradients import add_gradients  # pragma: no cover
from pypulseq.align import align  # pragma: no cover
from pypulseq.calc_duration import calc_duration  # pragma: no cover
from pypulseq.calc_rf_bandwidth import calc_rf_bandwidth  # pragma: no cover
from pypulseq.calc_rf_center import calc_rf_center  # pragma: no cover
from pypulseq.convert import convert  # pragma: no cover
from pypulseq.make_adc import make_adc  # pragma: no cover
from pypulseq.make_adiabatic_pulse import make_adiabatic_pulse  # pragma: no cover
from pypulseq.make_arbitrary_grad import make_arbitrary_grad  # pragma: no cover
from pypulseq.make_arbitrary_rf import make_arbitrary_rf  # pragma: no cover
from pypulseq.make_block_pulse import make_block_pulse  # pragma: no cover
from pypulseq.make_delay import make_delay  # pragma: no cover
from pypulseq.make_digital_output_pulse import (
    make_digital_output_pulse,
)  # pragma: no cover
from pypulseq.make_extended_trapezoid import make_extended_trapezoid  # pragma: no cover
from pypulseq.make_extended_trapezoid_area import (
    make_extended_trapezoid_area,
)  # pragma: no cover
from pypulseq.make_label import make_label  # pragma: no cover
from pypulseq.make_trapezoid import make_trapezoid  # pragma: no cover
from pypulseq.make_trigger import make_trigger  # pragma: no cover
from pypulseq.points_to_waveform import points_to_waveform  # pragma: no cover
from pypulseq.scale_grad import scale_grad  # pragma: no cover
from pypulseq.traj_to_grad import traj_to_grad  # pragma: no cover

from .ordering import (
    make_centerout_ordering_1d,
    make_centerout_ordering_2d,
    make_interleaved_ordering_1d,
    make_radial_ordering_2d,
    make_random_ordering_1d,
    make_random_ordering_2d,
    make_spiral_ordering_2d,
)
from .rf import make_slr_pulse, make_sms_pulse, make_spsp_pulse

# from .grad import make_epi
# from .grad import make_spiral
# from .grad import make_wave
from .sampling import (
    make_caipirinha_sampling,
    make_partial_fourier_sampling,
    make_poisson_disk_sampling,
    make_regular_sampling,
)
from .utils import (
    DUMMY_OPTS,
    calc_kspace_band_jump,
    calc_kspace_line_jump,
    calc_kspace_readout_params,
    calc_spoil_area,
    split_waveform,
    split_waveform_at,
    time_revert_waveform,
)
