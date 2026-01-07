"""
Pulseq sub-package.

This sub-package contains all low-level routines extending Pulseq design toolbox.

"""

__all__ = [
    "Sequence",
    "Opts",
    "DUMMY_OPTS",
    "add_gradients",
    "align",
    "calc_waveform_area",
    "calc_duration",
    "calc_kspace_band_jump",
    "calc_kspace_line_jump",
    "calc_kspace_readout_params",
    "calc_rf_bandwidth",
    "calc_rf_center",
    "convert",
    "points_to_waveform",
    "split_waveform",
    "split_waveform_at",
    "time_revert_waveform",
    "traj_to_grad",
    "make_adiabatic_pulse",
    # "make_adiabatic_t2prep",
    "make_arbitrary_rf",
    "make_block_pulse",
    "make_slr_pulse",
    "make_sms_pulse",
    "make_spsp_pulse",
    "make_arbitrary_grad",
    # "make_epi",
    "make_extended_trapezoid",
    "make_extended_trapezoid_area",
    # "make_spiral",
    "make_trapezoid",
    # "make_wave",
    "make_caipirinha_sampling",
    "make_partial_fourier_sampling",
    "make_poisson_disk_sampling",
    "make_regular_sampling",
    "make_centerout_ordering_1d",
    "make_centerout_ordering_2d",
    "make_interleaved_ordering_1d",
    "make_radial_ordering_2d",
    "make_random_ordering_1d",
    "make_random_ordering_2d",
    "make_spiral_ordering_2d",
]

# %% Core
from pypulseq import Sequence # pragma: no cover
from pypulseq import Opts # pragma: no cover


# %% Utilities
from pypulseq.add_gradients import add_gradients # pragma: no cover
from pypulseq.align import align # pragma: no cover
from pypulseq.calc_duration import calc_duration # pragma: no cover
from pypulseq.calc_rf_bandwidth import calc_rf_bandwidth # pragma: no cover
from pypulseq.calc_rf_center import calc_rf_center # pragma: no cover
from pypulseq.convert import convert # pragma: no cover
from pypulseq.points_to_waveform import points_to_waveform # pragma: no cover
from pypulseq.traj_to_grad import traj_to_grad # pragma: no cover

from .utils import DUMMY_OPTS
from .utils import calc_kspace_line_jump
from .utils import calc_kspace_band_jump
from .utils import calc_kspace_readout_params
from .utils import split_waveform
from .utils import split_waveform_at
from .utils import time_revert_waveform


# %% RF Pulses
from pypulseq.make_adiabatic_pulse import make_adiabatic_pulse # pragma: no cover
from pypulseq.make_arbitrary_rf import make_arbitrary_rf # pragma: no cover
from pypulseq.make_block_pulse import make_block_pulse # pragma: no cover

from .rf import make_slr_pulse
from .rf import make_sms_pulse
from .rf import make_spsp_pulse

# %% Gradient Waveforms
from pypulseq.make_arbitrary_grad import make_arbitrary_grad # pragma: no cover
from pypulseq.make_extended_trapezoid import make_extended_trapezoid # pragma: no cover
from pypulseq.make_extended_trapezoid_area import make_extended_trapezoid_area # pragma: no cover
from pypulseq.make_trapezoid import make_trapezoid # pragma: no cover

# from .grad import make_epi
# from .grad import make_spiral
# from .grad import make_wave

# %% Sampling Patterns
from .sampling import make_caipirinha_sampling
from .sampling import make_partial_fourier_sampling
from .sampling import make_poisson_disk_sampling
from .sampling import make_regular_sampling


# %% Acquisition Orderings
from .ordering import make_centerout_ordering_1d
from .ordering import make_centerout_ordering_2d
from .ordering import make_interleaved_ordering_1d
from .ordering import make_radial_ordering_2d
from .ordering import make_random_ordering_1d
from .ordering import make_random_ordering_2d
from .ordering import make_spiral_ordering_2d
