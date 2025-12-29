"""
Pulseq sub-package.

This sub-package contains all low-level routines extending Pulseq design toolbox.

"""

__all__ = [
    "Sequence",
    "Opts",
    "add_gradients",
    "align",
    "calc_duration",
    "calc_kspace_band_jump",
    "calc_kspace_line_jump",
    "calc_kspace_readout_params",
    "calc_rf_bandwidth",
    "calc_rf_center",
    "convert",
    "points_to_waveform",
    # "split_waveform",
    # "split_waveform_at",
    "traj_to_grad",
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
from pypulseq import Sequence
from pypulseq import Opts


# %% Utilities
from pypulseq.add_gradients import add_gradients
from pypulseq.align import align
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_bandwidth import calc_rf_bandwidth
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.convert import convert
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.traj_to_grad import traj_to_grad

from .utils import calc_kspace_line_jump
from .utils import calc_kspace_band_jump
from .utils import calc_kspace_readout_params
# from .utils import split_waveform
# from .utils import split_waveform_at


# %% RF Pulses


# %% Gradient Waveforms


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













