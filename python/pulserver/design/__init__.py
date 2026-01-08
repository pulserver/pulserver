"""
Design sub-package.

This sub-package contains all high-level routines containings Sequence blocks.

"""

__all__ = [
    "NonselectiveExcitation",
    "FrequencySelectiveExcitation",
    "SpatiallySelectiveExcitation",
    "SmsExcitation",
    "SpspExcitation",
    "general_line_readout",
    "line_readout",
    "spoiled_line_readout",
    "fse_line_readout",
    "make_phasor",
    "make_blip",
    "make_crusher",
]

# %% Excitations
from .excitation import NonselectiveExcitation
from .excitation import FrequencySelectiveExcitation
from .excitation import SpatiallySelectiveExcitation
from .excitation import SmsExcitation
from .excitation import SpspExcitation

# %% Readouts
from .readout import general_line_readout
from .readout import line_readout
from .readout import spoiled_line_readout
from .readout import fse_line_readout

# %% Phasors
from .phasor import make_phasor
from .phasor import make_blip
from .phasor import make_crusher
