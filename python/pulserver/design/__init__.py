"""
Design sub-package.

This sub-package contains all high-level routines containings Sequence blocks.

"""

__all__ = [
    'FrequencySelectiveExcitation',
    'NonselectiveExcitation',
    'SmsExcitation',
    'SpatiallySelectiveExcitation',
    'SpspExcitation',
    'fse_line_readout',
    'general_line_readout',
    'line_readout',
    'make_blip',
    'make_crusher',
    'make_phasor',
    'spoiled_line_readout',
]

# %% Excitations
from .excitation import (
    FrequencySelectiveExcitation,
    NonselectiveExcitation,
    SmsExcitation,
    SpatiallySelectiveExcitation,
    SpspExcitation,
)

# %% Phasors
from .phasor import make_blip, make_crusher, make_phasor

# %% Readouts
from .readout import (
    fse_line_readout,
    general_line_readout,
    line_readout,
    spoiled_line_readout,
)
