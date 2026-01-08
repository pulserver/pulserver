"""
Design sub-package.

This sub-package contains all high-level routines containins Sequence blocks.

"""

__all__ = [
    "NonselectiveExcitation",
    "FrequencySelectiveExcitation",
    "SpatiallySelectiveExcitation",
    "SmsExcitation",
    "SpspExcitation",
    "line_readout",
]

from .excitation import NonselectiveExcitation
from .excitation import FrequencySelectiveExcitation
from .excitation import SpatiallySelectiveExcitation
from .excitation import SmsExcitation
from .excitation import SpspExcitation

from .readout import line_readout