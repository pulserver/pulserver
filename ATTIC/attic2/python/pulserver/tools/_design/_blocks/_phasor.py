"""
"""

__all__ = []

import logging
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from .._params import calc_kspace_readout_params


def log_and_raise(msg, exc=ValueError):
    logging.error(msg)
    raise exc(msg)
    
def make_phasor(
    area: float,
    system: pp.Opts | None = None,
) -> SimpleNamespace:
    if system is None:
        system = pp.Opts.default
    
    
    
    
    
    
    
