"""2D Spoiled Gradient Echo sequence."""

__all__ = ["SPGR2D"]

from pulserver.parsing import Cartesian2DParams
from pulserver.sequences import design_2D_spgr


def SPGR2D(kwargs):
    """
    Generate a 2D Spoiled Gradient Recalled Echo (SPGR) pulse sequence.

    This function wraps ``pulserver.sequences.design_2D_spgr`` - see
    the corresponding docstrings for more information.

    """
    # parse parameters
    params = Cartesian2DParams(**kwargs, fudge_factor=0.9)

    # call design function (exclude header for now)
    seq, _ = design_2D_spgr(**params.asdict(), platform="gehc")

    return seq.export(), None
