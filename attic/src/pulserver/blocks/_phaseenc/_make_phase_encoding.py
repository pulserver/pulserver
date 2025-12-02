"""Phase encoding gradient creation subroutines."""

__all__ = ["make_phase_encoding"]

from types import SimpleNamespace

import pypulseq as pp


def make_phase_encoding(
    channel: str,
    system: pp.Opts,
    fov: float,
    npix: int,
) -> SimpleNamespace:
    """
    Prepare phase encoding gradient for a given resolution.

    Parameters
    ----------
    channel : str
        Phase encoding axis. Must be
        one between ``x``, ``y`` and `z``.
    system : pypulseq.Opts
        System limits.
    fov : float
        Field of view in the phase encoding direction in ``[mm]``.
    npix : int
        Matrix size in the phase encoding direction,

    Returns
    -------
    SimpleNamespace
        Phase encoding event on the specified axes.


    """
    # get axis
    if channel not in ["x", "y", "z"]:
        raise ValueError(f"Unrecognized channel {channel} - must be 'x', 'y', or 'z'.")

    # unit conversion (mm -> m)
    fov *= 1e-3

    # k space area
    dr = fov / npix

    # prepare phase encoding gradient lobe
    return pp.make_trapezoid(channel=channel, area=1 / dr, system=system)
