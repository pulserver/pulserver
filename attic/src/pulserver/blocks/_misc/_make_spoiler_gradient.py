"""Spoiler gradient creation subroutines."""

__all__ = ["make_spoiler_gradient"]


from types import SimpleNamespace

import numpy as np
import pypulseq as pp


def make_spoiler_gradient(
    channel: str,
    system: pp.Opts,
    ncycles: int,
    voxel_size: float,
    duration: float | None = None,
) -> SimpleNamespace:
    """
    Prepare spoiler gradient with given dephasing across
    the given spatial length.

    Parameters
    ----------
    channel : str
        Phase encoding axis. Must be
        one between ``x``, ``y`` and `z``.
    system : pypulseq.Opts
        System limits.
    ncycles : int
        Number of spoiling cycles per voxel.
    voxel_size : float
        Voxel size in the spoiling direction in ``[mm]``.
    duration : float | None, optional
        Duration of spoiling gradient in ``[s]``.
        If not provided, use minimum duration
        given by area and system specs.
        The default is ``None`` (minumum duration).

    Returns
    -------
    SimpleNamespace
        Spoiling event on the specified axes.

    """
    # get axis
    if channel not in ["x", "y", "z"]:
        raise ValueError(f"Unrecognized channel {channel} - must be 'x', 'y', or 'z'.")

    dr = voxel_size * 1e-3  # mm -> m

    # prepare phase encoding gradient lobe
    if duration:
        return pp.make_trapezoid(
            channel=channel,
            area=(ncycles * np.pi / dr),
            system=system,
            duration=duration,
        )

    return pp.make_trapezoid(
        channel=channel, area=(ncycles * np.pi / dr), system=system
    )
