"""Delay calculation subroutines."""

__all__ = ["calc_delay"]


from types import SimpleNamespace

import numpy as np
import pypulseq as pp


def calc_delay(
    system: pp.Opts, target_duration: float, *block_or_duration: SimpleNamespace | float
) -> tuple[SimpleNamespace | None, float]:
    """
    Calculate a PyPulseq delay to extend a given block to a target duration.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    target_duration : float
        Target duration in ``[s]``.
    *block_or_duration : SimpleNamespace | float
        Either a PyPulseq block or a time interval in ``[s]``.

    Returns
    -------
    SimpleNamespace | None
        Delay event to reach target duration.
        If current duration is equal or larger than target,
        return ``None``.
    float
        Actual time after application of delay.

    """
    total_duration = 0
    for arg in block_or_duration:
        if isinstance(arg, SimpleNamespace):
            total_duration += pp.calc_duration(arg)
        else:
            total_duration += arg

    if target_duration <= total_duration:
        return None, total_duration
    else:
        return (
            pp.make_delay(
                system.grad_raster_time
                * np.ceil((target_duration - total_duration) / system.grad_raster_time)
            ),
            target_duration,
        )
