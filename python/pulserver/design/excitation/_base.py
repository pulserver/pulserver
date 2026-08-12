"""What every RF module has in common."""

from __future__ import annotations

__all__ = ["RfModule", "rf_reference"]

from typing import Any

from ..._core._module import SequenceModule


def rf_reference(rf: Any) -> float:
    """Time from the start of an RF pulse's block to the point it is timed against.

    ``rf.center`` is measured from the first sample of the envelope, while a TE
    budget counts from the start of the block -- and the two differ by the
    pulse's own ``delay``, which is where a dead time or a selection gradient's
    ramp puts the envelope.
    """
    return float(rf.delay) + float(rf.center)


class RfModule(SequenceModule):
    """A module whose point is an RF pulse.

    Adds the one view the sequence-level analyses cannot give: what the pulse
    does to the magnetisation. Everything else -- publication, ``blocks``,
    ``duration``, plotting, k-space -- is :class:`~pulserver.SequenceModule`'s.
    """

    def sim_rf(self, **kwargs):
        """Simulate this module's pulse across off-resonance.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`pulserver.pypulseq.sim_rf` (``df``,
            ``bandwidth_multiplier``, ``compat``, ...).

        Returns
        -------
        tuple or pulserver.pypulseq.RfResponse
            The transverse and longitudinal profiles against frequency.
            Upstream's tuple by default; pass ``compat=False`` for the named
            form.

        Examples
        --------
        >>> import pulserver.design as design
        >>> import pulserver.pypulseq as pp
        >>> pulse = design.SpatialSelectiveExcitation(pp.Opts(), 15.0, 5e-3)
        >>> response = pulse.sim_rf(compat=False)
        >>> response.mz_z.shape == response.frequency.shape
        True
        """
        from ... import pypulseq as pp

        return pp.sim_rf(self.events.rf, **kwargs)
