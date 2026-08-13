"""Excitation confined in two dimensions rather than one."""

from __future__ import annotations

__all__ = ["SpatialSelective2DExcitation"]

import numpy as np

from ... import pypulseq as pp
from ._base import RfModule, rf_reference


class SpatialSelective2DExcitation(RfModule):
    """A pulse that excites a column rather than a slice.

    Two gradients run during the pulse instead of one, tracing a spiral through
    excitation k-space, and the envelope is the desired profile's Fourier
    transform sampled along that path. So the excited region is a disc in plane
    and unbounded through it -- a pencil beam, which is what an inner-volume or
    navigator acquisition wants.

    The spiral ends where it started, so the pulse refocuses itself: there is
    no rephaser to place and ``gradients`` is the whole gradient story.

    What a 2D pulse cannot escape is that the profile **repeats**. The
    trajectory samples excitation k-space at a finite pitch, so a second
    excited disc sits one ``fov`` away along each axis; ``fov`` has to cover
    the object, not just the spot.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float
        Nominal flip angle (degrees), reached at the centre of the disc.
    fov : float
        Excitation field of view (m), square. The distance to the first
        repeated disc.
    matrix : int
        Excitation grid size, square. How finely the profile is specified, and
        so how far out the trajectory reaches.
    selective_size : float or sequence of float, optional
        Diameter (m) of the excited disc. Half the field of view by default.
    target : numpy.ndarray, optional
        Complex desired profile on the ``(matrix, matrix)`` grid, instead of a
        disc.
    n_interleaves : int, optional
        Spiral arms to play. The default covers excitation k-space at Nyquist;
        fewer is a proportionally shorter pulse and a repeated disc that moves
        proportionally closer.
    axes : sequence of str, optional
        The two gradient channels the trajectory runs on.
    use : str, optional
        What the pulse is for; the trajectory core reads it.

    Attributes
    ----------
    rf : RfEvent
        The pulse.
    gradients : tuple of GradEvent
        One per axis, played in the pulse's own block.
    rephasers : tuple of TrapEvent
        Empty for the spiral trajectory; the attribute is there because a
        different path would need one.
    self_refocused : bool
        Whether the trajectory came back to the origin on its own.
    duration_s : float
        Pulse length (s), which the trajectory fixed.

    Raises
    ------
    ValueError
        If a size is out of range, ``target`` does not match ``matrix``, or the
        requested profile has no response on the trajectory.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> pencil = design.SpatialSelective2DExcitation(
    ...     system, 30.0, fov=0.256, matrix=32, selective_size=0.04
    ... )
    >>> len(pencil.blocks), [gradient.channel for gradient in pencil.gradients]
    (1, ['x', 'y'])

    The spiral closes on itself, so nothing has to unwind it:

    >>> pencil.self_refocused, pencil.rephasers
    (True, ())
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float,
        *,
        fov: float,
        matrix: int,
        selective_size: object = None,
        target: object = None,
        n_interleaves: int | None = None,
        axes: tuple[str, ...] = ("x", "y"),
        use: str = "excitation",
    ) -> None:
        rf, gradients, rephasers = pp.make_2d_selective_pulse(
            np.deg2rad(flip_angle_deg),
            fov=fov,
            matrix=matrix,
            selective_size=selective_size,
            target=target,
            n_interleaves=n_interleaves,
            axes=axes,
            system=system,
            use=use,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf, *gradients)
        if rephasers:
            self.seq.add_block(*rephasers)

        self.register(gradients=gradients, rephasers=rephasers)
        self.center = rf_reference(rf)
        self.self_refocused = not rephasers
        self.duration_s = pp.calc_duration(rf, *gradients)
