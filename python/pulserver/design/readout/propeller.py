"""PROPELLER readouts: a narrow EPI blade, turned about the centre of k-space."""

from __future__ import annotations

__all__ = ["PropellerReadout2D", "PropellerStackReadout"]

import math
from typing import Any

import numpy as np

from ... import pypulseq as pp
from .epi import _EpiReadout

_SCHEMES = ("uniform", "golden")


class _PropellerReadout(_EpiReadout):
    """One blade: a short EPI train across the centre of k-space, and the angles to turn it to.

    A blade is ``blade_width`` phase-encode lines wide and the full matrix
    long, straddling ``k = 0``, so every blade samples the centre and none of
    them needs placing: the loop turns the blade and plays it again::

        from scipy.spatial.transform import Rotation

        for angle in blade.blade_angles:
            rotation = pp.make_rotation(Rotation.from_euler("z", angle))
            seq.add_block(blade.rf, blade.gz)
            seq.add_block(blade.gx_pre, pp.scale_grad(blade.gy_pre, blade.blade_start),
                          rotation)
            for line in range(blade.etl):
                seq.add_block(blade.gx[line], blade.adc,
                              *present(blade.gy_blips[line]), rotation)
            seq.add_block(blade.gx_spoil, rotation)

    The gradients stay on their own axes and a ``ROTATIONS`` extension turns
    the block, which is what keeps one blade waveform serving every angle.

    Attributes
    ----------
    blade_angles : numpy.ndarray
        In-plane angles (rad), one per blade. Turn about ``z``.
    blade_start : float
        What to scale ``gy_pre`` by to put the blade astride ``k = 0``. It does
        not vary: a blade is centred by construction, so only the angle moves.
    blade_width : int
        Phase-encode lines in one blade.
    n_blades : int
        Blades in the set.

    Everything else -- ``gx``, ``gy_blips``, ``adc``, ``esp`` and the rest --
    is :class:`~pulserver.design.EpiReadout2D`'s, the blade being an EPI train.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        The pulse that opens the repetition.
    gz : GradEvent, optional
        A selection gradient played in the same block as ``rf``.
    gz_reph : GradEvent, optional
        The rephaser that unwinds ``gz``.
    fov : float
        In-plane field of view (m), isotropic: the blade turns, so it cannot be
        anything else.
    matrix : int
        In-plane matrix size, isotropic.
    blade_width : int, optional
        Phase-encode lines per blade. Narrow blades turn a scan into many short
        trains -- less distortion per blade, more blades to cover the disc.
    n_blades : int, optional
        Blades in the set. The default is the smallest that samples the edge of
        the disc at Nyquist, ``ceil(pi * matrix / (2 * blade_width))``.
    scheme : {'uniform', 'golden'}, optional
        How the default angles are spread. ``'uniform'`` divides half a turn
        between the blades, which is exact for the whole set and only for the
        whole set. ``'golden'`` leaves any prefix of the set near-uniform,
        which is what a motion-corrected or abandoned scan needs.
    angles : array_like, optional
        In-plane angles (rad) to use instead of generating a set.
    fov_z, matrix_z : float, int
        Slab field of view (m) and partitions. :class:`PropellerStackReadout`
        only, where they are required.

    Raises
    ------
    ValueError
        If a count is out of range, ``scheme`` is unknown, or ``angles`` is
        given alongside a count it would contradict.
    """

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        gz_reph: Any = None,
        *,
        fov: float,
        matrix: int,
        blade_width: int = 16,
        n_blades: int | None = None,
        scheme: str = "uniform",
        angles: Any = None,
        fov_z: float | None = None,
        matrix_z: int | None = None,
        **kwargs: Any,
    ) -> None:
        blade_width = int(blade_width)
        if blade_width < 1:
            raise ValueError("blade_width must be positive")
        if scheme not in _SCHEMES:
            raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}")

        if angles is None:
            if n_blades is None:
                # Adjacent blades have to meet at the rim of the disc, where
                # the gap between them grows with radius: half a turn divided
                # by the angle a blade subtends there.
                n_blades = max(1, math.ceil(np.pi * int(matrix) / (2.0 * blade_width)))
            n_blades = int(n_blades)
            if n_blades < 1:
                raise ValueError("n_blades must be positive")
            blade_angles = (
                np.asarray(pp.calc_uniform_angles(n_blades, span=np.pi))
                if scheme == "uniform"
                else np.asarray(pp.calc_golden_angles(n_blades))
            )
        else:
            if n_blades is not None:
                raise ValueError(
                    "angles and n_blades are alternatives; pass the angles or ask for a count"
                )
            blade_angles = np.asarray(angles, dtype=float).ravel()
            if not len(blade_angles):
                raise ValueError("angles must hold at least one blade")

        # A blade straddles k = 0, so its lines run from -(w - 1) / 2 upward
        # and the prewinder is the same for every blade. Against a template
        # built at w / 2, that is the fraction below.
        blade_start = -(blade_width - 1) / blade_width

        ndim = self._ndim
        if ndim == 3:
            if fov_z is None or matrix_z is None:
                raise ValueError("a stack of propellers needs fov_z and matrix_z")
            shape: Any = (float(fov), float(fov), float(fov_z))
            counts: Any = (int(matrix), blade_width, int(matrix_z))
        else:
            shape, counts = (float(fov), float(fov)), (int(matrix), blade_width)

        super().init_module(
            system,
            rf,
            gz,
            gz_reph,
            fov=shape,
            matrix=counts,
            order=np.arange(blade_width),
            **kwargs,
        )
        self.register(blade_angles=blade_angles)
        self.blade_start = blade_start
        self.blade_width = blade_width
        self.n_blades = len(blade_angles)


class PropellerReadout2D(_PropellerReadout):
    """A 2D PROPELLER blade set: EPI blades turned about the centre of k-space.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> excitation = design.SpatialSelectiveExcitation(system, 60.0, 3e-3)
    >>> blade = design.PropellerReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     fov=0.22, matrix=128, blade_width=16,
    ... )
    >>> blade.etl, blade.n_blades
    (16, 13)

    Every blade is the same train, so the loop turns it rather than rebuilding
    it, and the angles divide half a turn:

    >>> float(round(blade.blade_angles[1] - blade.blade_angles[0], 6))
    0.241661
    """

    _ndim = 2


class PropellerStackReadout(_PropellerReadout):
    """A stack of 2D PROPELLER blade sets, partition-encoded along z.

    The blade turns about ``z``, which is the one axis the rotation leaves
    alone, so the partition encode rides through it untouched: scale ``gz_pre``
    per partition and turn the block per blade.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> slab = design.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
    >>> blade = design.PropellerStackReadout(
    ...     system, slab.rf, slab.gz, fov=0.22, matrix=128, blade_width=16,
    ...     fov_z=0.12, matrix_z=24,
    ... )
    >>> blade.n_blades, blade.gz_pre.channel
    (13, 'z')

    The partition does not move inside a blade, so nothing is blipped on z:

    >>> set(blade.gz_blips)
    {None}
    """

    _ndim = 3
