"""Fat saturation, optionally confined to a band and pointed where asked."""

from __future__ import annotations

__all__ = ["FatSaturation"]

from typing import Any

import numpy as np

from ... import pypulseq as pp
from ..excitation._base import RfModule, rf_reference
from ._common import AXES, spoiler_gradients

#: Where the main fat resonance sits relative to water. Negative: fat precesses
#: more slowly, so it is downfield of water in frequency.
FAT_SHIFT_PPM = -3.45

#: The transform exemptions a placed module carries. Its pulse already points
#: where it was told to, so nothing downstream may point it again.
FOV_EXEMPT_FLAGS = ("NOPOS", "NOROT")


class FatSaturation(RfModule):
    """A fat-selective pulse and the spoiler that discards what it tipped.

    Saturation, not suppression by inversion: the pulse turns fat past the
    transverse plane and the spoiler destroys it, so the imaging excitation
    that follows sees water alone. Past the plane, not onto it -- 110 degrees
    by default -- because overshooting is what makes the residual longitudinal
    magnetisation insensitive to a transmit field that falls short.

    The pulse is SLR, designed for the ``"sat"`` profile: the spectral
    passband is stated as a bandwidth and a ripple rather than discovered from
    a window function, which is what keeps the stopband off the water peak
    250 Hz away.

    **Pointing it somewhere.** With ``thickness_m`` the pulse becomes a band
    rather than the whole volume, and ``position_mm`` and ``orientation`` say
    where that band goes. They are applied here, at design time, through
    :class:`~pulserver.pypulseq.TransformFOV` -- and the module then declares
    ``NOPOS`` and ``NOROT`` on its first block, so a transform applied to the
    finished scan moves the imaging volume and leaves this band where it was
    put. The flags are cleared on the last block, because Pulseq labels are
    sticky and an uncleared exemption would follow the rest of the sequence.

    A band that is both fat-selective and slice-selective is displaced along
    its own selection axis by the chemical shift, by ``freq_offset_ppm`` over
    the selection gradient's amplitude. That is a property of the pulse, not an
    error, and it is why a saturation band is usually made generously thick.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    freq_offset_ppm : float, optional
        Fat offset from water (ppm). The default is -3.5 ppm.
    flip_angle_deg : float, optional
        Saturation flip angle (degrees).
    bandwidth_hz : float, optional
        Spectral passband (Hz). With ``time_bw_product`` it fixes the pulse
        duration, which is ``time_bw_product / bandwidth_hz``.
    time_bw_product : float, optional
        Time-bandwidth product.
    thickness_m : float, optional
        Band thickness (m). ``None`` saturates the whole transmit volume,
        which is the usual fat saturation and the only form that needs no
        gradient.
    axis : {'z', 'x', 'y'}, optional
        Band normal, before ``orientation`` turns it.
    position_mm : sequence of float, optional
        ``(dx, dy, dz)`` offset of the band from the isocentre, in the logical
        frame, in millimetres. Needs ``thickness_m``.
    orientation : array_like or scipy.spatial.transform.Rotation, optional
        A ``(3, 3)`` matrix, or a rotation, taking the logical frame to the one
        the band should sit in. Needs ``thickness_m``.
    use_rotation_extension : bool, optional
        Carry the orientation as a ``ROTATIONS`` extension rather than baking
        it into new waveforms. Only ``True`` is implemented; see
        :class:`~pulserver.pypulseq.TransformFOV`.
    spoiling_cycles : float, optional
        Cycles of dephasing each spoiler axis winds across ``voxel_size_m``.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m).

    Attributes
    ----------
    rf_prep : RfEvent
        The saturation pulse, as placed.
    gz : GradEvent
        Its selection gradient. Only when ``thickness_m`` was given, and named
        for the logical axis whatever ``axis`` was.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        The three-axis spoiler. Three axes rather than one because what is
        being destroyed is a full excitation, not a residue.
    prep_labels : list of LabelSetEvent
        ``NOPOS`` and ``NOROT``, set on the first block.
    reset_labels : list of LabelSetEvent
        The same two cleared, on the last.
    freq_offset_hz : float
        The fat offset actually used (Hz).

    Raises
    ------
    ValueError
        If a bandwidth, thickness or voxel size is not positive, ``axis`` is
        not a gradient channel, the spoiler was asked for zero cycles, or a
        position or orientation was given for a band that has no thickness.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> fatsat = design.FatSaturation(pp.Opts(B0=3.0))
    >>> len(fatsat.blocks), round(fatsat.freq_offset_hz)
    (2, -441)

    The exemption is on the pulse's own block, alongside it rather than in a
    block of its own:

    >>> [event.type for event in fatsat.blocks[0]]
    ['rf', 'labelset', 'labelset']

    A band, tilted and offset, that a later FOV transform will leave alone::

        from scipy.spatial.transform import Rotation

        band = design.FatSaturation(
            system, thickness_m=0.08, position_mm=(0.0, 0.0, 25.0),
            orientation=Rotation.from_euler("y", 30, degrees=True),
        )
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        freq_offset_ppm: float | None = None,
        flip_angle_deg: float = 110.0,
        bandwidth_hz: float = 250.0,
        time_bw_product: float = 2.0,
        thickness_m: float | None = None,
        axis: str = "z",
        position_mm: Any = None,
        orientation: Any = None,
        use_rotation_extension: bool = True,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
    ) -> None:
        if bandwidth_hz <= 0 or time_bw_product <= 0:
            raise ValueError("bandwidth_hz and time_bw_product must be positive")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if spoiling_cycles <= 0:
            raise ValueError(
                "a fat saturation has to be spoiled: without it the fat the pulse just "
                "tipped over stays in the transverse plane and is read with the water"
            )
        if axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
        if thickness_m is not None and thickness_m <= 0:
            raise ValueError("thickness_m must be positive")
        if thickness_m is None and (position_mm is not None or orientation is not None):
            raise ValueError(
                "a saturation without a thickness covers the whole transmit volume, so it "
                "has nowhere to be put; give thickness_m to make it a band"
            )

        if freq_offset_hz is None:
            freq_offset_hz = FAT_SHIFT_PPM * 1e-6 * system.B0 * system.gamma

        designed = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=time_bw_product / bandwidth_hz,
            slice_thickness=0.0 if thickness_m is None else thickness_m,
            time_bw_product=time_bw_product,
            pulse_type="sat",
            freq_ppm=freq_offset_ppm,
            return_gz=thickness_m is not None,
            use="saturation",
            system=system,
        )
        rf_prep, gz = (designed[0], designed[1]) if thickness_m is not None else (designed, None)
        if gz is not None:
            gz.channel = axis

        gx_spoil, gy_spoil, gz_spoil = spoiler_gradients(system, spoiling_cycles, voxel_size_m)

        designed_seq = pp.Sequence(system)
        if gz is not None:
            designed_seq.add_block(rf_prep, gz)
        else:
            designed_seq.add_block(rf_prep)
        designed_seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        if position_mm is not None or orientation is not None:
            # Over the pulse block only. The spoiler has no orientation to get
            # right -- it dephases the same however it is turned -- and turning
            # it would mix three lobes each already solved against the full
            # slew limit, so the rotated sum would exceed it.
            designed_seq = pp.TransformFOV(
                rotation=_as_matrix(orientation),
                translation=None if position_mm is None else tuple(map(float, position_mm)),
                use_rotation_extension=use_rotation_extension,
                system=system,
            ).apply_to_sequence(
                designed_seq, time_range=[0.0, 0.5 * designed_seq.block_durations[0]]
            )

        # Rebuilt rather than edited in place: the placement rewrote the pulse,
        # so what this module plays -- and therefore publishes -- is the
        # transformed event, and the flags go on with it. The exemption belongs
        # on the block that carries the pulse, and a block of its own would be
        # a block of dead time.
        placed = [
            designed_seq.get_block(index) for index in range(1, designed_seq.num_blocks + 1)
        ]
        rf_prep, gz = placed[0].rf, placed[0].gz
        gx_spoil, gy_spoil, gz_spoil = placed[-1].gx, placed[-1].gy, placed[-1].gz

        prep_labels = [
            pp.make_label(type="SET", label=name, value=1) for name in FOV_EXEMPT_FLAGS
        ]
        # Cleared on the way out: Pulseq labels are sticky, so an exemption left
        # set would go on exempting every block after this module.
        reset_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in FOV_EXEMPT_FLAGS
        ]

        self.seq = pp.Sequence(system)
        for index, block in enumerate(placed):
            self.seq.add_block(
                *pp.block_to_events(block),
                *(prep_labels if index == 0 else ()),
                *(reset_labels if index == len(placed) - 1 else ()),
            )

        self.center = rf_reference(rf_prep)
        self.freq_offset_hz = float(1e-6 * system.gamma * system.B0 * freq_offset_ppm)


def _as_matrix(orientation: Any) -> np.ndarray | None:
    """An orientation as the ``(3, 3)`` matrix ``TransformFOV`` takes."""
    if orientation is None:
        return None
    if hasattr(orientation, "as_matrix"):
        return orientation.as_matrix()
    return np.asarray(orientation, dtype=float)
