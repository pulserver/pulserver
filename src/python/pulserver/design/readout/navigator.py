"""Orthogonal spiral navigators, for tracking rigid motion during a scan."""

from __future__ import annotations

__all__ = ["SpiralNavigator"]

from typing import Any

from ... import pypulseq as pp
from .._module import SequenceModule
from ..excitation.selective import SpatialSelectiveExcitation
from .noncartesian import SpiralReadout2D

#: The planes a navigator reads, and the rotation that carries the designed
#: interleave onto each. The module is designed in x-y with its selection
#: gradient on z, which is the axial plane; the other two are that same
#: module turned a quarter turn, so one waveform serves all three. Axes follow
#: the patient convention: x left-right, y anterior-posterior, z
#: inferior-superior.
PLANES = ("axial", "sagittal", "coronal")

_TURNS = {
    "axial": None,
    "sagittal": ("y", 90.0),
    "coronal": ("x", 90.0),
}


class SpiralNavigator(SequenceModule):
    """Three orthogonal spiral navigators, for image-based motion tracking.

    One low-flip thick-slab excitation and one single-shot spiral arm are
    designed, then played three times under the quarter turns that carry them
    onto the axial, sagittal and coronal planes. Three planes are what make the
    pose observable: a single plane leaves through-plane translation and two
    rotations unconstrained.

    A navigator resolves rigid pose, not anatomy, so the in-plane resolution is
    deliberately coarse and the slab deliberately thick — a short readout is
    less sensitive to off-resonance, and the tracking runs on image content
    rather than on detail.

    The **flip angle is not a free parameter**. A navigator excites the same
    water the host sequence images, so every navigator costs the imaging volume
    some longitudinal magnetization. Raising the flip buys navigator SNR
    directly out of the host sequence's contrast, which is why the default is
    low and why a train of them belongs in dead time rather than anywhere else.

    Every acquisition is labelled ``NAV``, which is what puts the readouts in
    their own encoding space for a reconstruction to find.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float, optional
        In-plane field of view (m). Wide enough to contain the head.
    matrix : int, optional
        Acquired in-plane matrix. Coarse on purpose.
    thickness_m : float, optional
        Slab thickness (m).
    flip_angle_deg : float, optional
        Nominal flip angle (degrees). See above.
    duration_s : float, optional
        Excitation pulse duration (s).
    readout_bandwidth_hz : float, optional
        Requested sample spacing along the arm.
    navigator_tr : float, optional
        Repetition time of one three-plane navigator (s). ``None`` packs the
        planes back to back. A longer one leaves room for the reconstruction
        and the pose estimate to come back before the next navigator.
    spoiling_cycles : float, optional
        Spoiler moment at the end of each plane, in cycles across the slab.

    Attributes
    ----------
    excitation : SpatialSelectiveExcitation
        The pulse every plane is opened by.
    readout : SpiralReadout2D
        The designed interleave every plane plays.
    rotations : dict
        The rotation carried by each plane's blocks, keyed by plane name.
    plane_duration : float
        Length of one plane (s).

    Examples
    --------
    The three planes are one designed arm under three rotations, so the module
    holds one waveform however many planes it plays:

    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m",
    ...                  max_slew=150, slew_unit="T/m/s")
    >>> navigator = design.SpiralNavigator(system)
    >>> sorted(navigator.rotations)
    ['axial', 'coronal', 'sagittal']
    >>> navigator.readout.adc.num_samples > 0
    True
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        fov: float = 0.32,
        matrix: int = 32,
        thickness_m: float = 10e-3,
        flip_angle_deg: float = 8.0,
        duration_s: float = 1e-3,
        readout_bandwidth_hz: float = 250e3,
        navigator_tr: float | None = None,
        spoiling_cycles: float = 4.0,
        **kwargs: Any,
    ) -> None:
        if flip_angle_deg <= 0:
            raise ValueError("flip_angle_deg must be positive")
        if matrix < 8:
            raise ValueError("matrix must be at least 8 to resolve a pose")

        excitation = SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            thickness_m,
            duration_s,
            is_slab=False,
        )
        readout = SpiralReadout2D(
            system,
            excitation.rf,
            excitation.gz,
            excitation.gz_reph,
            fov=fov,
            matrix=matrix,
            design_interleaves=1,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            **kwargs,
        )

        self.seq = pp.Sequence(system)
        rotations = {
            plane: None if turn is None else pp.make_rotation(_turn_of(turn))
            for plane, turn in _TURNS.items()
        }
        plane_duration = readout.duration

        # A label is sequence state, not a property of the block that sets it,
        # so the flag has to be turned off again: an imaging readout that
        # followed a navigator would otherwise be delivered as navigator data.
        nav_on = pp.make_label(type="SET", label="NAV", value=1)
        nav_off = pp.make_label(type="SET", label="NAV", value=0)

        pad = _plane_padding(navigator_tr, plane_duration, system.block_duration_raster)
        for plane in PLANES:
            rotation = rotations[plane]
            turn = () if rotation is None else (rotation,)
            for block in readout.blocks:
                flag = (nav_on,) if _acquires(block) else ()
                self.seq.add_block(*block, *turn, *flag)
            if pad is not None:
                self.seq.add_block(pad)
        self.seq.add_block(pp.make_delay(system.block_duration_raster), nav_off)

        self.register(
            excitation=excitation,
            readout=readout,
            rotations=rotations,
            plane_duration=plane_duration,
            nav_on=nav_on,
            nav_off=nav_off,
        )

    def fit(
        self, window: float, requested: int | str = "auto", *, limit: int | None = None
    ) -> int:
        """How many navigators to play in a window of dead time.

        **Filling the window is not the goal.** The dead time a navigator train
        rides in is usually a longitudinal recovery, and a navigator excites the
        very magnetisation that recovery is restoring: at the default flip each
        one costs ``cos(8 deg) ** 3 = 0.97`` of Mz where its slabs cut the
        imaging volume, so a train that filled a recovery would take a quarter
        of it. ``limit`` is how a caller keeps the recovery a recovery.

        Parameters
        ----------
        window : float
            Dead time available (s).
        requested : int or str, optional
            ``"auto"`` takes as many as fit, up to ``limit``; an integer asks
            for exactly that many and is refused if they do not fit.
        limit : int, optional
            Ceiling on the ``"auto"`` count. Ignored for an explicit request,
            which is the caller saying it has already decided.

        Returns
        -------
        int
            Navigators to play.

        Raises
        ------
        ValueError
            If ``requested`` is negative, or asks for more than ``window``
            holds.
        """
        if requested == "auto":
            fits = max(int(window // self.duration), 0)
            return fits if limit is None else min(fits, int(limit))
        count = int(requested)
        if count < 0:
            raise ValueError("n_navigators must not be negative")
        if count * self.duration > window:
            raise ValueError(
                f"{count} navigators take {count * self.duration * 1e3:.1f} ms, "
                f"more than the {window * 1e3:.1f} ms of dead time available; "
                f"at most {int(window // self.duration)} fit"
            )
        return count


def _acquires(block: tuple) -> bool:
    """Whether ``block`` carries an acquisition window."""
    return any(getattr(event, "type", "") == "adc" for event in block)


def _turn_of(turn: tuple[str, float]) -> Any:
    """Return the rotation a plane is reached by."""
    from scipy.spatial.transform import Rotation

    axis, degrees = turn
    return Rotation.from_euler(axis, degrees, degrees=True)


def _plane_padding(
    navigator_tr: float | None, plane_duration: float, raster_s: float
) -> Any:
    """Return the delay that paces the planes, or None when they are packed."""
    if navigator_tr is None:
        return None
    pad = navigator_tr / len(PLANES) - plane_duration
    if pad < 0:
        raise ValueError(
            f"navigator_tr={navigator_tr * 1e3:.1f} ms is shorter than the "
            f"{len(PLANES)} planes it has to hold "
            f"({len(PLANES) * plane_duration * 1e3:.1f} ms)"
        )
    return pp.make_delay(pp.round_to_raster(pad, raster_s))
