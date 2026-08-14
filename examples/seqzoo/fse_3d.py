"""3D Cartesian fast spin echo, slab-selective, with selectable view ordering.

One long CPMG train per excitation over a ``(ky, kz)`` grid --
:class:`design.FseReadout3D` -- with the two decisions a long train forces
made explicit, after Busse et al. 2008 (MRM 60:640), followed loosely:

**View ordering** maps the train's signal modulation into k-space. The
*coherent* orderings sort views so successive echoes stay close in k-space
and the centre lands at the echo whose time is the requested effective TE:
``linear`` (sorted along ky, echo groups rolled onto the target echo, kz
sorted within a group), ``radial`` (centre-out by k-space radius, effective
TE at the first echo), and ``radial_adaptive`` (by radius, but echo groups
assigned outward from the target echo, so the centre is late without a
seam). The *incoherent* ordering, ``shuffling``, scatters views across
echoes for a subspace reconstruction -- pair it with
:mod:`pulserver.reczoo.subspace_basis`. Every acquisition carries its echo
index as ``ECO``, which is what any of them reconstruct from.

**Refocusing flip modulation** trades signal for blur over long trains. The
optional variable train is a TRAPS-style piecewise ramp in flip space --
``alpha_max`` at the first echo, down to ``alpha_min``, back up to
``alpha_center`` at the effective-TE echo, and on to ``alpha_max`` by the
train's end -- the shape Busse's prescribed-envelope method produces,
without the EPG inversion. The played angles are written into the
``RefocusingFlipAngles`` definition.

``main`` returns the :class:`pulserver.pypulseq.Sequence`; ``PLUGIN`` is the
same sequence behind the scanner protocol contract, and running this module
as a script writes a ``.seq`` from the same controls.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver import (
    Description,
    DropdownFloatParam,
    DropdownIntParam,
    OffFloatParam,
    SequencePlugin,
    TEPreset,
    TRPreset,
    TypeinFloatParam,
    UIParam,
    dict_to_protocol,
    main_kwargs,
    params,
    protocol_to_dict,
    run_cli,
    write_sequence,
)

_PULSE_DURATION = 3e-3
_TIME_BW_PRODUCT = 4.0

#: The selectable view orderings; see the module docstring.
ORDERINGS = ("linear", "radial", "radial_adaptive", "shuffling")


def traps_flip_schedule(
    etl: int,
    n_center: int,
    *,
    alpha_min_deg: float = 60.0,
    alpha_center_deg: float = 100.0,
    alpha_max_deg: float = 160.0,
    n_down: int = 6,
) -> np.ndarray:
    """A TRAPS-style refocusing flip train, in degrees.

    Piecewise-linear in flip space between the four control points Busse's
    method exposes: ``alpha_max`` at the first echo, ``alpha_min`` from echo
    ``n_down``, ``alpha_center`` at the effective-TE echo, and ``alpha_max``
    again at the train's end. Loose by design: the reference derives the
    ramps from a prescribed signal envelope through EPG, and this keeps its
    control points and its shape without the inversion.

    Parameters
    ----------
    etl : int
        Echo train length.
    n_center : int
        Zero-based echo index of the k-space centre.
    alpha_min_deg : float, optional
        Floor of the ramp-down, reached at echo ``n_down``. Default is 60.
    alpha_center_deg : float, optional
        Flip at the centre-of-k-space echo. Default is 100.
    alpha_max_deg : float, optional
        Flip at the first and last echoes. Default is 160.
    n_down : int, optional
        Echo (zero-based) where the initial ramp-down bottoms out. Default
        is 6.

    Returns
    -------
    numpy.ndarray
        One flip per echo, degrees.
    """
    if etl < 1:
        raise ValueError("etl must be >= 1")
    control_echoes = [0]
    control_flips = [alpha_max_deg]
    bottom = min(max(1, n_down), max(1, etl - 1))
    if bottom > 0 and bottom < etl:
        control_echoes.append(bottom)
        control_flips.append(alpha_min_deg)
    if n_center > bottom:
        control_echoes.append(n_center)
        control_flips.append(alpha_center_deg)
    if etl - 1 > max(control_echoes):
        control_echoes.append(etl - 1)
        control_flips.append(alpha_max_deg)
    return np.interp(np.arange(etl), control_echoes, control_flips)


def order_views(
    views: list[tuple[int, int]],
    etl: int,
    n_center: int,
    ordering: str,
    grid: tuple[int, int],
    *,
    seed: int = 0,
) -> list[list[tuple[int, int] | None]]:
    """Deal ``(line, partition)`` views into trains, one view per echo.

    Parameters
    ----------
    views : list of tuple
        The views the sampling plan asks for.
    etl : int
        Echo train length.
    n_center : int
        Zero-based echo index the k-space centre should be acquired at.
    ordering : str
        One of :data:`ORDERINGS`.
    grid : tuple of int
        ``(n_y, n_z)``, locating the centre of k-space.
    seed : int, optional
        Seed of the shuffling permutation, so a scan is reproducible.

    Returns
    -------
    list of list
        One list per train, indexed by echo; ``None`` pads echoes with
        nothing left to encode.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
    n_y, n_z = grid
    n_trains = -(-len(views) // etl)

    def padded(ordered: list) -> list:
        return ordered + [None] * (etl * n_trains - len(ordered))

    if ordering == "shuffling":
        permuted = list(views)
        np.random.default_rng(seed).shuffle(permuted)
        flat = padded(permuted)
        return [flat[train * etl : (train + 1) * etl] for train in range(n_trains)]

    def radius(view):
        line, partition = view
        return ((line - n_y / 2) / n_y) ** 2 + ((partition - n_z / 2) / n_z) ** 2

    def angle(view):
        line, partition = view
        return np.arctan2((partition - n_z / 2) / n_z, (line - n_y / 2) / n_y)

    if ordering == "linear":
        ranked = sorted(views, key=lambda view: (view[0], view[1]))
        groups = [
            ranked[group * n_trains : (group + 1) * n_trains] for group in range(etl)
        ]
        # Roll so the group holding the centre plays at the target echo; the
        # wrap seam is the price of a late effective TE, as in 2D.
        centre = min(views, key=radius)
        holds_centre = next(
            index for index, group in enumerate(groups) if centre in group
        )
        roll = (holds_centre - n_center) % etl
        groups = groups[roll:] + groups[:roll]
        echo_of_group = list(range(etl))
    else:
        ranked = sorted(views, key=radius)
        groups = [
            ranked[group * n_trains : (group + 1) * n_trains] for group in range(etl)
        ]
        if ordering == "radial":
            echo_of_group = list(range(etl))
        else:  # radial_adaptive: nearest radii at the target echo, outward
            echo_of_group = sorted(range(etl), key=lambda echo: abs(echo - n_center))

    trains: list[list[tuple[int, int] | None]] = [
        [None] * etl for _ in range(n_trains)
    ]
    for rank, group in enumerate(groups):
        echo = echo_of_group[rank]
        # Within a group, sort by the orthogonal coordinate (linear) or by
        # angle (radial), so successive echoes of one train stay neighbours.
        keyed = sorted(
            (view for view in group if view is not None),
            key=(lambda view: view[1]) if ordering == "linear" else angle,
        )
        for train, view in enumerate(keyed):
            trains[train][echo] = view
    return trains


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "fse_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    etl: int = 32,
    te: float | None = 100e-3,
    tr: float | None = 1000e-3,
    ordering: str = "radial_adaptive",
    variable_flip: bool = True,
    alpha_min_deg: float = 60.0,
    alpha_center_deg: float = 100.0,
    alpha_max_deg: float = 160.0,
    readout_bandwidth_hz: float = 250e3,
    acceleration: int = 1,
    acceleration_z: int = 1,
    n_acs: int = 24,
    n_acs_z: int = 16,
    n_dummy: int = 0,
    shuffle_seed: int = 0,
    n_gain_calibration_readouts: int = 1,
    crusher_cycles: float = 4.0,
    readout_crusher_cycles: float = 0.0,
) -> pp.Sequence:
    """Create a 3D Cartesian fast spin-echo sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'fse_3d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float or tuple of float, optional
        In-plane field of view in meters, (fov_x, fov_y) if a tuple. Default
        is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters along the logical
        readout, phase and slab axes. Default is (0.0, 0.0, 0.0).
    n_x : int, optional
        Number of readout samples. Default is 128.
    n_y : int, optional
        Number of phase encoding steps. Default is 128.
    n_z : int, optional
        Number of partition encoding steps. Default is 64.
    slab_thickness : float, optional
        Excited slab thickness in meters, also the field of view along z.
        Default is 128e-3.
    etl : int, optional
        Echo train length: views per excitation. Default is 32.
    te : float or None, optional
        Effective echo time in seconds, rounded onto the echo grid. Ignored
        by the ``radial`` ordering (centre-out puts it at the first echo)
        and meaningless under ``shuffling``. None is the first echo.
        Default is 100e-3.
    tr : float or None, optional
        Repetition time in seconds, one per train. Default is 1000e-3.
    ordering : str, optional
        One of ``linear``, ``radial``, ``radial_adaptive``, ``shuffling``.
        Default is 'radial_adaptive'.
    variable_flip : bool, optional
        Play the TRAPS-style refocusing train of
        :func:`traps_flip_schedule` rather than constant 180s. Default is
        True.
    alpha_min_deg, alpha_center_deg, alpha_max_deg : float, optional
        The variable train's control points. Defaults are 60, 100 and 160.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    acceleration : int, optional
        Uniform phase-encode undersampling factor along y. Default is 1.
    acceleration_z : int, optional
        Uniform partition-encode undersampling factor along z. Default is 1.
    n_acs : int, optional
        Autocalibration extent along y, in lines. Default is 24.
    n_acs_z : int, optional
        Autocalibration extent along z, in partitions. Default is 16.
    n_dummy : int, optional
        Trains played without acquiring before the first. Default is 0.
    shuffle_seed : int, optional
        Seed of the shuffling permutation. Default is 0.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.
    crusher_cycles : float, optional
        Cycles of dephasing each crusher beside every refocusing pulse
        winds. Default is 4.0.
    readout_crusher_cycles : float, optional
        Read-axis crushing each side of every acquisition. Default is 0.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The FSE sequence object.
    """
    system = pp.Opts() if system is None else system

    kernel = FSE3DKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=slab_thickness,
        etl=etl,
        te=te,
        tr=tr,
        ordering=ordering,
        variable_flip=variable_flip,
        alpha_min_deg=alpha_min_deg,
        alpha_center_deg=alpha_center_deg,
        alpha_max_deg=alpha_max_deg,
        readout_bandwidth_hz=readout_bandwidth_hz,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        n_acs=n_acs,
        n_acs_z=n_acs_z,
        n_dummy=n_dummy,
        shuffle_seed=shuffle_seed,
        crusher_cycles=crusher_cycles,
        readout_crusher_cycles=readout_crusher_cycles,
    )
    fse = kernel.readout
    timing = kernel.timing
    fov_x, fov_y = kernel.fov
    flips = kernel.flip_schedule_deg

    acs_y = range(max(0, n_y // 2 - n_acs // 2), min(n_y, n_y // 2 + -(-n_acs // 2)))
    acs_z = range(
        max(0, n_z // 2 - n_acs_z // 2), min(n_z, n_z // 2 + -(-n_acs_z // 2))
    )

    seq = pp.Sequence(system)
    ima_label, seg_label, eco_label = fse.adc_labels
    seg_label.value = 0
    nominal = fse.rf_ref.amplitude

    def train(views, acquire: bool, mark=None) -> None:
        """Play one train, acquiring or not."""
        seq.add_block(fse.rf, fse.gz, *([mark] if mark is not None else []))
        seq.add_block(fse.gx_pre)
        for echo in range(etl):
            view = views[echo]
            fse.rf_ref.amplitude = nominal * np.deg2rad(flips[echo]) / np.pi
            seq.add_block(fse.rf_ref, fse.gz_ref)
            if echo == 0 and fse.esp_first > fse.esp:
                seq.add_block(fse.wait_esp1)
            if view is None:
                ky, kz = 0.0, 0.0
            else:
                line, partition = view
                ky = (line - n_y / 2) / (n_y / 2)
                kz = (partition - n_z / 2) / (n_z / 2)
            seq.add_block(
                fse.gx_bridge_pre,
                pp.scale_grad(fse.gy_pre, ky),
                pp.scale_grad(fse.gz_pre, kz),
            )
            if acquire and view is not None:
                line, partition = view
                ima_label.value = int(line in acs_y and partition in acs_z)
                eco_label.value = echo
                seq.add_block(fse.gx, fse.adc, ima_label, seg_label, eco_label)
            else:
                seq.add_block(fse.gx)
            seq.add_block(
                fse.gx_bridge_post,
                pp.scale_grad(fse.gy_rew, ky),
                pp.scale_grad(fse.gz_rew, kz),
            )
        if timing.wait_tr is not None:
            seq.add_block(timing.wait_tr)
        fse.rf_ref.amplitude = nominal

    for i_dummy in range(n_dummy):
        train(
            [None] * etl,
            acquire=False,
            mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
        )

    clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
    for views in kernel.trains:
        train(views, acquire=True, mark=clear_once)
        clear_once = None

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset), system=system
    ).apply_to_sequence(seq, in_place=True)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_z])
    seq.set_definition(key="Name", value="fse_3d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(key="EchoSpacing", value=kernel.echo_spacing)
    seq.set_definition(key="EchoTrainLength", value=etl)
    seq.set_definition(key="ViewOrdering", value=ordering)
    seq.set_definition(key="RefocusingFlipAngles", value=list(flips))
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    seq.auto_label()

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


def FSE3DKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    etl: int = 32,
    te: float | None = 100e-3,
    tr: float | None = 1000e-3,
    ordering: str = "radial_adaptive",
    variable_flip: bool = True,
    alpha_min_deg: float = 60.0,
    alpha_center_deg: float = 100.0,
    alpha_max_deg: float = 160.0,
    readout_bandwidth_hz: float = 250e3,
    acceleration: int = 1,
    acceleration_z: int = 1,
    n_acs: int = 24,
    n_acs_z: int = 16,
    n_dummy: int = 0,
    shuffle_seed: int = 0,
    crusher_cycles: float = 4.0,
    readout_crusher_cycles: float = 0.0,
) -> SimpleNamespace:
    """Design the train, its flip schedule, and the view order that fills it.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_z, slab_thickness, etl, te, tr, ordering, \
variable_flip, alpha_min_deg, alpha_center_deg, alpha_max_deg, \
readout_bandwidth_hz, acceleration, acceleration_z, n_acs, n_acs_z, \
n_dummy, shuffle_seed, crusher_cycles, readout_crusher_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``refocusing``, ``readout``, ``timing``, ``fov``,
        ``trains`` (one view list per train, indexed by echo),
        ``n_center``, ``flip_schedule_deg``, ``echo_spacing``,
        ``echo_time``, ``repetition_time``, ``bandwidth_hz`` and
        ``duration``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = design.SpatialSelectiveExcitation(
        system,
        90.0,
        slab_thickness,
        duration_s=_PULSE_DURATION,
        time_bw_product=_TIME_BW_PRODUCT,
        is_slab=True,
    )
    refocusing = design.SpatialSelectiveRefocusing(
        system,
        slab_thickness,
        duration_s=_PULSE_DURATION,
        time_bw_product=_TIME_BW_PRODUCT,
        spoiling_cycles=crusher_cycles,
    )

    fse = design.FseReadout3D(
        system,
        excitation.rf,
        excitation.gz,
        rf_ref=refocusing.rf_ref,
        gz_ref=refocusing.gz,
        fov=(fov_x, fov_y, slab_thickness),
        matrix=(n_x, n_y, n_z),
        etl=etl,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=readout_crusher_cycles,
        labels=("IMA", "SEG", "ECO"),
    )
    esp = fse.esp

    if ordering == "radial":
        n_center = 0
    elif te is None:
        n_center = 0
    else:
        n_center = int(np.clip(round(te / esp) - 1, 0, etl - 1))
    echo_time = float(fse.echo_times[n_center])

    length = fse.seq.duration()[0]
    wait_tr = None
    if tr is not None:
        if tr < length - 1e-9:
            raise ValueError(
                f"TR {tr * 1e3:.1f} ms is shorter than one train takes "
                f"({length * 1e3:.1f} ms)"
            )
        pad = pp.round_to_raster(tr - length, system.block_duration_raster)
        if pad > 0:
            wait_tr = pp.make_delay(pad)
            length += pad
    timing = SimpleNamespace(wait_tr=wait_tr, length=length)

    flips = (
        traps_flip_schedule(
            etl,
            n_center,
            alpha_min_deg=alpha_min_deg,
            alpha_center_deg=alpha_center_deg,
            alpha_max_deg=alpha_max_deg,
        )
        if variable_flip
        else np.full(etl, 180.0)
    )

    sampled_lines = pp.calc_sampled_lines(n_y, acceleration, n_acs)
    sampled_partitions = pp.calc_sampled_lines(n_z, acceleration_z, n_acs_z)
    views = [
        (line, partition)
        for partition in sampled_partitions
        for line in sampled_lines
    ]
    trains = order_views(
        views, etl, n_center, ordering, (n_y, n_z), seed=shuffle_seed
    )

    duration = (n_dummy + len(trains)) * timing.length

    return SimpleNamespace(
        excitation=excitation,
        refocusing=refocusing,
        readout=fse,
        timing=timing,
        fov=(fov_x, fov_y),
        trains=trains,
        n_center=n_center,
        flip_schedule_deg=flips,
        echo_spacing=esp,
        echo_time=echo_time,
        repetition_time=timing.length,
        bandwidth_hz=fse.bandwidth_hz,
        duration=duration,
    )


class Fse3D(SequencePlugin):
    """The 3D fast spin echo behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=100.0,
                    min=8.0,
                    max=600.0,
                    incr=1.0,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 60.0, 100.0, 150.0, 250.0],
                ),
                UIParam.TR: DropdownFloatParam(
                    value=1000.0,
                    min=100.0,
                    max=5000.0,
                    incr=10.0,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 600.0, 1000.0, 1500.0, 2500.0],
                ),
                UIParam.FOV: DropdownFloatParam(
                    value=220.0,
                    min=80.0,
                    max=500.0,
                    incr=1.0,
                    unit="mm",
                    options=[180.0, 220.0, 280.0, 340.0, 500.0],
                ),
                UIParam.PHASE_FOV: DropdownFloatParam(
                    value=220.0,
                    min=80.0,
                    max=500.0,
                    incr=1.0,
                    unit="mm",
                    options=[180.0, 220.0, 280.0, 340.0, 500.0],
                ),
                UIParam.SLICE_THICKNESS: DropdownFloatParam(
                    value=2.0,
                    min=0.5,
                    max=10.0,
                    incr=0.1,
                    unit="mm",
                    options=[1.0, 1.5, 2.0, 3.0, 5.0],
                ),
                UIParam.NX: DropdownIntParam(
                    value=128,
                    min=16,
                    max=512,
                    incr=1,
                    options=[64, 128, 192, 256, 384],
                ),
                UIParam.NY: DropdownIntParam(
                    value=128,
                    min=16,
                    max=512,
                    incr=1,
                    options=[64, 128, 192, 256, 384],
                ),
                UIParam.NSLICES: DropdownIntParam(
                    value=64,
                    min=8,
                    max=256,
                    incr=1,
                    options=[32, 64, 96, 128, 192],
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=250e3,
                    min=5e3,
                    max=500e3,
                    incr=100.0,
                    unit="Hz",
                ),
                UIParam.RY: TypeinFloatParam(
                    value=1.0,
                    min=1.0,
                    max=8.0,
                    incr=1.0,
                    unit="",
                ),
                UIParam.FOV_OFFSET_X: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Y: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Z: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.user_name(0): Description(text="ACS lines (y)"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=24.0, min=0.0, max=512.0, incr=1.0, unit="lines"
                ),
                UIParam.user_name(1): Description(text="Echo train length"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=32.0, min=1.0, max=256.0, incr=1.0, unit=""
                ),
                UIParam.user_name(2): Description(
                    text="Ordering 0=lin 1=rad 2=adaptive 3=shuffle"
                ),
                UIParam.user_value(2): TypeinFloatParam(
                    value=2.0, min=0.0, max=3.0, incr=1.0, unit=""
                ),
                UIParam.user_name(3): Description(text="Variable flip train"),
                UIParam.user_value(3): TypeinFloatParam(
                    value=1.0, min=0.0, max=1.0, incr=1.0, unit=""
                ),
                UIParam.user_name(4): Description(text="Acceleration (z)"),
                UIParam.user_value(4): TypeinFloatParam(
                    value=1.0, min=1.0, max=8.0, incr=1.0, unit=""
                ),
                UIParam.user_name(5): Description(text="ACS partitions (z)"),
                UIParam.user_value(5): TypeinFloatParam(
                    value=16.0, min=0.0, max=256.0, incr=1.0, unit="lines"
                ),
                UIParam.user_name(6): Description(text="Min/centre flips [deg]"),
                UIParam.user_value(6): TypeinFloatParam(
                    value=60.0, min=10.0, max=180.0, incr=1.0, unit="deg"
                ),
                UIParam.user_name(7): Description(text="Centre flip [deg]"),
                UIParam.user_value(7): TypeinFloatParam(
                    value=100.0, min=10.0, max=180.0, incr=1.0, unit="deg"
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        kwargs = _main_kwargs(system, protocol)
        try:
            kernel = FSE3DKernel(
                system,
                **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s over {len(kernel.trains)} trains, "
                f"TEeff = {kernel.echo_time * 1e3:.1f} ms at echo "
                f"{kernel.n_center + 1}, ESP = {kernel.echo_spacing * 1e3:.2f} ms"
            ),
        }

    def make_sequence(
        self,
        system: pp.Opts,
        protocol: dict[str, dict],
        output_path: str,
        *,
        offline: bool = False,
    ) -> None:
        """Build the sequence and write it to ``output_path``."""
        seq = main(**_main_kwargs(system, protocol))
        write_sequence(seq, output_path, offline=offline)


_KERNEL_ARGUMENTS = frozenset(
    (
        "fov",
        "n_x",
        "n_y",
        "n_z",
        "slab_thickness",
        "etl",
        "te",
        "tr",
        "ordering",
        "variable_flip",
        "alpha_min_deg",
        "alpha_center_deg",
        "alpha_max_deg",
        "readout_bandwidth_hz",
        "acceleration",
        "acceleration_z",
        "n_acs",
        "n_acs_z",
        "n_dummy",
        "shuffle_seed",
        "crusher_cycles",
        "readout_crusher_cycles",
    )
)


def _main_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
    """The prescribed quantities, plus this sequence's own user slots."""
    prot = dict_to_protocol(protocol)
    n_z = params.param_int(prot, UIParam.NSLICES)
    partition_thickness = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    return main_kwargs(
        main,
        system,
        protocol,
        n_z=n_z,
        slab_thickness=n_z * partition_thickness,
        etl=max(1, round(params.user_float(prot, 1, 32.0))),
        ordering=ORDERINGS[int(np.clip(round(params.user_float(prot, 2, 2.0)), 0, 3))],
        variable_flip=bool(round(params.user_float(prot, 3, 1.0))),
        n_acs=params.acs_lines_from_protocol(prot, params.param_int(prot, UIParam.NY), 0),
        acceleration_z=max(1, round(params.user_float(prot, 4, 1.0))),
        n_acs_z=max(0, round(params.user_float(prot, 5, 16.0))),
        alpha_min_deg=params.user_float(prot, 6, 60.0),
        alpha_center_deg=params.user_float(prot, 7, 100.0),
    )


PLUGIN = Fse3D()


def get_default_protocol(system):
    """Bridge entry point: the plugin's default protocol."""
    return PLUGIN.get_default_protocol(system)


def validate_protocol(system, protocol):
    """Bridge entry point: protocol feasibility and scan duration."""
    return PLUGIN.validate_protocol(system, protocol)


def make_sequence(system, protocol, output_path):
    """Bridge entry point: write the ``.seq`` file."""
    return PLUGIN.make_sequence(system, protocol, output_path)


_ARG_MAP = [
    ("--te-ms", UIParam.TE, float, "Effective echo time [ms], or a negative TEPreset"),
    ("--tr-ms", UIParam.TR, float, "Repetition time [ms], or a negative TRPreset"),
    ("--fov-mm", UIParam.FOV, float, "Readout FOV [mm]"),
    ("--phase-fov-mm", UIParam.PHASE_FOV, float, "Phase-encode FOV [mm]"),
    (
        "--partition-thickness-mm",
        UIParam.SLICE_THICKNESS,
        float,
        "Partition thickness [mm]; the slab is this times the partition count",
    ),
    ("--nx", UIParam.NX, int, "Readout matrix size"),
    ("--ny", UIParam.NY, int, "Phase-encode matrix size"),
    ("--nz", UIParam.NSLICES, int, "Partition count"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--ry", UIParam.RY, float, "Phase-encode undersampling factor"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    ("--offset-y-mm", UIParam.FOV_OFFSET_Y, float, "Volume offset along phase encode [mm]"),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slab [mm]"),
    ("--acs-lines", UIParam.user_value(0), float, "Number of ACS lines along y"),
    ("--etl", UIParam.user_value(1), float, "Echo train length"),
    (
        "--ordering",
        UIParam.user_value(2),
        float,
        "View ordering: 0 linear, 1 radial, 2 radial-adaptive, 3 shuffling",
    ),
    ("--constant-flip", UIParam.user_value(3), lambda value: 0.0 if float(value) else 1.0,
     "Pass 1 for constant 180s (0, the default value, keeps the variable train)"),
    ("--rz", UIParam.user_value(4), float, "Partition-encode undersampling factor"),
    ("--acs-partitions", UIParam.user_value(5), float, "Number of ACS partitions along z"),
    ("--alpha-min-deg", UIParam.user_value(6), float, "Variable train's floor flip [deg]"),
    ("--alpha-centre-deg", UIParam.user_value(7), float, "Variable train's centre flip [deg]"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate a 3D Cartesian fast spin-echo .seq offline.",
            default_output="fse_3d.seq",
        )
    )
