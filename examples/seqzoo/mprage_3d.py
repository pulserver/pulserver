"""3D MPRAGE: inversion-prepared, segmented, RF-spoiled gradient echo.

One adiabatic inversion per segment, a wait that puts the segment's
centre-of-k-space view at the requested TI, a train of spoiled low-flip
:class:`design.LineReadout3D` repetitions, and a recovery wait that makes
every inversion-to-inversion interval the outer TR. The ``(ky, kz)`` views
are dealt into segments by the same orderings the 3D fast spin echo uses --
``linear``, ``radial``, ``radial_adaptive`` (coherent, TI-targeted) or
``shuffling`` (incoherent, for a subspace reconstruction) -- with each
acquisition carrying its within-segment index as ``ECO``.
:mod:`pulserver.reczoo.mprage_3d` reads the result back.

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
from pulserver.seqzoo.fse_3d import ORDERINGS, order_views

#: Per-plugin ceilings on the gradient and slew limits, in mT/m and T/m/s. The
#: sequence is held below the smaller of these and what the scanner reports, so
#: lowering them here -- on the scanner console, even -- reruns the whole script
#: under gentler gradients (for PNS headroom, acoustic comfort, eddy currents)
#: without touching anything else. Defaults sit above typical hardware, so they
#: cap nothing until you lower them.
MAX_GRAD = 80.0
MAX_SLEW = 200.0


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "mprage_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    flip_angle_deg: float = 9.0,
    ti: float = 900e-3,
    tr_outer: float = 2000e-3,
    views_per_segment: int = 64,
    ordering: str = "linear",
    te: float | None = None,
    tr: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    acceleration: int = 1,
    acceleration_z: int = 1,
    n_acs: int = 24,
    n_acs_z: int = 16,
    shuffle_seed: int = 0,
    n_gain_calibration_readouts: int = 1,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Create a 3D MPRAGE sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'mprage_3d.seq'.
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
    flip_angle_deg : float, optional
        Readout excitation flip angle in degrees. Default is 9.0.
    ti : float, optional
        Inversion time in seconds, from the inversion pulse's centre to the
        excitation of the segment's centre-of-k-space view. Default is
        900e-3.
    tr_outer : float, optional
        Inversion-to-inversion interval in seconds. Default is 2000e-3.
    views_per_segment : int, optional
        Views acquired per inversion. Default is 64.
    ordering : str, optional
        How views are dealt into segments: ``linear``, ``radial``,
        ``radial_adaptive`` or ``shuffling``. Default is 'linear'.
    te : float or None, optional
        Readout echo time in seconds. None is as short as possible. Default
        is None.
    tr : float or None, optional
        Inner repetition time in seconds, one per readout. None is as short
        as possible. Default is None.
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
    shuffle_seed : int, optional
        Seed of the shuffling permutation. Default is 0.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment in degrees. Default is 117.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left on the readout axis at the end of each
        inner repetition. Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The MPRAGE sequence object.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    kernel = Mprage3DKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=slab_thickness,
        flip_angle_deg=flip_angle_deg,
        ti=ti,
        tr_outer=tr_outer,
        views_per_segment=views_per_segment,
        ordering=ordering,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        n_acs=n_acs,
        n_acs_z=n_acs_z,
        shuffle_seed=shuffle_seed,
        spoiling_cycles=spoiling_cycles,
    )
    readout = kernel.readout
    inversion = kernel.inversion
    fov_x, fov_y = kernel.fov

    acs_y = range(max(0, n_y // 2 - n_acs // 2), min(n_y, n_y // 2 + -(-n_acs // 2)))
    acs_z = range(
        max(0, n_z // 2 - n_acs_z // 2), min(n_z, n_z // 2 + -(-n_acs_z // 2))
    )

    rf_phases = pp.make_rf_spoiling_schedule(
        len(kernel.segments) * views_per_segment,
        increment=np.deg2rad(rf_spoiling_increment_deg),
    )

    seq = pp.Sequence(system)
    spoiling_phase = iter(rf_phases)
    ima_label, seg_label, eco_label = readout.adc_labels
    seg_label.value = 0
    wait_te = getattr(readout, "wait_te", None)
    wait_tr = getattr(readout, "wait_tr", None)

    def repetition(view, index: int) -> None:
        """One inner spoiled GRE repetition."""
        rf_phase = next(spoiling_phase)
        readout.rf.phase_offset = rf_phase
        readout.adc.phase_offset = rf_phase

        if view is None:
            ky, kz = 0.0, 0.0
        else:
            line, partition = view
            ky = (line - n_y / 2) / (n_y / 2)
            kz = (partition - n_z / 2) / (n_z / 2)

        seq.add_block(readout.rf, readout.gz)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(
            readout.gx_pre,
            pp.scale_grad(readout.gy_pre, ky),
            pp.scale_grad(readout.gz_pre, kz),
        )
        if view is not None:
            line, partition = view
            ima_label.value = int(line in acs_y and partition in acs_z)
            eco_label.value = index
            seq.add_block(readout.gx, readout.adc, ima_label, seg_label, eco_label)
        else:
            seq.add_block(readout.gx)
        seq.add_block(
            readout.gx_spoil,
            pp.scale_grad(readout.gy_rew, ky),
            pp.scale_grad(readout.gz_rew, kz),
        )
        if wait_tr is not None:
            seq.add_block(wait_tr)

    for views in kernel.segments:
        seq.add_block(inversion.rf_prep)
        seq.add_block(inversion.gz_spoil)
        seq.add_block(kernel.wait_ti)
        for index, view in enumerate(views):
            repetition(view, index)
        seq.add_block(kernel.wait_recovery)

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset), system=system
    ).apply_to_sequence(seq, in_place=True)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_z])
    seq.set_definition(key="Name", value="mprage_3d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=tr_outer)
    seq.set_definition(key="TI", value=ti)
    seq.set_definition(key="InnerTR", value=kernel.inner_tr)
    seq.set_definition(key="ViewOrdering", value=ordering)
    seq.set_definition(key="ViewsPerSegment", value=views_per_segment)
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    seq.auto_label()

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


def Mprage3DKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    flip_angle_deg: float = 9.0,
    ti: float = 900e-3,
    tr_outer: float = 2000e-3,
    views_per_segment: int = 64,
    ordering: str = "linear",
    te: float | None = None,
    tr: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    acceleration: int = 1,
    acceleration_z: int = 1,
    n_acs: int = 24,
    n_acs_z: int = 16,
    shuffle_seed: int = 0,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """Design one segment, and the plan that repeats it.

    The requested TI runs from the inversion pulse's centre to the
    *excitation* of the segment's centre-of-k-space view, so the wait after
    the crusher is the TI minus the inversion tail and minus however many
    inner repetitions precede that view -- which the ordering decides. A TI
    too short for either raises ``ValueError``, as does a ``tr_outer``
    shorter than inversion, wait, train and nothing left.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_z, slab_thickness, flip_angle_deg, ti, tr_outer, \
views_per_segment, ordering, te, tr, readout_bandwidth_hz, acceleration, \
acceleration_z, n_acs, n_acs_z, shuffle_seed, spoiling_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``inversion``, ``excitation``, ``readout``, ``segments`` (one view
        list per inversion), ``n_center`` (within-segment index of the
        centre view), ``wait_ti``, ``wait_recovery``, ``fov``,
        ``echo_time``, ``inner_tr``, ``bandwidth_hz`` and ``duration``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    inversion = design.InversionPreparation(
        system, voxel_size_m=min(fov_x / n_x, slab_thickness / n_z)
    )
    excitation = design.SpatialSelectiveExcitation(
        system, flip_angle_deg, slab_thickness, is_slab=True
    )
    readout = design.LineReadout3D(
        system,
        excitation.rf,
        excitation.gz,
        fov=(fov_x, fov_y, slab_thickness),
        matrix=(n_x, n_y, n_z),
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        labels=("IMA", "SEG", "ECO"),
    )
    inner_tr = readout.duration

    sampled_lines = pp.calc_sampled_lines(n_y, acceleration, n_acs)
    sampled_partitions = pp.calc_sampled_lines(n_z, acceleration_z, n_acs_z)
    views = [
        (line, partition)
        for partition in sampled_partitions
        for line in sampled_lines
    ]

    # The centre sits mid-segment for the coherent orderings: TI is defined
    # at the centre view, and mid-segment splits the transient evenly around
    # it. Radial pins the centre to the first view instead.
    n_center = 0 if ordering == "radial" else views_per_segment // 2
    segments = order_views(
        views, views_per_segment, n_center, ordering, (n_y, n_z), seed=shuffle_seed
    )

    # TI from the inversion centre to the centre view's excitation centre:
    # the rest of the inversion module, the solved wait, the repetitions
    # before that view, and the excitation's own place in its block.
    inversion_centre = inversion.rf_prep.delay + inversion.rf_prep.center
    inversion_tail = inversion.seq.duration()[0] - inversion_centre
    excitation_centre = readout.rf.delay + readout.rf.center
    ti_floor = inversion_tail + n_center * inner_tr + excitation_centre
    wait = ti - ti_floor
    if wait < 0:
        raise ValueError(
            f"TI {ti * 1e3:.0f} ms is shorter than the inversion tail and the "
            f"{n_center} repetitions before the centre view admit; the "
            f"minimum is {ti_floor * 1e3:.0f} ms"
        )
    wait_ti = pp.make_delay(
        max(
            system.block_duration_raster,
            pp.round_to_raster(wait, system.block_duration_raster),
        )
    )

    segment_body = (
        inversion.seq.duration()[0]
        + wait_ti.delay
        + views_per_segment * inner_tr
    )
    recovery = tr_outer - segment_body
    if recovery < 0:
        raise ValueError(
            f"TR {tr_outer * 1e3:.0f} ms is shorter than one segment takes "
            f"({segment_body * 1e3:.0f} ms)"
        )
    wait_recovery = pp.make_delay(
        max(
            system.block_duration_raster,
            pp.round_to_raster(recovery, system.block_duration_raster),
        )
    )

    duration = len(segments) * (segment_body + wait_recovery.delay)

    return SimpleNamespace(
        inversion=inversion,
        excitation=excitation,
        readout=readout,
        segments=segments,
        n_center=n_center,
        wait_ti=wait_ti,
        wait_recovery=wait_recovery,
        fov=(fov_x, fov_y),
        echo_time=readout.echo_time,
        inner_tr=inner_tr,
        bandwidth_hz=readout.bandwidth_hz,
        duration=duration,
    )


class Mprage3D(SequencePlugin):
    """The 3D MPRAGE behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.PREP_TIME: TypeinFloatParam(
                    value=900.0,
                    min=50.0,
                    max=3000.0,
                    incr=10.0,
                    unit="ms",
                ),
                UIParam.TR: DropdownFloatParam(
                    value=2000.0,
                    min=300.0,
                    max=8000.0,
                    incr=10.0,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 1500.0, 2000.0, 2500.0, 3000.0],
                ),
                UIParam.FLIP: DropdownFloatParam(
                    value=9.0,
                    min=2.0,
                    max=20.0,
                    incr=1.0,
                    unit="deg",
                    options=[7.0, 9.0, 12.0, 15.0, 18.0],
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
                    value=128, min=16, max=512, incr=1, options=[64, 128, 192, 256, 384]
                ),
                UIParam.NY: DropdownIntParam(
                    value=128, min=16, max=512, incr=1, options=[64, 128, 192, 256, 384]
                ),
                UIParam.NSLICES: DropdownIntParam(
                    value=64, min=8, max=256, incr=1, options=[32, 64, 96, 128, 192]
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=250e3, min=5e3, max=500e3, incr=100.0, unit="Hz"
                ),
                UIParam.RY: TypeinFloatParam(
                    value=1.0, min=1.0, max=8.0, incr=1.0, unit=""
                ),
                UIParam.RZ: TypeinFloatParam(
                    value=1.0, min=1.0, max=8.0, incr=1.0, unit=""
                ),
                UIParam.FOV_OFFSET_X: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Y: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Z: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.user_name(0): Description(text="ACS lines (y)"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=24.0, min=0.0, max=512.0, incr=1.0, unit="lines"
                ),
                UIParam.user_name(1): Description(text="Views per segment"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=64.0, min=1.0, max=1024.0, incr=1.0, unit=""
                ),
                UIParam.user_name(2): Description(
                    text="Ordering 0=lin 1=rad 2=adaptive 3=shuffle"
                ),
                UIParam.user_value(2): TypeinFloatParam(
                    value=0.0, min=0.0, max=3.0, incr=1.0, unit=""
                ),
                UIParam.user_name(5): Description(text="ACS partitions (z)"),
                UIParam.user_value(5): TypeinFloatParam(
                    value=16.0, min=0.0, max=256.0, incr=1.0, unit="lines"
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = _main_kwargs(system, protocol)
        try:
            kernel = Mprage3DKernel(
                system,
                **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s over {len(kernel.segments)} "
                f"segments, inner TR = {kernel.inner_tr * 1e3:.2f} ms"
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
        "flip_angle_deg",
        "ti",
        "tr_outer",
        "views_per_segment",
        "ordering",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "acceleration",
        "acceleration_z",
        "n_acs",
        "n_acs_z",
        "shuffle_seed",
        "spoiling_cycles",
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
        ti=params.param_float(prot, UIParam.PREP_TIME) * 1e-3,
        tr_outer=params.param_float(prot, UIParam.TR) * 1e-3,
        # The UI's TR is the inversion-to-inversion interval; the generic
        # mapping would also hand it to the *inner* repetition, which pads
        # every readout to seconds.
        tr=None,
        views_per_segment=max(1, round(params.user_float(prot, 1, 64.0))),
        ordering=ORDERINGS[int(np.clip(round(params.user_float(prot, 2, 0.0)), 0, 3))],
        n_acs=params.acs_lines_from_protocol(prot, params.param_int(prot, UIParam.NY), 0),
        n_acs_z=max(0, round(params.user_float(prot, 5, 16.0))),
    )


PLUGIN = Mprage3D()


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
    ("--ti-ms", UIParam.PREP_TIME, float, "Inversion time [ms], centre of inversion to centre view"),
    ("--tr-ms", UIParam.TR, float, "Inversion-to-inversion interval [ms]"),
    ("--flip-deg", UIParam.FLIP, float, "Readout flip angle [deg]"),
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
    ("--ry", UIParam.RY, float, "Phase-encode undersampling factor along y"),
    ("--rz", UIParam.RZ, float, "Partition-encode undersampling factor along z"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    ("--offset-y-mm", UIParam.FOV_OFFSET_Y, float, "Volume offset along phase encode [mm]"),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slab [mm]"),
    ("--acs-lines", UIParam.user_value(0), float, "Number of ACS lines along y"),
    ("--views-per-segment", UIParam.user_value(1), float, "Views acquired per inversion"),
    (
        "--ordering",
        UIParam.user_value(2),
        float,
        "View ordering: 0 linear, 1 radial, 2 radial-adaptive, 3 shuffling",
    ),
    ("--acs-partitions", UIParam.user_value(5), float, "Number of ACS partitions along z"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate a 3D MPRAGE .seq offline.",
            default_output="mprage_3d.seq",
        )
    )
