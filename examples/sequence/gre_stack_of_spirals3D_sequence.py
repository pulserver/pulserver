"""RF-spoiled 3D stack-of-spirals gradient echo.

Spiral interleaves in-plane, Cartesian partitions along z --
:class:`design.SpiralStackReadout` from a slab excitation. One solved
interleave serves every ``(arm, partition)``: the arm is a ``ROTATIONS``
extension and the partition an amplitude on the encode pair. The FOV offset
goes through ``TransformFOV`` in server mode.
:mod:`pulserver.app.recon.noncartesian_stack_recon` reconstructs by NUFFT
in-plane and FFT along the stack.

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
from scipy.spatial.transform import Rotation

from pulserver.app.sequence.gre_spiral2D_sequence import arm_angles
from pulserver.app.sequence.gre_stack_of_stars3D_sequence import (
    StackShotKernel,
    stack_angles,
)

#: SLR design of the selective pulses, held here rather than left at the design
#: module's default so a script can retune the excitation without touching the
#: loop. The selection amplitude follows as
#: ``time_bw_product / (duration * thickness)``, which is also what a slice
#: offset is converted against.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0

#: Per-plugin ceilings on the gradient and slew limits, in mT/m and T/m/s. The
#: sequence is held below the smaller of these and what the scanner reports, so
#: lowering them here -- on the scanner console, even -- reruns the whole script
#: under gentler gradients (for PNS headroom, acoustic comfort, eddy currents)
#: without touching anything else. Defaults sit above typical hardware, so they
#: cap nothing until you lower them.
MAX_GRAD = 80.0
MAX_SLEW = 200.0

#: Swap the slab excitation for a spectral-spatial, slab- *and* water-selective
#: pulse. A long readout lets fat -- a few ppm off water -- shift and blur; the
#: non-Cartesian trajectories here spread that shift into a swirl rather than a
#: clean displacement, so exciting water only removes the source. A script-level
#: toggle rather than a UI control: it reshapes the excitation the whole
#: sequence is timed around, so it belongs to whoever runs the script.
SPSP_EXCITATION = False

#: Chemical shift of the fat methylene resonance from water, in ppm. Held in
#: ppm rather than hertz so it is field-strength independent: the water-only
#: excitation converts it against ``system.B0`` when the pulse is built, so the
#: same script targets fat at 1.5 T and 3 T alike.
FAT_SHIFT_PPM = -3.4


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "gre_stack_of_spirals_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    n_arms: int = 16,
    angle_scheme: str = "uniform",
    flip_angle_deg: float = 12.0,
    te: float | None = None,
    tr: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    n_dummy: int = 32,
    n_gain_calibration_readouts: int = 1,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
    partition_angle_offset_deg: float = 0.0,
    use_rotation_ext: bool = True,
) -> pp.Sequence:
    """Create an RF-spoiled 3D stack-of-spirals gradient-echo sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is
        'gre_stack_of_spirals_3d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float, optional
        Isotropic in-plane field of view in meters. Default is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters along the logical
        readout, phase and slab axes. Applied in server mode. Default is
        (0.0, 0.0, 0.0).
    n_x : int, optional
        In-plane matrix size. Default is 128.
    n_z : int, optional
        Number of partitions. Default is 64.
    slab_thickness : float, optional
        Excited slab thickness in meters, also the field of view along z.
        Default is 128e-3.
    n_arms : int, optional
        Interleaves per partition, also the designed pitch. Default is 16.
    angle_scheme : str, optional
        ``uniform`` or ``golden``. Default is 'uniform'.
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 12.0.
    te : float or None, optional
        Echo time in seconds. None is as short as possible. Default is None.
    tr : float or None, optional
        Repetition time in seconds, one per excitation. None is as short as
        possible. Default is None.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    n_dummy : int, optional
        Repetitions played without acquiring before the scan. Default is 32.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment in degrees. Default is 117.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left at the end of each repetition. Default is
        4.0.
    partition_angle_offset_deg : float, optional
        Angle added per partition step, in degrees. Zero plays a given arm
        at the same angle in every partition; anything else turns it a little
        further with each one, staggering the sampling along kz the way a
        CAIPIRINHA shift does. Default is 0.0.
    use_rotation_ext : bool, optional
        Hold one arm and turn it per shot with a ``ROTATIONS`` extension
        (the default), or write every shot out as its own waveform. The
        second costs a waveform per shot -- one per arm and partition both,
        under a partition offset -- and reads without composing a rotation.
        Default is True.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The stack-of-spirals sequence object.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    kernel = StackOfSpiralsKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_z=n_z,
        slab_thickness=slab_thickness,
        n_arms=n_arms,
        angle_scheme=angle_scheme,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        n_dummy=n_dummy,
        spoiling_cycles=spoiling_cycles,
        partition_angle_offset_deg=partition_angle_offset_deg,
        use_rotation_ext=use_rotation_ext,
    )
    readout = kernel.readout
    angles = kernel.angles
    shot_index = kernel.shot_index

    rf_phases = pp.make_rf_spoiling_schedule(
        len(angles) * n_z + n_dummy,
        increment=np.deg2rad(rf_spoiling_increment_deg),
    )
    rotations = (
        [
            pp.make_rotation(Rotation.from_euler("z", float(angle)))
            for angle in kernel.shot_angles
        ]
        if use_rotation_ext
        else [None] * len(kernel.shot_angles)
    )

    seq = pp.Sequence(system)
    spoiling_phase = iter(rf_phases)
    lin_label, par_label = readout.adc_labels

    def repetition(
        i_arm: int, partition: int, kz: float, acquire: bool, mark=None
    ) -> None:
        shot = shot_index(i_arm, partition)
        rf_phase = next(spoiling_phase)
        readout.rf.phase_offset = rf_phase
        readout.adc.phase_offset = rf_phase
        lin_label.value = int(i_arm)
        par_label.value = int(partition)
        StackShotKernel(
            seq,
            readout,
            readout.arm(shot),
            rotations[shot],
            kz,
            acquire=acquire,
            first_extra=() if mark is None else (mark,),
        )

    for i_dummy in range(n_dummy):
        repetition(
            0,
            0,
            0.0,
            acquire=False,
            mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
        )

    clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
    for i_arm in range(len(angles)):
        for partition in range(n_z):
            kz = (partition - n_z / 2) / (n_z / 2)
            repetition(i_arm, partition, kz, acquire=True, mark=clear_once)
            clear_once = None

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset),
        system=system,
        server_mode=True,
    ).apply_to_sequence(seq, in_place=True)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    seq.set_definition(key="FOV", value=[fov, fov, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_x, n_z])
    seq.set_definition(key="Name", value="gre_stack_of_spirals_3d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(key="Trajectory", value="stack_of_spirals")
    seq.set_definition(key="NumArms", value=len(angles))
    seq.set_definition(key="AngleScheme", value=angle_scheme)
    seq.set_definition(key="PartitionAngleOffset", value=partition_angle_offset_deg)
    seq.set_definition(key="kSpaceCenterPartition", value=n_z // 2)
    seq.set_definition(key="kSpaceCenterSample", value=readout.center_sample)
    seq.set_definition(key="SliceThickness", value=slab_thickness)
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def SlabExcitationKernel(system: pp.Opts, flip_angle_deg: float, thickness_m: float):
    """The slab excitation, spectral-spatial when ``SPSP_EXCITATION`` is set.

    Returns ``(excitation, rf, gz)``. The selection gradient carries its own
    rephaser folded onto the end -- as a slab excitation does -- so the stack's
    partition prewinder block, which already encodes z, never holds a second
    z gradient.
    """
    if SPSP_EXCITATION:
        fat_offset_hz = FAT_SHIFT_PPM * 1e-6 * system.gamma * system.B0
        excitation = design.SpspExcitation(
            system,
            flip_angle_deg,
            thickness_m=thickness_m,
            spectral_bandwidth_hz=abs(fat_offset_hz),
            freq_offset_hz=0.0,
        )
        # Concatenate the rephaser onto the alternating selection gradient, the
        # way is_slab does, and hand the readout one merged z lobe.
        gz = pp.concatenate_gradients(excitation.gz, excitation.gz_reph, system=system)
        return excitation, excitation.rf, gz
    excitation = design.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        thickness_m,
        duration_s=PULSE_DURATION,
        is_slab=True,
        time_bw_product=TIME_BW_PRODUCT,
    )
    return excitation, excitation.rf, excitation.gz


def StackOfSpiralsKernel(
    system: pp.Opts,
    *,
    fov: float = 220e-3,
    n_x: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    n_arms: int = 16,
    angle_scheme: str = "uniform",
    flip_angle_deg: float = 12.0,
    te: float | None = None,
    tr: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    n_dummy: int = 32,
    spoiling_cycles: float = 4.0,
    partition_angle_offset_deg: float = 0.0,
    use_rotation_ext: bool = True,
) -> SimpleNamespace:
    """Design the interleave and its stack, and the plan that turns them.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_z, slab_thickness, n_arms, angle_scheme, flip_angle_deg, \
te, tr, readout_bandwidth_hz, n_dummy, spoiling_cycles, \
partition_angle_offset_deg, use_rotation_ext
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``readout``, ``angles``, ``shot_angles``,
        ``shot_index``, ``echo_time``, ``repetition_time``, ``bandwidth_hz``
        and ``duration``.
    """
    excitation, exc_rf, exc_gz = SlabExcitationKernel(
        system, flip_angle_deg, slab_thickness
    )
    angles = arm_angles(n_arms, angle_scheme)
    shot_angles, shot_index = stack_angles(angles, n_z, partition_angle_offset_deg)

    readout = design.SpiralStackReadout(
        system,
        exc_rf,
        exc_gz,
        None,
        fov=fov,
        matrix=n_x,
        fov_z=slab_thickness,
        matrix_z=n_z,
        design_interleaves=n_arms,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        # LIN carries the shot: a spiral plane has no phase-encode line, so
        # the counter a reconstruction grids by is the arm index.
        labels=("LIN", "PAR"),
        explicit=not use_rotation_ext,
        angles=None if use_rotation_ext else shot_angles,
    )

    duration = (n_dummy + len(angles) * n_z) * readout.duration

    return SimpleNamespace(
        excitation=excitation,
        readout=readout,
        angles=angles,
        shot_angles=shot_angles,
        shot_index=shot_index,
        echo_time=readout.echo_time,
        repetition_time=readout.duration,
        bandwidth_hz=readout.bandwidth_hz,
        duration=duration,
    )


# ======================================================================
# The scanner protocol contract
# ======================================================================


class GreStackOfSpirals3D(SequencePlugin):
    """The 3D stack-of-spirals gradient echo behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=-1.0,
                    min=-1.0,
                    max=40.0,
                    incr=0.1,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 2.0, 4.0, 8.0, 15.0],
                ),
                UIParam.TR: DropdownFloatParam(
                    value=-1.0,
                    min=-1.0,
                    max=5000.0,
                    incr=0.1,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 5.0, 10.0, 20.0, 50.0],
                ),
                UIParam.FLIP: DropdownFloatParam(
                    value=12.0,
                    min=1.0,
                    max=90.0,
                    incr=1.0,
                    unit="deg",
                    options=[5.0, 12.0, 30.0, 60.0, 90.0],
                ),
                UIParam.FOV: DropdownFloatParam(
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
                UIParam.NSLICES: DropdownIntParam(
                    value=64, min=8, max=256, incr=1, options=[32, 64, 96, 128, 192]
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=250e3, min=5e3, max=500e3, incr=100.0, unit="Hz"
                ),
                UIParam.FOV_OFFSET_X: OffFloatParam(
                    value=0.0, min=-500.0, max=500.0, unit="mm"
                ),
                UIParam.FOV_OFFSET_Y: OffFloatParam(
                    value=0.0, min=-500.0, max=500.0, unit="mm"
                ),
                UIParam.FOV_OFFSET_Z: OffFloatParam(
                    value=0.0, min=-500.0, max=500.0, unit="mm"
                ),
                UIParam.user_name(0): Description(text="Arms"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=16.0, min=1.0, max=512.0, incr=1.0, unit=""
                ),
                UIParam.user_name(1): Description(text="Angles 0=uniform 1=golden"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=0.0, min=0.0, max=1.0, incr=1.0, unit=""
                ),
                UIParam.user_name(2): Description(text="Dummy scans"),
                UIParam.user_value(2): TypeinFloatParam(
                    value=32.0, min=0.0, max=256.0, incr=1.0, unit="TR"
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = protocol_kwargs(system, protocol)
        try:
            kernel = StackOfSpiralsKernel(
                system,
                **{
                    name: value
                    for name, value in kwargs.items()
                    if name in KERNEL_ARGUMENTS
                },
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s over {len(kernel.angles)} arms "
                f"per partition"
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
        seq = main(**protocol_kwargs(system, protocol))
        write_sequence(seq, output_path, offline=offline)


KERNEL_ARGUMENTS = frozenset(
    (
        "fov",
        "n_x",
        "n_z",
        "slab_thickness",
        "n_arms",
        "angle_scheme",
        "flip_angle_deg",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "n_dummy",
        "spoiling_cycles",
    )
)


def protocol_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
    """The prescribed quantities, plus this sequence's own user slots."""
    prot = dict_to_protocol(protocol)
    n_z = params.param_int(prot, UIParam.NSLICES)
    partition_thickness = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    return main_kwargs(
        main,
        system,
        protocol,
        fov=params.param_float(prot, UIParam.FOV) * 1e-3,
        n_z=n_z,
        slab_thickness=n_z * partition_thickness,
        n_arms=max(1, round(params.user_float(prot, 0, 16.0))),
        angle_scheme="golden" if round(params.user_float(prot, 1, 0.0)) else "uniform",
        n_dummy=max(0, round(params.user_float(prot, 2, 32.0))),
    )


PLUGIN = GreStackOfSpirals3D()


def get_default_protocol(system):
    """Bridge entry point: the plugin's default protocol."""
    return PLUGIN.get_default_protocol(system)


def validate_protocol(system, protocol):
    """Bridge entry point: protocol feasibility and scan duration."""
    return PLUGIN.validate_protocol(system, protocol)


def make_sequence(system, protocol, output_path):
    """Bridge entry point: write the ``.seq`` file."""
    return PLUGIN.make_sequence(system, protocol, output_path)


ARG_MAP = [
    ("--te-ms", UIParam.TE, float, "Echo time [ms], or a negative TEPreset"),
    ("--tr-ms", UIParam.TR, float, "Repetition time [ms], or a negative TRPreset"),
    ("--flip-deg", UIParam.FLIP, float, "Flip angle [deg]"),
    ("--fov-mm", UIParam.FOV, float, "In-plane FOV [mm]"),
    (
        "--partition-thickness-mm",
        UIParam.SLICE_THICKNESS,
        float,
        "Partition thickness [mm]; the slab is this times the partition count",
    ),
    ("--nx", UIParam.NX, int, "In-plane matrix size"),
    ("--nz", UIParam.NSLICES, int, "Partition count"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    (
        "--offset-y-mm",
        UIParam.FOV_OFFSET_Y,
        float,
        "Volume offset along phase encode [mm]",
    ),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slab [mm]"),
    ("--arms", UIParam.user_value(0), float, "Interleaves per partition"),
    ("--angles", UIParam.user_value(1), float, "Angle scheme: 0 uniform, 1 golden"),
    ("--dummies", UIParam.user_value(2), float, "Unacquired repetitions"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=ARG_MAP,
            description="Generate a 3D stack-of-spirals gradient-echo .seq offline.",
            default_output="gre_stack_of_spirals_3d.seq",
        )
    )
