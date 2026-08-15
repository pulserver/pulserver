"""3D MPRAGE on a golden-angle stack of spirals, with explicit arms.

The untemplatable-waveform stress case: every golden-angle arm is written
out as its own gradient waveform -- ``explicit=True`` on
:class:`design.SpiralStackReadout`, no ``ROTATIONS`` extension, the shape
library growing with the arm count -- which is exactly the path an
interpreter's waveform streaming has to survive. One inversion per segment;
a segment plays one arm through every partition, the centre partition at the
requested TI; segments step by the golden angle so any prefix of the scan
covers the disc near-uniformly.
:mod:`pulserver.reczoo.mprage_stack_of_spirals_3d` reconstructs the stack
against the trajectory the acquisitions carry.

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


def _build_slab_excitation(
    system: pp.Opts, flip_angle_deg: float, thickness_m: float
):
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
        gz = pp.concatenate_gradients(
            excitation.gz, excitation.gz_reph, system=system
        )
        return excitation, excitation.rf, gz
    excitation = design.SpatialSelectiveExcitation(
        system, flip_angle_deg, thickness_m, is_slab=True
    )
    return excitation, excitation.rf, excitation.gz


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "mprage_stack_of_spirals_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    n_arms: int = 16,
    design_interleaves: int | None = None,
    flip_angle_deg: float = 9.0,
    ti: float = 900e-3,
    tr_outer: float = 2000e-3,
    te: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    n_gain_calibration_readouts: int = 1,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Create a golden-angle stack-of-spirals 3D MPRAGE with explicit arms.

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
        'mprage_stack_of_spirals_3d.seq'.
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
        Number of partitions; a segment plays them all. Default is 64.
    slab_thickness : float, optional
        Excited slab thickness in meters, also the field of view along z.
        Default is 128e-3.
    n_arms : int, optional
        Golden-angle arms, one per segment -- and, explicit, one waveform
        each. Default is 16.
    design_interleaves : int or None, optional
        The pitch the spiral is designed for. None matches ``n_arms``.
        Default is None.
    flip_angle_deg : float, optional
        Readout excitation flip angle in degrees. Default is 9.0.
    ti : float, optional
        Inversion time in seconds, inversion centre to the centre
        partition's excitation. Default is 900e-3.
    tr_outer : float, optional
        Inversion-to-inversion interval in seconds. Default is 2000e-3.
    te : float or None, optional
        Readout echo time in seconds. None is as short as possible. Default
        is None.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment in degrees. Default is 117.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left at the end of each inner repetition.
        Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The MPRAGE sequence object.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    kernel = MprageStackOfSpiralsKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_z=n_z,
        slab_thickness=slab_thickness,
        n_arms=n_arms,
        design_interleaves=design_interleaves,
        flip_angle_deg=flip_angle_deg,
        ti=ti,
        tr_outer=tr_outer,
        te=te,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
    )
    readout = kernel.readout
    inversion = kernel.inversion

    rf_phases = pp.make_rf_spoiling_schedule(
        n_arms * n_z,
        increment=np.deg2rad(rf_spoiling_increment_deg),
    )

    seq = pp.Sequence(system)
    spoiling_phase = iter(rf_phases)
    par_label = pp.make_label("PAR", "SET", 0)
    seg_label = pp.make_label("SEG", "SET", 0)
    template = readout.blocks[:4]

    def repetition(arm: int, partition: int) -> None:
        """One inner repetition, the arm's own waveforms substituted in."""
        kz = (partition - n_z / 2) / (n_z / 2)
        rf_phase = next(spoiling_phase)
        readout.rf.phase_offset = rf_phase
        readout.adc.phase_offset = rf_phase
        par_label.value = int(partition)
        seg_label.value = int(arm)

        for block in template:
            events = []
            for event in block:
                if event is readout.gx[0]:
                    event = readout.gx[arm]
                elif event is readout.gy[0]:
                    event = readout.gy[arm]
                elif event is readout.gx_rew[0]:
                    event = readout.gx_rew[arm]
                elif event is readout.gy_rew[0]:
                    event = readout.gy_rew[arm]
                elif event is readout.gz_pre:
                    event = pp.scale_grad(readout.gz_pre, kz)
                elif event is readout.gz_rew:
                    event = pp.scale_grad(readout.gz_rew, kz)
                events.append(event)
            if any(getattr(event, "type", "") == "adc" for event in events):
                seq.add_block(*events, par_label, seg_label)
            else:
                seq.add_block(*events)

    for arm in range(n_arms):
        seq.add_block(inversion.rf_prep)
        seq.add_block(inversion.gz_spoil)
        seq.add_block(kernel.wait_ti)
        for partition in range(n_z):
            repetition(arm, partition)
        seq.add_block(kernel.wait_recovery)

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
    seq.set_definition(key="Name", value="mprage_stack_of_spirals_3d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=tr_outer)
    seq.set_definition(key="TI", value=ti)
    seq.set_definition(key="InnerTR", value=kernel.inner_tr)
    seq.set_definition(key="Trajectory", value="stack_of_spirals")
    seq.set_definition(key="NumArms", value=n_arms)
    seq.set_definition(key="AngleScheme", value="golden")
    seq.set_definition(key="ExplicitArms", value=1)
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


def MprageStackOfSpiralsKernel(
    system: pp.Opts,
    *,
    fov: float = 220e-3,
    n_x: int = 128,
    n_z: int = 64,
    slab_thickness: float = 128e-3,
    n_arms: int = 16,
    design_interleaves: int | None = None,
    flip_angle_deg: float = 9.0,
    ti: float = 900e-3,
    tr_outer: float = 2000e-3,
    te: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """Design the explicit arms and the segment timing around them.

    TI runs from the inversion pulse's centre to the excitation of the
    centre partition -- partition ``n_z // 2``, played mid-segment since the
    partitions step linearly. A TI or an outer TR too short for the segment
    raises ``ValueError``.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_z, slab_thickness, n_arms, design_interleaves, \
flip_angle_deg, ti, tr_outer, te, readout_bandwidth_hz, spoiling_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``inversion``, ``excitation``, ``readout`` (explicit: ``gx``/``gy``
        and their rewinders are lists, one per arm), ``angles``,
        ``wait_ti``, ``wait_recovery``, ``echo_time``, ``inner_tr``,
        ``bandwidth_hz`` and ``duration``.
    """
    inversion = design.InversionPreparation(
        system, voxel_size_m=min(fov / n_x, slab_thickness / n_z)
    )
    excitation, exc_rf, exc_gz = _build_slab_excitation(
        system, flip_angle_deg, slab_thickness
    )
    angles = np.asarray(pp.calc_golden_angles(n_arms, full_circle=True))
    readout = design.SpiralStackReadout(
        system,
        exc_rf,
        exc_gz,
        None,
        fov=fov,
        matrix=n_x,
        fov_z=slab_thickness,
        matrix_z=n_z,
        design_interleaves=n_arms if design_interleaves is None else design_interleaves,
        explicit=True,
        angles=angles,
        te=te,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
    )
    # One explicit repetition spans four blocks; every arm lays the same
    # timing, so the first arm's span is the inner TR.
    inner_tr = sum(pp.calc_duration(*block) for block in readout.blocks[:4])

    n_center = n_z // 2
    inversion_centre = inversion.rf_prep.delay + inversion.rf_prep.center
    inversion_tail = inversion.seq.duration()[0] - inversion_centre
    excitation_centre = readout.rf.delay + readout.rf.center
    ti_floor = inversion_tail + n_center * inner_tr + excitation_centre
    wait = ti - ti_floor
    if wait < 0:
        raise ValueError(
            f"TI {ti * 1e3:.0f} ms is shorter than the inversion tail and the "
            f"{n_center} repetitions before the centre partition admit; the "
            f"minimum is {ti_floor * 1e3:.0f} ms"
        )
    wait_ti = pp.make_delay(
        max(
            system.block_duration_raster,
            pp.round_to_raster(wait, system.block_duration_raster),
        )
    )

    segment_body = inversion.seq.duration()[0] + wait_ti.delay + n_z * inner_tr
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

    duration = n_arms * (segment_body + wait_recovery.delay)

    return SimpleNamespace(
        inversion=inversion,
        excitation=excitation,
        readout=readout,
        angles=angles,
        wait_ti=wait_ti,
        wait_recovery=wait_recovery,
        echo_time=readout.echo_time,
        inner_tr=inner_tr,
        bandwidth_hz=readout.bandwidth_hz,
        duration=duration,
    )


class MprageStackOfSpirals3D(SequencePlugin):
    """The explicit-arms spiral MPRAGE behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.PREP_TIME: TypeinFloatParam(
                    value=900.0, min=50.0, max=3000.0, incr=10.0, unit="ms"
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
                UIParam.FOV_OFFSET_X: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Y: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Z: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.user_name(0): Description(text="Arms (one per inversion)"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=16.0, min=1.0, max=512.0, incr=1.0, unit=""
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = _main_kwargs(system, protocol)
        try:
            kernel = MprageStackOfSpiralsKernel(
                system,
                **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s over {len(kernel.angles)} "
                f"explicit arms, inner TR = {kernel.inner_tr * 1e3:.2f} ms"
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
        "n_z",
        "slab_thickness",
        "n_arms",
        "design_interleaves",
        "flip_angle_deg",
        "ti",
        "tr_outer",
        "te",
        "readout_bandwidth_hz",
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
        fov=params.param_float(prot, UIParam.FOV) * 1e-3,
        n_z=n_z,
        slab_thickness=n_z * partition_thickness,
        ti=params.param_float(prot, UIParam.PREP_TIME) * 1e-3,
        tr_outer=params.param_float(prot, UIParam.TR) * 1e-3,
        n_arms=max(1, round(params.user_float(prot, 0, 16.0))),
    )


PLUGIN = MprageStackOfSpirals3D()


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
    ("--ti-ms", UIParam.PREP_TIME, float, "Inversion time [ms]"),
    ("--tr-ms", UIParam.TR, float, "Inversion-to-inversion interval [ms]"),
    ("--flip-deg", UIParam.FLIP, float, "Readout flip angle [deg]"),
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
    ("--offset-y-mm", UIParam.FOV_OFFSET_Y, float, "Volume offset along phase encode [mm]"),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slab [mm]"),
    ("--arms", UIParam.user_value(0), float, "Golden-angle arms, one per inversion"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate an explicit-arms stack-of-spirals MPRAGE .seq offline.",
            default_output="mprage_stack_of_spirals_3d.seq",
        )
    )
