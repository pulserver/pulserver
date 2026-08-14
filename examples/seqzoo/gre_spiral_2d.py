"""RF-spoiled 2D spiral gradient echo, multi-slice.

One solved spiral interleave -- :class:`design.SpiralReadout2D` -- turned per
shot by a ``ROTATIONS`` extension: one registered waveform however many arms
the scan plays. The FOV offset goes through ``TransformFOV`` in server mode,
where a rotated readout defers its ADC shift to the consumer of the base
trajectory. :mod:`pulserver.reczoo.gre_spiral_2d` reconstructs by NUFFT
against the trajectory the file itself carries.

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

#: The arm-angle schemes on offer. A spiral covers a full turn, so golden is
#: the 2*pi golden angle and uniform divides the whole circle.
ANGLE_SCHEMES = ("golden", "uniform")

#: The golden angle over a full turn, pi * (3 - sqrt(5)).
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def arm_angles(n_arms: int, scheme: str) -> np.ndarray:
    """The rotation of every arm, in radians.

    Parameters
    ----------
    n_arms : int
        How many arms the scan plays.
    scheme : str
        One of :data:`ANGLE_SCHEMES`.

    Returns
    -------
    numpy.ndarray
        One angle per arm.
    """
    if scheme not in ANGLE_SCHEMES:
        raise ValueError(f"scheme must be one of {ANGLE_SCHEMES}, got {scheme!r}")
    if scheme == "golden":
        return np.arange(n_arms) * GOLDEN_ANGLE
    return np.arange(n_arms) * 2.0 * np.pi / n_arms


def play(seq, readout, rotation, *, acquire: bool, labels=(), first_extra=()) -> None:
    """Replay the module's blocks with the shot's orientation injected.

    Every block that drives an in-plane gradient gains the rotation; the
    acquisition block gains the labels; a dummy drops the ADC. The blocks are
    replayed as the module laid them out, delays included, so the TE budget
    the module solved survives the loop untouched.
    """
    extra_first = list(first_extra)
    for block in readout.blocks:
        events = [
            event
            for event in block
            if acquire or getattr(event, "type", "") != "adc"
        ]
        additions = list(extra_first)
        extra_first = []
        if any(getattr(event, "channel", "") in ("x", "y") for event in events):
            additions.append(rotation)
        if acquire and any(getattr(event, "type", "") == "adc" for event in events):
            additions.extend(labels)
        seq.add_block(*events, *additions)


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "gre_spiral_2d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_arms: int = 16,
    angle_scheme: str = "uniform",
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    slice_order: str = "interleaved",
    flip_angle_deg: float = 12.0,
    te: float | None = None,
    tr: float | None = 20e-3,
    readout_bandwidth_hz: float = 250e3,
    n_dummy: int = 16,
    n_gain_calibration_readouts: int | None = None,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Create an RF-spoiled 2D spiral gradient-echo sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'gre_spiral_2d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float, optional
        Isotropic in-plane field of view in meters. Default is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters along the logical
        readout, phase and slice axes. Applied in server mode. Default is
        (0.0, 0.0, 0.0).
    n_x : int, optional
        In-plane matrix size. Default is 128.
    n_arms : int, optional
        Interleaves played, which is also the pitch the spiral is designed
        for. Default is 16.
    angle_scheme : str, optional
        ``uniform`` or ``golden``. Default is 'uniform'.
    n_slices : int, optional
        Number of slices. Default is 1.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 5e-3.
    slice_gap : float, optional
        Gap between adjacent slices in meters. Default is 0.0.
    slice_order : str, optional
        Order the slices of one pass are excited in. Default is
        'interleaved'.
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 12.0.
    te : float or None, optional
        Echo time in seconds, to the start of the outward path. None is as
        short as possible. Default is None.
    tr : float or None, optional
        Repetition time in seconds, between successive excitations of the
        same slice. Default is 20e-3.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    n_dummy : int, optional
        Repetitions played without acquiring, before the first arm of each
        pass. Default is 16.
    n_gain_calibration_readouts : int or None, optional
        Written as the ``NumGainCalibrationReadouts`` definition. None is
        one per slice. Default is None.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment in degrees. Default is 117.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left on the slice axis at the end of each
        repetition. Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The spiral GRE sequence object.
    """
    system = pp.Opts() if system is None else system

    kernel = SpiralKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_arms=n_arms,
        angle_scheme=angle_scheme,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        slice_order=slice_order,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        n_dummy=n_dummy,
        spoiling_cycles=spoiling_cycles,
    )
    excitation = kernel.excitation
    angles = kernel.angles
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )
    rf_phases = pp.make_rf_spoiling_schedule(
        (len(angles) + n_dummy) * n_slices,
        increment=np.deg2rad(rf_spoiling_increment_deg),
    )
    rotations = [
        pp.make_rotation(Rotation.from_euler("z", float(angle))) for angle in angles
    ]

    seq = pp.Sequence(system)
    spoiling_phase = iter(rf_phases)
    slc_label = pp.make_label("SLC", "SET", 0)

    def repetition(readout, slices, rotation, acquire: bool, mark=None) -> None:
        for i_slice in slices:
            rf_phase = next(spoiling_phase)
            readout.rf.freq_offset = excitation.gz.amplitude * slice_positions[i_slice]
            readout.rf.phase_offset = (
                rf_phase - 2 * np.pi * readout.rf.freq_offset * readout.rf.center
            )
            readout.adc.phase_offset = rf_phase
            slc_label.value = int(i_slice)

            play(
                seq,
                readout,
                rotation,
                acquire=acquire,
                labels=(slc_label,),
                first_extra=() if mark is None else (mark,),
            )
            mark = None

    for slices in kernel.passes:
        readout = kernel.readouts[len(slices)]
        for i_dummy in range(n_dummy):
            repetition(
                readout,
                slices,
                rotations[0],
                acquire=False,
                mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
            )
        clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
        for rotation in rotations:
            repetition(readout, slices, rotation, acquire=True, mark=clear_once)
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

    slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap
    seq.set_definition(key="FOV", value=[fov, fov, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_x, n_slices])
    seq.set_definition(key="Name", value="gre_spiral_2d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(key="Trajectory", value="spiral")
    seq.set_definition(key="NumArms", value=len(angles))
    seq.set_definition(key="AngleScheme", value=angle_scheme)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=n_slices if n_gain_calibration_readouts is None else n_gain_calibration_readouts,
    )

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


def SpiralKernel(
    system: pp.Opts,
    *,
    fov: float = 220e-3,
    n_x: int = 128,
    n_arms: int = 16,
    angle_scheme: str = "uniform",
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_order: str = "interleaved",
    flip_angle_deg: float = 12.0,
    te: float | None = None,
    tr: float | None = 20e-3,
    readout_bandwidth_hz: float = 250e3,
    n_dummy: int = 16,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """Design the interleave, and the plan that turns it.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_arms, angle_scheme, n_slices, slice_thickness, slice_order, \
flip_angle_deg, te, tr, readout_bandwidth_hz, n_dummy, spoiling_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``readouts`` (keyed by pass size), ``passes``,
        ``angles``, ``echo_time``, ``repetition_time``, ``bandwidth_hz``
        and ``duration``.
    """
    excitation = design.SpatialSelectiveExcitation(
        system, flip_angle_deg, slice_thickness
    )

    def readout(module_tr: float | None):
        return design.SpiralReadout2D(
            system,
            excitation.rf,
            excitation.gz,
            excitation.gz_reph,
            fov=fov,
            matrix=n_x,
            design_interleaves=n_arms,
            te=te,
            tr=module_tr,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
        )

    shortest = readout(None)
    per_pass = n_slices if tr is None else max(1, int(tr / shortest.duration))
    n_passes = -(-n_slices // per_pass)
    passes = [
        [int(i) for i in range(start, n_slices, n_passes)] for start in range(n_passes)
    ]
    passes = [
        [group[int(i)] for i in pp.calc_traversal_order(len(group), slice_order)]
        for group in passes
        if group
    ]
    readouts = {
        size: shortest if tr is None else readout(tr / size)
        for size in {len(group) for group in passes}
    }

    angles = arm_angles(n_arms, angle_scheme)
    pass_time = sum(len(group) * readouts[len(group)].duration for group in passes)
    duration = (n_dummy + len(angles)) * pass_time

    return SimpleNamespace(
        excitation=excitation,
        readouts=readouts,
        passes=passes,
        angles=angles,
        echo_time=shortest.echo_time,
        repetition_time=max(
            len(group) * readouts[len(group)].duration for group in passes
        ),
        bandwidth_hz=shortest.bandwidth_hz,
        duration=duration,
    )


class GreSpiral2D(SequencePlugin):
    """The 2D spiral gradient echo behind the scanner protocol contract."""

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
                    value=20.0,
                    min=2.0,
                    max=5000.0,
                    incr=0.1,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 10.0, 20.0, 50.0, 250.0],
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
                    value=5.0,
                    min=1.0,
                    max=20.0,
                    incr=0.5,
                    unit="mm",
                    options=[1.0, 3.0, 5.0, 8.0, 10.0],
                ),
                UIParam.SLICE_SPACING: DropdownFloatParam(
                    value=5.0,
                    min=1.0,
                    max=20.0,
                    incr=0.5,
                    unit="mm",
                    options=[1.0, 3.0, 5.0, 8.0, 10.0],
                ),
                UIParam.NX: DropdownIntParam(
                    value=128, min=16, max=512, incr=1, options=[64, 128, 192, 256, 384]
                ),
                UIParam.NSLICES: DropdownIntParam(
                    value=1, min=1, max=128, incr=1, options=[1, 5, 10, 20, 40]
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=250e3, min=5e3, max=500e3, incr=100.0, unit="Hz"
                ),
                UIParam.FOV_OFFSET_X: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Y: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Z: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
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
                    value=16.0, min=0.0, max=128.0, incr=1.0, unit="TR"
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        kwargs = _main_kwargs(system, protocol)
        try:
            kernel = SpiralKernel(
                system,
                **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s over {len(kernel.angles)} arms, "
                f"{len(kernel.passes)} pass(es)"
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
        "n_arms",
        "angle_scheme",
        "n_slices",
        "slice_thickness",
        "slice_order",
        "flip_angle_deg",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "n_dummy",
        "spoiling_cycles",
    )
)


def _main_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
    """The prescribed quantities, plus this sequence's own user slots."""
    prot = dict_to_protocol(protocol)
    return main_kwargs(
        main,
        system,
        protocol,
        fov=params.param_float(prot, UIParam.FOV) * 1e-3,
        n_arms=max(1, round(params.user_float(prot, 0, 16.0))),
        angle_scheme="golden" if round(params.user_float(prot, 1, 0.0)) else "uniform",
        n_dummy=max(0, round(params.user_float(prot, 2, 16.0))),
    )


PLUGIN = GreSpiral2D()


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
    ("--te-ms", UIParam.TE, float, "Echo time [ms], or a negative TEPreset"),
    ("--tr-ms", UIParam.TR, float, "Repetition time [ms], or a negative TRPreset"),
    ("--flip-deg", UIParam.FLIP, float, "Flip angle [deg]"),
    ("--fov-mm", UIParam.FOV, float, "In-plane FOV [mm]"),
    ("--slice-thickness-mm", UIParam.SLICE_THICKNESS, float, "Slice thickness [mm]"),
    ("--slice-spacing-mm", UIParam.SLICE_SPACING, float, "Slice spacing [mm]"),
    ("--nx", UIParam.NX, int, "In-plane matrix size"),
    ("--nslices", UIParam.NSLICES, int, "Number of slices"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    ("--offset-y-mm", UIParam.FOV_OFFSET_Y, float, "Volume offset along phase encode [mm]"),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slice [mm]"),
    ("--arms", UIParam.user_value(0), float, "Interleaves to play"),
    ("--angles", UIParam.user_value(1), float, "Angle scheme: 0 uniform, 1 golden"),
    ("--dummies", UIParam.user_value(2), float, "Unacquired repetitions per pass"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate a 2D spiral gradient-echo .seq offline.",
            default_output="gre_spiral_2d.seq",
        )
    )
