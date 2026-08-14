"""2D PROPELLER spin echo, multi-slice.

One spin echo per blade: a slice-selective SLR excitation, one SLR 180 with
bridged crushers, and a short EPI train -- :class:`design.PropellerReadout2D`
-- straddling the centre of k-space, turned per shot by a ``ROTATIONS``
extension. The blade's centre line is placed at the spin-echo condition: the
requested TE runs excitation centre to the blade's central echo, with the
180 at its midpoint, so the half the readout owns is solved as
``TE/2 - (blade_width/2) * esp`` on its first line. Every blade samples the
centre, which is what PROPELLER trades speed for.
:mod:`pulserver.reczoo.se_propeller_2d` reconstructs the blades as one
non-Cartesian set against the trajectory the acquisitions carry.

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

#: SLR design shared by the excitation and the refocusing pulse.
_PULSE_DURATION = 3e-3
_TIME_BW_PRODUCT = 4.0


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "se_propeller_2d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    blade_width: int = 16,
    n_blades: int | None = None,
    angle_scheme: str = "uniform",
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    slice_order: str = "interleaved",
    te: float | None = 80e-3,
    tr: float | None = 2000e-3,
    readout_bandwidth_hz: float = 250e3,
    crusher_cycles: float = 4.0,
    n_gain_calibration_readouts: int | None = None,
) -> pp.Sequence:
    """Create a 2D PROPELLER spin-echo sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'se_propeller_2d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float, optional
        Isotropic in-plane field of view in meters -- the blade turns, so it
        cannot be anything else. Default is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters along the logical
        readout, phase and slice axes. Applied in server mode: rotated
        readouts defer their ADC shift to the consumer. Default is
        (0.0, 0.0, 0.0).
    n_x : int, optional
        In-plane matrix size. Default is 128.
    blade_width : int, optional
        Phase-encode lines per blade. Default is 16.
    n_blades : int or None, optional
        Blades in the set. None is the smallest count that samples the rim
        at Nyquist. Default is None.
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
    te : float or None, optional
        Effective echo time in seconds, excitation centre to the blade's
        central line, with the 180 at its midpoint. None is as short as
        possible. Default is 80e-3.
    tr : float or None, optional
        Repetition time in seconds, between successive excitations of the
        same slice. Default is 2000e-3.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    crusher_cycles : float, optional
        Cycles of dephasing each crusher beside the refocusing pulse winds.
        Default is 4.0.
    n_gain_calibration_readouts : int or None, optional
        Written as the ``NumGainCalibrationReadouts`` definition. None is
        one per slice. Default is None.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The PROPELLER sequence object.
    """
    system = pp.Opts() if system is None else system

    kernel = PropellerKernel(
        system,
        fov=fov,
        n_x=n_x,
        blade_width=blade_width,
        n_blades=n_blades,
        angle_scheme=angle_scheme,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        slice_order=slice_order,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        crusher_cycles=crusher_cycles,
    )
    excitation = kernel.excitation
    refocusing = kernel.refocusing
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )

    seq = pp.Sequence(system)
    slc_label = pp.make_label("SLC", "SET", 0)
    seg_label = pp.make_label("SEG", "SET", 0)
    eco_label = pp.make_label("ECO", "SET", 0)

    def shot(blade, timing, slices, rotation, blade_index: int) -> None:
        """One blade of every slice of a pass."""
        for i_slice in slices:
            position = slice_positions[i_slice]
            exc_hz = excitation.gz.amplitude * position
            ref_hz = kernel.refocusing_amplitude * position
            blade.rf.freq_offset = ref_hz
            blade.rf.phase_offset = np.pi / 2 - 2 * np.pi * ref_hz * blade.rf.center
            excitation.rf.freq_offset = exc_hz
            excitation.rf.phase_offset = -2 * np.pi * exc_hz * excitation.rf.center
            slc_label.value = int(i_slice)
            seg_label.value = int(blade_index)

            seq.add_block(excitation.rf, excitation.gz)
            seq.add_block(excitation.gz_reph)
            if timing.wait_half_te is not None:
                seq.add_block(timing.wait_half_te)

            seq.add_block(blade.rf, blade.gz)
            if getattr(blade, "wait_te", None) is not None:
                seq.add_block(blade.wait_te)
            seq.add_block(
                blade.gx_pre,
                pp.scale_grad(blade.gy_pre, blade.blade_start),
                rotation,
            )
            for line in range(blade.etl):
                eco_label.value = int(line)
                blip = blade.gy_blips[line]
                seq.add_block(
                    blade.gx[line],
                    blade.adc,
                    *([blip] if blip is not None else []),
                    rotation,
                    slc_label,
                    seg_label,
                    eco_label,
                )
            seq.add_block(blade.gx_spoil, rotation)
            if timing.wait_tr is not None:
                seq.add_block(timing.wait_tr)

    for slices in kernel.passes:
        blade, timing = kernel.repetitions[len(slices)]
        rotations = [
            pp.make_rotation(Rotation.from_euler("z", float(angle)))
            for angle in blade.blade_angles
        ]
        for blade_index, rotation in enumerate(rotations):
            shot(blade, timing, slices, rotation, blade_index)

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
    seq.set_definition(key="Name", value="se_propeller_2d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(key="Trajectory", value="propeller")
    seq.set_definition(key="BladeWidth", value=blade_width)
    seq.set_definition(key="NumBlades", value=kernel.n_blades)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=n_slices if n_gain_calibration_readouts is None else n_gain_calibration_readouts,
    )

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


def PropellerKernel(
    system: pp.Opts,
    *,
    fov: float = 220e-3,
    n_x: int = 128,
    blade_width: int = 16,
    n_blades: int | None = None,
    angle_scheme: str = "uniform",
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_order: str = "interleaved",
    te: float | None = 80e-3,
    tr: float | None = 2000e-3,
    readout_bandwidth_hz: float = 250e3,
    crusher_cycles: float = 4.0,
) -> SimpleNamespace:
    """Design the blade, and the spin-echo timing that centres it.

    The blade is an EPI train whose ``te`` sets its *first* line, while the
    spin-echo condition belongs to its *central* line: the half the readout
    owns is therefore ``TE/2 - (blade_width // 2) * esp`` on the first line,
    and the excitation half is a delay sized as in the plain spin echo. A TE
    shorter than either half admits raises ``ValueError``.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, blade_width, n_blades, angle_scheme, n_slices, \
slice_thickness, slice_order, te, tr, readout_bandwidth_hz, crusher_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``refocusing``, ``refocusing_amplitude``,
        ``repetitions`` (``(blade, timing)`` keyed by pass size),
        ``passes``, ``n_blades``, ``echo_time``, ``repetition_time``,
        ``bandwidth_hz`` and ``duration``.
    """
    excitation = design.SpatialSelectiveExcitation(
        system,
        90.0,
        slice_thickness,
        duration_s=_PULSE_DURATION,
        time_bw_product=_TIME_BW_PRODUCT,
    )
    refocusing = design.SpatialSelectiveRefocusing(
        system,
        slice_thickness,
        duration_s=_PULSE_DURATION,
        time_bw_product=_TIME_BW_PRODUCT,
        spoiling_cycles=crusher_cycles,
    )
    refocusing_amplitude = _TIME_BW_PRODUCT / (_PULSE_DURATION * slice_thickness)

    exc_center = excitation.rf.delay + excitation.rf.center
    half_te_floor = (
        excitation.seq.duration()[0] - exc_center
    ) + refocusing.center

    def build(first_line_te: float | None):
        return design.PropellerReadout2D(
            system,
            refocusing.rf_ref,
            refocusing.gz,
            fov=fov,
            matrix=n_x,
            blade_width=blade_width,
            n_blades=n_blades,
            scheme=angle_scheme,
            te=first_line_te,
            readout_bandwidth_hz=readout_bandwidth_hz,
        )

    probe = build(None)
    centre_delta = float(
        probe.echo_times[blade_width // 2] - probe.echo_times[0]
    )

    if te is None:
        half_te = max(probe.echo_time + centre_delta, half_te_floor)
    else:
        half_te = te / 2
    first_line_te = half_te - centre_delta
    if first_line_te < probe.echo_time - 1e-9:
        raise ValueError(
            f"TE {2 * half_te * 1e3:.1f} ms is shorter than the blade half "
            f"admits; the minimum is "
            f"{2 * (probe.echo_time + centre_delta) * 1e3:.1f} ms"
        )
    if half_te_floor > half_te + 1e-9:
        raise ValueError(
            f"TE {2 * half_te * 1e3:.1f} ms is shorter than the excitation "
            f"half admits; the minimum is {2 * half_te_floor * 1e3:.1f} ms"
        )

    blade = build(first_line_te)
    wait = half_te - half_te_floor
    wait_half_te = (
        pp.make_delay(pp.round_to_raster(wait, system.block_duration_raster))
        if wait > 0
        else None
    )

    length = (
        excitation.seq.duration()[0]
        + (wait_half_te.delay if wait_half_te is not None else 0.0)
        + blade.seq.duration()[0]
    )

    def with_tr(module_tr: float | None):
        wait_tr = None
        total = length
        if module_tr is not None:
            if module_tr < length - 1e-9:
                raise ValueError(
                    f"TR {module_tr * 1e3:.1f} ms is shorter than one blade "
                    f"takes ({length * 1e3:.1f} ms)"
                )
            pad = pp.round_to_raster(
                module_tr - length, system.block_duration_raster
            )
            if pad > 0:
                wait_tr = pp.make_delay(pad)
                total += pad
        return SimpleNamespace(
            wait_half_te=wait_half_te, wait_tr=wait_tr, length=total
        )

    shortest_timing = with_tr(None)
    per_pass = n_slices if tr is None else max(1, int(tr / shortest_timing.length))
    n_passes = -(-n_slices // per_pass)
    passes = [
        [int(i) for i in range(start, n_slices, n_passes)] for start in range(n_passes)
    ]
    passes = [
        [group[int(i)] for i in pp.calc_traversal_order(len(group), slice_order)]
        for group in passes
        if group
    ]
    repetitions = {
        size: (blade, shortest_timing if tr is None else with_tr(tr / size))
        for size in {len(group) for group in passes}
    }

    pass_time = sum(
        len(group) * repetitions[len(group)][1].length for group in passes
    )
    duration = blade.n_blades * pass_time

    return SimpleNamespace(
        excitation=excitation,
        refocusing=refocusing,
        refocusing_amplitude=refocusing_amplitude,
        repetitions=repetitions,
        passes=passes,
        n_blades=blade.n_blades,
        echo_time=2 * half_te,
        repetition_time=max(
            len(group) * repetitions[len(group)][1].length for group in passes
        ),
        bandwidth_hz=blade.bandwidth_hz,
        duration=duration,
    )


class SePropeller2D(SequencePlugin):
    """The 2D PROPELLER spin echo behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=80.0,
                    min=10.0,
                    max=300.0,
                    incr=1.0,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 60.0, 80.0, 100.0, 120.0],
                ),
                UIParam.TR: DropdownFloatParam(
                    value=2000.0,
                    min=100.0,
                    max=15000.0,
                    incr=10.0,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 2000.0, 3000.0, 4000.0, 6000.0],
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
                UIParam.user_name(0): Description(text="Blade width [lines]"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=16.0, min=4.0, max=128.0, incr=1.0, unit="lines"
                ),
                UIParam.user_name(1): Description(text="Blades (0 = Nyquist)"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=0.0, min=0.0, max=256.0, incr=1.0, unit=""
                ),
                UIParam.user_name(2): Description(text="Angles 0=uniform 1=golden"),
                UIParam.user_value(2): TypeinFloatParam(
                    value=0.0, min=0.0, max=1.0, incr=1.0, unit=""
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        kwargs = _main_kwargs(system, protocol)
        try:
            kernel = PropellerKernel(
                system,
                **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s over {kernel.n_blades} blades, "
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
        "blade_width",
        "n_blades",
        "angle_scheme",
        "n_slices",
        "slice_thickness",
        "slice_order",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "crusher_cycles",
    )
)


def _main_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
    """The prescribed quantities, plus this sequence's own user slots."""
    prot = dict_to_protocol(protocol)
    requested = round(params.user_float(prot, 1, 0.0))
    return main_kwargs(
        main,
        system,
        protocol,
        fov=params.param_float(prot, UIParam.FOV) * 1e-3,
        blade_width=max(4, round(params.user_float(prot, 0, 16.0))),
        n_blades=None if requested <= 0 else requested,
        angle_scheme="golden" if round(params.user_float(prot, 2, 0.0)) else "uniform",
    )


PLUGIN = SePropeller2D()


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
    ("--fov-mm", UIParam.FOV, float, "In-plane FOV [mm]"),
    ("--slice-thickness-mm", UIParam.SLICE_THICKNESS, float, "Slice thickness [mm]"),
    ("--slice-spacing-mm", UIParam.SLICE_SPACING, float, "Slice spacing [mm]"),
    ("--nx", UIParam.NX, int, "In-plane matrix size"),
    ("--nslices", UIParam.NSLICES, int, "Number of slices"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    ("--offset-y-mm", UIParam.FOV_OFFSET_Y, float, "Volume offset along phase encode [mm]"),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slice [mm]"),
    ("--blade-width", UIParam.user_value(0), float, "Phase-encode lines per blade"),
    ("--blades", UIParam.user_value(1), float, "Blades in the set (0 = Nyquist)"),
    ("--angles", UIParam.user_value(2), float, "Angle scheme: 0 uniform, 1 golden"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate a 2D PROPELLER spin-echo .seq offline.",
            default_output="se_propeller_2d.seq",
        )
    )
