"""2D Cartesian fast spin echo, multi-slice.

A CPMG train per excitation -- :class:`design.FseReadout2D`: one SLR 90, then
``etl`` refocusing pulses, each echo phase-encoded by its own trapezoid pair
and unwound before the next 180, which is what survives the train's sign
flips. The phase-encode lines are dealt across shots so that the centre of
k-space is acquired at the echo whose time is the requested effective TE --
the rolled-linear ordering, delegated to :func:`pulserver.pypulseq.make_linear_order`,
whose wrap every product FSE accepts for the same reason. Slices are dealt into passes exactly as the gradient echo deals
them. :mod:`pulserver.reczoo.fse_2d` reads the result back.

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

#: SLR design shared by the excitation and the refocusing pulses, so one
#: slice-offset frequency serves both (the selection amplitude is
#: ``time_bw_product / (duration * thickness)``).
_PULSE_DURATION = 3e-3
_TIME_BW_PRODUCT = 4.0

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
    seq_filename: str = "fse_2d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    slice_order: str = "interleaved",
    etl: int = 8,
    te: float | None = 80e-3,
    tr: float | None = 2000e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    n_averages: int = 1,
    n_dummy: int = 0,
    n_gain_calibration_readouts: int | None = None,
    crusher_cycles: float = 4.0,
    readout_crusher_cycles: float = 0.0,
) -> pp.Sequence:
    """Create a 2D Cartesian fast spin-echo sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'fse_2d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float or tuple of float, optional
        In-plane field of view in meters, (fov_x, fov_y) if a tuple. Default
        is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters along the logical
        readout, phase and slice axes. Default is (0.0, 0.0, 0.0).
    n_x : int, optional
        Number of readout samples. Default is 128.
    n_y : int, optional
        Number of phase encoding steps. Default is 128.
    n_slices : int, optional
        Number of slices. Default is 1.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 5e-3.
    slice_gap : float, optional
        Gap between adjacent slices in meters. Default is 0.0.
    slice_order : str, optional
        Order the slices of one pass are excited in. Default is
        'interleaved'.
    etl : int, optional
        Echo train length: lines per excitation. Default is 8.
    te : float or None, optional
        Effective echo time in seconds -- the echo at which the centre of
        k-space is acquired, rounded to the train's echo spacing. None puts
        the centre on the first echo. Default is 80e-3.
    tr : float or None, optional
        Repetition time in seconds, between successive excitations of the
        same slice. Default is 2000e-3.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired, in (0.5, 1]. Default
        is 1.0.
    acceleration : int, optional
        Uniform phase-encode undersampling factor. Default is 1.
    n_acs : int, optional
        Number of fully sampled autocalibration lines. Default is 24.
    n_averages : int, optional
        How many times the scan is acquired, written into the block table.
        Default is 1.
    n_dummy : int, optional
        Trains played without acquiring before the first shot of each pass.
        A long-TR spin echo starts at equilibrium, so the default is 0.
    n_gain_calibration_readouts : int or None, optional
        Written as the ``NumGainCalibrationReadouts`` definition. None is
        one per slice. Default is None.
    crusher_cycles : float, optional
        Cycles of dephasing each crusher beside every refocusing pulse winds
        across one voxel. Default is 4.0.
    readout_crusher_cycles : float, optional
        Read-axis crushing each side of every acquisition, in cycles across
        one voxel. Default is 0.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The FSE sequence object.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    kernel = FSE2DKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        slice_order=slice_order,
        etl=etl,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        partial_fourier=partial_fourier,
        acceleration=acceleration,
        n_acs=n_acs,
        n_averages=n_averages,
        n_dummy=n_dummy,
        crusher_cycles=crusher_cycles,
        readout_crusher_cycles=readout_crusher_cycles,
    )
    excitation = kernel.excitation
    fov_x, fov_y = kernel.fov
    shots = kernel.shots

    acs_start = max(0, n_y // 2 - n_acs // 2)
    acs_stop = min(n_y, acs_start + n_acs)
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )

    seq = pp.Sequence(system)

    def train(fse, timing, slices, lines, acquire: bool, mark=None) -> None:
        """Play one shot's train for every slice of a pass."""
        lin_label, slc_label, ima_label, seg_label = fse.adc_labels
        seg_label.value = 0
        for i_slice in slices:
            # `slice_positions` is in ascending order, so the loop index is the
            # geometric index a reconstruction stacks by.
            slc_label.value = i_slice
            position = slice_positions[i_slice]
            exc_hz = excitation.gz.amplitude * position
            ref_hz = kernel.refocusing_amplitude * position
            fse.rf.freq_offset = exc_hz
            fse.rf.phase_offset = -2 * np.pi * exc_hz * fse.rf.center
            fse.rf_ref.freq_offset = ref_hz
            fse.rf_ref.phase_offset = (
                np.pi / 2 - 2 * np.pi * ref_hz * fse.rf_ref.center
            )

            seq.add_block(fse.rf, fse.gz, *([mark] if mark is not None else []))
            mark = None
            seq.add_block(fse.gx_pre, fse.gz_reph)
            for echo, line in enumerate(lines):
                ky = 0.0 if line is None else (line - n_y / 2) / (n_y / 2)
                seq.add_block(fse.rf_ref, fse.gz_ref)
                if echo == 0 and fse.esp_first > fse.esp:
                    seq.add_block(fse.wait_esp1)
                seq.add_block(fse.gx_bridge_pre, pp.scale_grad(fse.gy_pre, ky))
                if acquire and line is not None:
                    lin_label.value = line
                    ima_label.value = int(acs_start <= line < acs_stop)
                    seq.add_block(fse.gx, fse.adc, *fse.adc_labels)
                else:
                    seq.add_block(fse.gx)
                seq.add_block(fse.gx_bridge_post, pp.scale_grad(fse.gy_rew, ky))
            if timing.wait_tr is not None:
                seq.add_block(timing.wait_tr)

    for slices in kernel.passes:
        fse, timing = kernel.repetitions[len(slices)]

        for i_dummy in range(n_dummy):
            train(
                fse,
                timing,
                slices,
                [None] * etl,
                acquire=False,
                mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
            )

        clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
        for lines in shots:
            train(fse, timing, slices, lines, acquire=True, mark=clear_once)
            clear_once = None

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset), system=system
    ).apply_to_sequence(seq, in_place=True)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap
    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_slices])
    seq.set_definition(key="Name", value="fse_2d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(key="EchoSpacing", value=kernel.echo_spacing)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=n_slices if n_gain_calibration_readouts is None else n_gain_calibration_readouts,
    )

    seq.set_definition(key="kSpaceCenterLine", value=n_y // 2)
    seq.set_definition(
        key="kSpaceCenterSample",
        value=kernel.repetitions[len(kernel.passes[0])][0].center_sample,
    )
    seq.set_definition(key="SlicePositions", value=slice_positions.tolist())
    seq.set_definition(key="SliceThickness", value=kernel.excitation.slice_thickness)
    seq.set_definition(
        key="SliceGap",
        value=slice_thickness + slice_gap - kernel.excitation.slice_thickness,
    )

    seq.expand_repeats(n_averages)

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


def FSE2DKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_order: str = "interleaved",
    etl: int = 8,
    te: float | None = 80e-3,
    tr: float | None = 2000e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    n_averages: int = 1,
    n_dummy: int = 0,
    crusher_cycles: float = 4.0,
    readout_crusher_cycles: float = 0.0,
) -> SimpleNamespace:
    """Design the train, and the plan that repeats it.

    Building :class:`design.FseReadout2D` *is* the feasibility check; the
    requested effective TE is then rounded onto the train's echo grid and
    the lines rolled -- by :func:`pulserver.pypulseq.make_linear_order` -- so
    the centre of k-space is acquired at that echo.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_slices, slice_thickness, slice_order, etl, te, tr, \
readout_bandwidth_hz, partial_fourier, acceleration, n_acs, n_averages, \
n_dummy, crusher_cycles, readout_crusher_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``refocusing``, ``refocusing_amplitude``,
        ``repetitions`` (``(readout, timing)`` keyed by pass size),
        ``passes``, ``fov``, ``shots`` (lines per shot, indexed by echo),
        ``n_center`` (zero-based echo of the k-space centre),
        ``echo_spacing``, ``echo_time``, ``repetition_time``,
        ``bandwidth_hz``, ``n_averages`` and ``duration``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

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

    def build(module_tr: float | None):
        fse = design.FseReadout2D(
            system,
            excitation.rf,
            excitation.gz,
            excitation.gz_reph,
            rf_ref=refocusing.rf_ref,
            gz_ref=refocusing.gz,
            fov=(fov_x, fov_y),
            matrix=(n_x, n_y),
            etl=etl,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=readout_crusher_cycles,
            labels=("LIN", "SLC", "IMA", "SEG"),
        )
        length = fse.seq.duration()[0]
        wait_tr = None
        if module_tr is not None:
            if module_tr < length - 1e-9:
                raise ValueError(
                    f"TR {module_tr * 1e3:.1f} ms is shorter than one train "
                    f"takes ({length * 1e3:.1f} ms)"
                )
            pad = pp.round_to_raster(
                module_tr - length, system.block_duration_raster
            )
            if pad > 0:
                wait_tr = pp.make_delay(pad)
                length += pad
        return fse, SimpleNamespace(wait_tr=wait_tr, length=length)

    shortest, shortest_timing = build(None)
    esp = shortest.esp

    # The requested effective TE, rounded onto the echo grid the train has.
    if te is None:
        n_center = 0
    else:
        n_center = int(np.clip(round(te / esp) - 1, 0, etl - 1))
    echo_time = float(shortest.echo_times[n_center])

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
        size: (
            (shortest, shortest_timing) if tr is None else build(tr / size)
        )
        for size in {len(group) for group in passes}
    }

    sampled_lines = pp.calc_sampled_lines(
        n_y, acceleration, n_acs, partial_fourier=partial_fourier
    )
    # Deal the ky lines into echo trains with the k-space centre placed at echo
    # ``n_center`` (effective-TE control), delegated to the linear-ordering
    # builtin, whose shots of indices we map back onto the sampled lines.
    shots = pp.make_linear_order(
        sampled_lines, etl, center=(n_y / 2, 0.0), center_echo=n_center, pad=True
    )
    shots = [
        [sampled_lines[i] if i is not None else None for i in shot] for shot in shots
    ]

    pass_time = sum(
        len(group) * repetitions[len(group)][1].length for group in passes
    )
    duration = (n_dummy + n_averages * len(shots)) * pass_time

    return SimpleNamespace(
        excitation=excitation,
        refocusing=refocusing,
        refocusing_amplitude=refocusing_amplitude,
        repetitions=repetitions,
        passes=passes,
        fov=(fov_x, fov_y),
        shots=shots,
        n_center=n_center,
        echo_spacing=esp,
        echo_time=echo_time,
        repetition_time=max(
            len(group) * repetitions[len(group)][1].length for group in passes
        ),
        bandwidth_hz=shortest.bandwidth_hz,
        n_averages=n_averages,
        duration=duration,
    )


class Fse2D(SequencePlugin):
    """The 2D fast spin echo behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=80.0,
                    min=8.0,
                    max=400.0,
                    incr=1.0,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 40.0, 80.0, 100.0, 120.0],
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
                UIParam.PHASE_FOV: DropdownFloatParam(
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
                    value=1,
                    min=1,
                    max=128,
                    incr=1,
                    options=[1, 5, 10, 20, 40],
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
                UIParam.NEX: DropdownFloatParam(
                    value=1.0,
                    min=1.0,
                    max=32.0,
                    incr=1.0,
                    unit="",
                    options=[1.0, 2.0, 4.0, 8.0, 16.0],
                ),
                UIParam.FOV_OFFSET_X: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Y: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Z: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.user_name(0): Description(text="ACS lines"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=24.0,
                    min=0.0,
                    max=512.0,
                    incr=1.0,
                    unit="lines",
                ),
                UIParam.user_name(1): Description(text="Echo train length"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=8.0,
                    min=1.0,
                    max=64.0,
                    incr=1.0,
                    unit="",
                ),
                UIParam.user_name(3): Description(text="Partial Fourier"),
                UIParam.user_value(3): TypeinFloatParam(
                    value=1.0,
                    min=0.75,
                    max=1.0,
                    incr=0.05,
                    unit="",
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = _main_kwargs(system, protocol)
        try:
            kernel = FSE2DKernel(
                system,
                **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s, "
                f"TEeff = {kernel.echo_time * 1e3:.1f} ms at echo "
                f"{kernel.n_center + 1}/{len(kernel.shots[0])}, "
                f"ESP = {kernel.echo_spacing * 1e3:.2f} ms"
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
        "n_slices",
        "slice_thickness",
        "slice_order",
        "etl",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "partial_fourier",
        "acceleration",
        "n_acs",
        "n_averages",
        "n_dummy",
        "crusher_cycles",
        "readout_crusher_cycles",
    )
)


def _main_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
    """The prescribed quantities, plus this sequence's own user slots."""
    prot = dict_to_protocol(protocol)
    return main_kwargs(
        main,
        system,
        protocol,
        etl=max(1, round(params.user_float(prot, 1, 8.0))),
        partial_fourier=params.user_float(prot, 3, 1.0),
        n_acs=params.acs_lines_from_protocol(prot, params.param_int(prot, UIParam.NY), 0),
    )


PLUGIN = Fse2D()


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
    ("--slice-thickness-mm", UIParam.SLICE_THICKNESS, float, "Slice thickness [mm]"),
    ("--slice-spacing-mm", UIParam.SLICE_SPACING, float, "Slice spacing [mm]"),
    ("--nx", UIParam.NX, int, "Readout matrix size"),
    ("--ny", UIParam.NY, int, "Phase-encode matrix size"),
    ("--nslices", UIParam.NSLICES, int, "Number of slices"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--ry", UIParam.RY, float, "Phase-encode undersampling factor"),
    ("--nex", UIParam.NEX, float, "Number of signal averages"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    ("--offset-y-mm", UIParam.FOV_OFFSET_Y, float, "Volume offset along phase encode [mm]"),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slice [mm]"),
    ("--acs-lines", UIParam.user_value(0), float, "Number of ACS lines"),
    ("--etl", UIParam.user_value(1), float, "Echo train length"),
    ("--partial-fourier", UIParam.user_value(3), float, "Acquired phase-encode fraction in (0.5, 1]"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate a 2D Cartesian fast spin-echo .seq offline.",
            default_output="fse_2d.seq",
        )
    )
