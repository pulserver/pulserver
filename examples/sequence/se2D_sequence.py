"""2D Cartesian spin echo, multi-slice.

The refocusing k-flip in its simplest setting: a slice-selective SLR
excitation, one SLR 180 with bridged crushers a half-TE later, and one
frequency-encoded line read out at the echo -- :class:`design.LineReadout2D`
opened by the refocusing pulse, exactly as its docstring prescribes. Phase
encoding may be undersampled with a fully sampled autocalibration block and
truncated by partial Fourier; the readout may be a partial echo.
:mod:`pulserver.app.recon.cartesian2D_recon` reads all three back.

TE spans excitation centre to echo, with the 180 at its midpoint: the readout
solves the second half (``te = TE/2`` from the refocusing pulse), and the
scan loop solves the first with a delay between the excitation's rephaser and
the 180. Slices are dealt into passes exactly as the gradient echo deals
them, and every pulse of a repetition is offset to its slice.

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

#: SLR design shared by the excitation and the refocusing pulse. The
#: selection amplitude is ``time_bw_product / (duration * thickness)``, so
#: designing both pulses with the same numbers is what lets one slice offset
#: frequency serve them both.
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


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "se_2d.seq",
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
    te: float | None = 15e-3,
    tr: float | None = 500e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_echo: float = 1.0,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    n_averages: int = 1,
    n_dummy: int = 0,
    n_gain_calibration_readouts: int | None = None,
    crusher_cycles: float = 4.0,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Create a 2D Cartesian spin-echo sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'se_2d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float or tuple of float, optional
        In-plane field of view in meters. If a single value, it is used for
        both x and y. If a tuple, it is (fov_x, fov_y). Default is 220e-3.
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
        Order the slices of one pass are excited in, as accepted by
        `pp.calc_traversal_order`. Default is 'interleaved'.
    te : float or None, optional
        Echo time in seconds, excitation centre to echo, with the refocusing
        pulse at its midpoint. None is as short as possible. Default is
        15e-3.
    tr : float or None, optional
        Repetition time in seconds, between successive excitations of the
        same slice. None is as short as possible, and puts every slice in one
        pass. Default is 500e-3.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    partial_echo : float, optional
        Fraction of the full echo acquired, in (0.5, 1]. Default is 1.0.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired, in (0.5, 1]. Default
        is 1.0.
    acceleration : int, optional
        Uniform phase-encode undersampling factor. Default is 1.
    n_acs : int, optional
        Number of fully sampled autocalibration lines at the center of
        k-space, acquired ahead of the rest of the scan. Default is 24.
    n_averages : int, optional
        How many times the scan is acquired, written into the block table.
        Default is 1.
    n_dummy : int, optional
        Repetitions played without acquiring before the first line of each
        pass. A spin echo at a full-relaxation TR starts at equilibrium, so
        the default is 0.
    n_gain_calibration_readouts : int or None, optional
        Written as the ``NumGainCalibrationReadouts`` definition. None is one
        per slice. Default is None.
    crusher_cycles : float, optional
        Cycles of dephasing each crusher beside the refocusing pulse winds
        across one voxel. Default is 4.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left on the readout axis at the end of each
        repetition. Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The spin-echo sequence object.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    kernel = SE2DKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        slice_order=slice_order,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        partial_echo=partial_echo,
        partial_fourier=partial_fourier,
        acceleration=acceleration,
        n_acs=n_acs,
        n_averages=n_averages,
        n_dummy=n_dummy,
        crusher_cycles=crusher_cycles,
        spoiling_cycles=spoiling_cycles,
    )
    excitation = kernel.excitation
    refocusing = kernel.refocusing
    fov_x, fov_y = kernel.fov
    sampled_lines = kernel.sampled_lines

    acs_start = max(0, n_y // 2 - n_acs // 2)
    acs_stop = min(n_y, acs_start + n_acs)
    last_calibration_line = kernel.n_calibration - 1
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )

    seq = pp.Sequence(system)

    def repetition(
        readout, timing, slices, ky: float, acquire: bool, mark=None
    ) -> None:
        """Play one TR of every slice of a pass, acquiring or not."""
        wait_te = getattr(readout, "wait_te", None)
        for i_slice in slices:
            # `slice_positions` is in ascending order, so the loop index is the
            # geometric index a reconstruction stacks by.
            slc_label.value = i_slice
            position = slice_positions[i_slice]
            exc_hz = excitation.gz.amplitude * position
            ref_hz = kernel.refocusing_amplitude * position

            # Both pulses select the same slice, each at its own gradient.
            # Phase is zeroed at each pulse's own centre; the 180 keeps its
            # CPMG quarter turn on top, and the echo then forms along the
            # excitation's axis, so the ADC needs no offset of its own.
            excitation.rf.freq_offset = exc_hz
            excitation.rf.phase_offset = -2 * np.pi * exc_hz * excitation.rf.center
            refocusing.rf_ref.freq_offset = ref_hz
            refocusing.rf_ref.phase_offset = (
                np.pi / 2 - 2 * np.pi * ref_hz * refocusing.rf_ref.center
            )

            seq.add_block(
                excitation.rf, excitation.gz, *([mark] if mark is not None else [])
            )
            mark = None
            seq.add_block(excitation.gz_reph)
            if timing.wait_half_te is not None:
                seq.add_block(timing.wait_half_te)
            seq.add_block(readout.rf, readout.gz)
            if wait_te is not None:
                seq.add_block(wait_te)
            seq.add_block(readout.gx_pre, pp.scale_grad(readout.gy_pre, ky))
            if acquire:
                seq.add_block(readout.gx, readout.adc, *readout.adc_labels)
            else:
                seq.add_block(readout.gx)
            seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, ky))
            if timing.wait_tr is not None:
                seq.add_block(timing.wait_tr)

    for slices in kernel.passes:
        readout, timing = kernel.repetitions[len(slices)]
        lin_label, slc_label, ima_label, seg_label = readout.adc_labels

        for i_dummy in range(n_dummy):
            repetition(
                readout,
                timing,
                slices,
                0.0,
                acquire=False,
                mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
            )

        clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
        for i_phase, line in enumerate(sampled_lines):
            ky = (line - n_y / 2) / (n_y / 2)
            lin_label.value = line
            ima_label.value = int(acs_start <= line < acs_stop)
            seg_label.value = int(i_phase > last_calibration_line)

            repetition(readout, timing, slices, ky, acquire=True, mark=clear_once)
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
    seq.set_definition(key="Name", value="se_2d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=n_slices
        if n_gain_calibration_readouts is None
        else n_gain_calibration_readouts,
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

    seq = pp.tile(seq, n_averages, in_place=True)

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def SE2DKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_order: str = "interleaved",
    te: float | None = 15e-3,
    tr: float | None = 500e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_echo: float = 1.0,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    n_averages: int = 1,
    n_dummy: int = 0,
    crusher_cycles: float = 4.0,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """Design the repetitions, and the plan that repeats them.

    The echo time is solved in two halves around the refocusing pulse. The
    readout owns the second -- ``te = TE/2`` from the 180's centre to the
    echo -- and the first is a delay between the excitation's rephaser and
    the 180, sized so the two centres sit ``TE/2`` apart. A requested TE
    shorter than either half admits raises ``ValueError``, which is what
    makes building the modules the feasibility check.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_slices, slice_thickness, slice_order, te, tr, \
readout_bandwidth_hz, partial_echo, partial_fourier, acceleration, n_acs, \
n_averages, n_dummy, crusher_cycles, spoiling_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``refocusing``, ``refocusing_amplitude`` (Hz/m, for
        per-slice offsets on the 180), ``repetitions`` (``(readout, timing)``
        keyed by pass size), ``passes``, ``fov``, ``sampled_lines``,
        ``n_calibration``, ``n_averages``, ``echo_time``,
        ``repetition_time``, ``bandwidth_hz`` and ``duration``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = design.SpatialSelectiveExcitation(
        system,
        90.0,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
    )
    refocusing = design.SpatialSelectiveRefocusing(
        system,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
        spoiling_cycles=crusher_cycles,
    )
    refocusing_amplitude = TIME_BW_PRODUCT / (PULSE_DURATION * slice_thickness)

    # The first half of TE runs from the excitation's centre, through the
    # rest of its blocks, to the 180's centre. What it can never be shorter
    # than is that path with no delay inserted.
    exc_center = excitation.rf.delay + excitation.rf.center
    pre_refocusing = (excitation.seq.duration()[0] - exc_center) + refocusing.center
    half_te_floor = pre_refocusing

    def repetition(half_te: float | None, module_tr: float | None):
        """One readout and the delays that place it, for one pass size."""
        readout = design.LineReadout2D(
            system,
            refocusing.rf_ref,
            refocusing.gz,
            fov=(fov_x, fov_y),
            matrix=(n_x, n_y),
            te=half_te,
            partial_echo=partial_echo,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            labels=("LIN", "SLC", "IMA", "SEG"),
        )
        achieved_half = readout.echo_time
        if half_te is not None and achieved_half > half_te + 1e-9:
            raise ValueError(
                f"TE {2 * half_te * 1e3:.2f} ms is shorter than the readout "
                f"half admits; the minimum is {2 * achieved_half * 1e3:.2f} ms"
            )

        wait = max(0.0, achieved_half - half_te_floor)
        if half_te is not None and half_te_floor > achieved_half + 1e-9:
            raise ValueError(
                f"TE {2 * half_te * 1e3:.2f} ms is shorter than the "
                f"excitation half admits; the minimum is "
                f"{2 * half_te_floor * 1e3:.2f} ms"
            )
        wait_half_te = (
            pp.make_delay(pp.round_to_raster(wait, system.block_duration_raster))
            if wait > 0
            else None
        )

        length = (
            excitation.seq.duration()[0]
            + (wait_half_te.delay if wait_half_te is not None else 0.0)
            + readout.duration
        )
        wait_tr = None
        if module_tr is not None:
            if module_tr < length - 1e-9:
                raise ValueError(
                    f"TR {module_tr * 1e3:.1f} ms is shorter than one "
                    f"repetition takes ({length * 1e3:.1f} ms)"
                )
            pad = pp.round_to_raster(module_tr - length, system.block_duration_raster)
            if pad > 0:
                wait_tr = pp.make_delay(pad)
                length = length + pad
        return readout, SimpleNamespace(
            wait_half_te=wait_half_te, wait_tr=wait_tr, length=length
        )

    # The half the readout owns cannot be shorter than either its own floor
    # or the excitation side's, so solve the minimum once and hold both
    # halves to it.
    shortest, _ = repetition(None, None)
    half_te = max(shortest.echo_time, half_te_floor) if te is None else te / 2
    readout_probe, timing_probe = repetition(half_te, None)

    per_pass = n_slices if tr is None else max(1, int(tr / timing_probe.length))
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
            (readout_probe, timing_probe)
            if tr is None
            else repetition(half_te, tr / size)
        )
        for size in {len(group) for group in passes}
    }

    sampled_lines = pp.calc_sampled_lines(
        n_y,
        acceleration,
        n_acs,
        order="calibration_first",
        partial_fourier=partial_fourier,
    )
    n_calibration = len(
        pp.calc_calibration_lines(n_y, n_acs, partial_fourier=partial_fourier)
    )

    pass_time = sum(len(group) * repetitions[len(group)][1].length for group in passes)
    duration = n_dummy * pass_time + n_averages * len(sampled_lines) * pass_time

    return SimpleNamespace(
        excitation=excitation,
        refocusing=refocusing,
        refocusing_amplitude=refocusing_amplitude,
        repetitions=repetitions,
        passes=passes,
        fov=(fov_x, fov_y),
        sampled_lines=sampled_lines,
        n_calibration=n_calibration,
        n_averages=n_averages,
        echo_time=2 * half_te,
        repetition_time=max(
            len(group) * repetitions[len(group)][1].length for group in passes
        ),
        bandwidth_hz=readout_probe.bandwidth_hz,
        duration=duration,
    )


# ======================================================================
# The scanner protocol contract
# ======================================================================


class Se2D(SequencePlugin):
    """The 2D spin echo behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=15.0,
                    min=5.0,
                    max=200.0,
                    incr=0.1,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 15.0, 30.0, 60.0, 90.0],
                ),
                UIParam.TR: DropdownFloatParam(
                    value=500.0,
                    min=50.0,
                    max=10000.0,
                    incr=1.0,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 300.0, 500.0, 2000.0, 4000.0],
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
                UIParam.FOV_OFFSET_X: OffFloatParam(
                    value=0.0, min=-500.0, max=500.0, unit="mm"
                ),
                UIParam.FOV_OFFSET_Y: OffFloatParam(
                    value=0.0, min=-500.0, max=500.0, unit="mm"
                ),
                UIParam.FOV_OFFSET_Z: OffFloatParam(
                    value=0.0, min=-500.0, max=500.0, unit="mm"
                ),
                UIParam.user_name(0): Description(text="ACS lines"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=24.0,
                    min=0.0,
                    max=512.0,
                    incr=1.0,
                    unit="lines",
                ),
                UIParam.user_name(1): Description(text="Partial echo"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=1.0,
                    min=0.75,
                    max=1.0,
                    incr=0.05,
                    unit="",
                ),
                UIParam.user_name(2): Description(text="Dummy scans"),
                UIParam.user_value(2): TypeinFloatParam(
                    value=0.0,
                    min=0.0,
                    max=128.0,
                    incr=1.0,
                    unit="TR",
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
        kwargs = protocol_kwargs(system, protocol)
        try:
            kernel = SE2DKernel(
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
                f"TA = {kernel.duration:.1f} s at "
                f"{kernel.bandwidth_hz * 1e-3:.1f} kHz, {len(kernel.passes)} pass(es)"
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
        "n_y",
        "n_slices",
        "slice_thickness",
        "slice_order",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "partial_echo",
        "partial_fourier",
        "acceleration",
        "n_acs",
        "n_averages",
        "n_dummy",
        "crusher_cycles",
        "spoiling_cycles",
    )
)


def protocol_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
    """The prescribed quantities, plus this sequence's own user slots."""
    prot = dict_to_protocol(protocol)
    return main_kwargs(
        main,
        system,
        protocol,
        partial_echo=params.user_float(prot, 1, 1.0),
        partial_fourier=params.user_float(prot, 3, 1.0),
        n_acs=params.acs_lines_from_protocol(
            prot, params.param_int(prot, UIParam.NY), 0
        ),
        n_dummy=max(0, round(params.user_float(prot, 2, 0.0))),
    )


PLUGIN = Se2D()


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
    (
        "--offset-y-mm",
        UIParam.FOV_OFFSET_Y,
        float,
        "Volume offset along phase encode [mm]",
    ),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slice [mm]"),
    ("--acs-lines", UIParam.user_value(0), float, "Number of ACS lines"),
    (
        "--partial-echo",
        UIParam.user_value(1),
        float,
        "Acquired echo fraction in (0.5, 1]",
    ),
    (
        "--partial-fourier",
        UIParam.user_value(3),
        float,
        "Acquired phase-encode fraction in (0.5, 1]",
    ),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=ARG_MAP,
            description="Generate a 2D Cartesian spin-echo .seq offline.",
            default_output="se_2d.seq",
        )
    )
