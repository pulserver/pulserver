"""RF-spoiled multi-echo 2D Cartesian gradient echo, multi-slice.

The echo-train readout: every repetition reads its line at each of
``n_echoes`` echo times, monopolar (a flyback rewinder between echoes, every
echo read the same way) or bipolar (alternating sign, faster, even echoes
read backwards). Each acquisition carries its echo index as ``ECO``, which is
the counter :mod:`pulserver.reczoo.gre_multiecho_2d` reconstructs per-echo
images from. Everything else -- slices and passes, autocalibration block,
partial Fourier, spoiling -- is :mod:`pulserver.seqzoo.gre_2d`.

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
    seq_filename: str = "gre_multiecho_2d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    n_echoes: int = 4,
    monopolar: bool = True,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    slice_order: str = "interleaved",
    flip_angle_deg: float = 15.0,
    te: float | None = None,
    tr: float | None = 50e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    n_averages: int = 1,
    n_dummy: int = 16,
    n_gain_calibration_readouts: int | None = None,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Create an RF-spoiled multi-echo 2D Cartesian gradient-echo sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'gre_multiecho_2d.seq'.
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
    n_echoes : int, optional
        Echoes per repetition. Default is 4.
    monopolar : bool, optional
        Rewind between echoes so every one is read the same way. False
        alternates the readout sign instead. Default is True.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 5e-3.
    slice_gap : float, optional
        Gap between adjacent slices in meters. Default is 0.0.
    slice_order : str, optional
        Order the slices of one pass are excited in. Default is
        'interleaved'.
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 15.0.
    te : float or None, optional
        First echo time in seconds; the rest follow at the train's echo
        spacing. None is as short as possible. Default is None.
    tr : float or None, optional
        Repetition time in seconds, between successive excitations of the
        same slice. Default is 50e-3.
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
        Repetitions played without acquiring, before the first line of each
        pass. Default is 16.
    n_gain_calibration_readouts : int or None, optional
        Written as the ``NumGainCalibrationReadouts`` definition. None is
        one per slice. Default is None.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment in degrees. Default is 117.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left on the readout axis at the end of each
        repetition. Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The multi-echo GRE sequence object.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    kernel = MultiechoKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_slices=n_slices,
        n_echoes=n_echoes,
        monopolar=monopolar,
        slice_thickness=slice_thickness,
        slice_order=slice_order,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        partial_fourier=partial_fourier,
        acceleration=acceleration,
        n_acs=n_acs,
        n_averages=n_averages,
        n_dummy=n_dummy,
        spoiling_cycles=spoiling_cycles,
    )
    excitation = kernel.excitation
    fov_x, fov_y = kernel.fov
    sampled_lines = kernel.sampled_lines

    acs_start = max(0, n_y // 2 - n_acs // 2)
    acs_stop = min(n_y, acs_start + n_acs)
    last_calibration_line = kernel.n_calibration - 1
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )
    rf_phases = pp.make_rf_spoiling_schedule(
        (len(sampled_lines) + n_dummy) * n_slices,
        increment=np.deg2rad(rf_spoiling_increment_deg),
    )

    seq = pp.Sequence(system)
    spoiling_phase = iter(rf_phases)

    def repetition(readout, slices, ky: float, acquire: bool, mark=None) -> None:
        """Play one TR of every slice of a pass, acquiring or not."""
        wait_te = getattr(readout, "wait_te", None)
        wait_tr = getattr(readout, "wait_tr", None)
        ima_label, seg_label, eco_label = readout.adc_labels
        for i_slice in slices:
            rf_phase = next(spoiling_phase)
            readout.rf.freq_offset = excitation.gz.amplitude * slice_positions[i_slice]
            readout.rf.phase_offset = (
                rf_phase - 2 * np.pi * readout.rf.freq_offset * readout.rf.center
            )
            readout.adc.phase_offset = rf_phase

            seq.add_block(readout.rf, readout.gz, *([mark] if mark is not None else []))
            mark = None
            if wait_te is not None:
                seq.add_block(wait_te, readout.gz_reph)
                seq.add_block(readout.gx_pre, pp.scale_grad(readout.gy_pre, ky))
            else:
                seq.add_block(
                    readout.gx_pre,
                    pp.scale_grad(readout.gy_pre, ky),
                    readout.gz_reph,
                )
            for i_echo in range(n_echoes):
                if monopolar and i_echo:
                    seq.add_block(readout.gx_flyback)
                lobe = (
                    readout.gx
                    if monopolar or i_echo % 2 == 0
                    else readout.gx_rev
                )
                if acquire:
                    eco_label.value = i_echo
                    seq.add_block(lobe, readout.adc, ima_label, seg_label, eco_label)
                else:
                    seq.add_block(lobe)
            seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, ky))
            if wait_tr is not None:
                seq.add_block(wait_tr)

    for slices in kernel.passes:
        readout = kernel.readouts[len(slices)]
        ima_label, seg_label, _ = readout.adc_labels

        for i_dummy in range(n_dummy):
            repetition(
                readout,
                slices,
                0.0,
                acquire=False,
                mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
            )

        clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
        for i_phase, line in enumerate(sampled_lines):
            ky = (line - n_y / 2) / (n_y / 2)
            ima_label.value = int(acs_start <= line < acs_stop)
            seg_label.value = int(i_phase > last_calibration_line)

            repetition(readout, slices, ky, acquire=True, mark=clear_once)
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
    seq.set_definition(key="Name", value="gre_multiecho_2d")
    seq.set_definition(key="TE", value=kernel.echo_times)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=n_slices if n_gain_calibration_readouts is None else n_gain_calibration_readouts,
    )

    seq.auto_label()
    seq.expand_repeats(n_averages)

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


def MultiechoKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    n_echoes: int = 4,
    monopolar: bool = True,
    slice_thickness: float = 5e-3,
    slice_order: str = "interleaved",
    flip_angle_deg: float = 15.0,
    te: float | None = None,
    tr: float | None = 50e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    n_averages: int = 1,
    n_dummy: int = 16,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """Design the repetitions, and the plan that repeats them.

    :func:`pulserver.seqzoo.gre_2d.GREKernel` with an echo train: the readout
    carries ``n_echoes`` and the train's polarity, and the echo times are
    read off the built blocks -- first echo at ``readout.echo_time``, the
    rest one echo spacing apart, where the spacing is what the readout lobe
    (plus, monopolar, its flyback) actually takes.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_slices, n_echoes, monopolar, slice_thickness, \
slice_order, flip_angle_deg, te, tr, readout_bandwidth_hz, partial_fourier, \
acceleration, n_acs, n_averages, n_dummy, spoiling_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        As :func:`pulserver.seqzoo.gre_2d.GREKernel` returns, with
        ``echo_times`` (one per echo) in place of ``echo_time``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = design.SpatialSelectiveExcitation(
        system, flip_angle_deg, slice_thickness
    )

    def readout(module_tr: float | None):
        return design.LineReadout2D(
            system,
            excitation.rf,
            excitation.gz,
            excitation.gz_reph,
            fov=(fov_x, fov_y),
            matrix=(n_x, n_y),
            te=te,
            tr=module_tr,
            n_echoes=n_echoes,
            flyback=monopolar,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            labels=("IMA", "SEG", "ECO"),
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

    # The spacing between echoes is what the built blocks take: one readout
    # lobe, plus the flyback a monopolar train rewinds with.
    echo_spacing = pp.calc_duration(shortest.gx)
    if monopolar and n_echoes > 1:
        echo_spacing += pp.calc_duration(shortest.gx_flyback)
    echo_times = [
        shortest.echo_time + i_echo * echo_spacing for i_echo in range(n_echoes)
    ]

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

    pass_time = sum(len(group) * readouts[len(group)].duration for group in passes)
    duration = n_dummy * pass_time + n_averages * len(sampled_lines) * pass_time

    return SimpleNamespace(
        excitation=excitation,
        readouts=readouts,
        passes=passes,
        fov=(fov_x, fov_y),
        sampled_lines=sampled_lines,
        n_calibration=n_calibration,
        n_averages=n_averages,
        echo_times=echo_times,
        repetition_time=max(
            len(group) * readouts[len(group)].duration for group in passes
        ),
        bandwidth_hz=shortest.bandwidth_hz,
        duration=duration,
    )


class GreMultiecho2D(SequencePlugin):
    """The multi-echo 2D gradient echo behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=-1.0,
                    min=-1.0,
                    max=80.0,
                    incr=0.1,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 3.0, 5.0, 8.0, 15.0],
                ),
                UIParam.TR: DropdownFloatParam(
                    value=50.0,
                    min=5.0,
                    max=5000.0,
                    incr=0.1,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 50.0, 100.0, 250.0, 500.0],
                ),
                UIParam.FLIP: DropdownFloatParam(
                    value=15.0,
                    min=1.0,
                    max=90.0,
                    incr=1.0,
                    unit="deg",
                    options=[5.0, 15.0, 30.0, 60.0, 90.0],
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
                UIParam.user_name(1): Description(text="Echoes"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=4.0,
                    min=1.0,
                    max=16.0,
                    incr=1.0,
                    unit="",
                ),
                UIParam.user_name(2): Description(text="Dummy scans"),
                UIParam.user_value(2): TypeinFloatParam(
                    value=16.0,
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
                UIParam.user_name(4): Description(text="Monopolar train"),
                UIParam.user_value(4): TypeinFloatParam(
                    value=1.0,
                    min=0.0,
                    max=1.0,
                    incr=1.0,
                    unit="",
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = _main_kwargs(system, protocol)
        try:
            kernel = MultiechoKernel(
                system,
                **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        first, last = kernel.echo_times[0], kernel.echo_times[-1]
        return {
            "valid": True,
            "duration": kernel.duration,
            "info": (
                f"TA = {kernel.duration:.1f} s, "
                f"TE {first * 1e3:.2f}..{last * 1e3:.2f} ms over "
                f"{len(kernel.echo_times)} echoes"
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
        "n_echoes",
        "monopolar",
        "slice_thickness",
        "slice_order",
        "flip_angle_deg",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "partial_fourier",
        "acceleration",
        "n_acs",
        "n_averages",
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
        n_echoes=max(1, round(params.user_float(prot, 1, 4.0))),
        monopolar=bool(round(params.user_float(prot, 4, 1.0))),
        partial_fourier=params.user_float(prot, 3, 1.0),
        n_acs=params.acs_lines_from_protocol(prot, params.param_int(prot, UIParam.NY), 0),
        n_dummy=max(0, round(params.user_float(prot, 2, 16.0))),
    )


PLUGIN = GreMultiecho2D()


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
    ("--te-ms", UIParam.TE, float, "First echo time [ms], or a negative TEPreset"),
    ("--tr-ms", UIParam.TR, float, "Repetition time [ms], or a negative TRPreset"),
    ("--flip-deg", UIParam.FLIP, float, "Flip angle [deg]"),
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
    ("--echoes", UIParam.user_value(1), float, "Echoes per repetition"),
    ("--partial-fourier", UIParam.user_value(3), float, "Acquired phase-encode fraction in (0.5, 1]"),
    (
        "--bipolar",
        UIParam.user_value(4),
        lambda value: 0.0 if float(value) else 1.0,
        "Pass 1 for a bipolar train (0, the default value, keeps it monopolar)",
    ),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate a multi-echo 2D Cartesian gradient-echo .seq offline.",
            default_output="gre_multiecho_2d.seq",
        )
    )
