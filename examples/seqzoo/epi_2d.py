"""2D gradient-echo EPI, multi-slice, exported as a linked pair of sequences.

The PNS and mechanical-resonance stress case: an EPI train whose blips ride
the read ramps -- :class:`design.EpiReadout2D` -- played single-shot or in
interleaved segments. Every line carries ``REV`` for its polarity, so a
reconstruction reverses what was read backwards instead of guessing.

**The export exercises the Sequence Collection path.** The phase navigator
-- the same train, blips nulled, three lines labelled ``NAV``/``REF``, plus
one opposite-phase-encode reference shot labelled ``SET = 1`` for a
distortion correction -- is its own :class:`pulserver.pypulseq.Sequence`,
written first, carrying ``NextSequence`` in its definitions; the main
acquisition is the file it points to, written beside it. The interpreter
follows the chain and treats each file as one subsequence.
:mod:`pulserver.reczoo.epi_2d` reads the navigator groups back through
:func:`pulserver.recon._mrd.epi.partition_epi_acquisitions`.

``main`` returns the main :class:`pulserver.pypulseq.Sequence` and
``navigator`` its partner; ``PLUGIN`` writes the linked pair. Running this
module as a script writes both from the same controls.
"""

from __future__ import annotations

import sys
from pathlib import Path
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

#: Blip-nulled lines in the navigator: odd, even, odd.
_NAV_LINES = 3

#: Per-plugin ceilings on the gradient and slew limits, in mT/m and T/m/s. The
#: sequence is held below the smaller of these and what the scanner reports, so
#: lowering them here -- on the scanner console, even -- reruns the whole script
#: under gentler gradients (for PNS headroom, acoustic comfort, eddy currents)
#: without touching anything else. Defaults sit above typical hardware, so they
#: cap nothing until you lower them.
MAX_GRAD = 80.0
MAX_SLEW = 200.0


def _shared_kernel(
    system: pp.Opts,
    *,
    fov,
    n_x: int,
    n_y: int,
    slice_thickness: float,
    flip_angle_deg: float,
    te: float | None = None,
    tr: float | None = None,
    segments: int = 1,
    acceleration: int = 1,
    readout_bandwidth_hz: float = 500e3,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """The excitation and train both sequences are built from."""
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov
    excitation = design.SpatialSelectiveExcitation(
        system, flip_angle_deg, slice_thickness
    )
    epi = design.EpiReadout2D(
        system,
        excitation.rf,
        excitation.gz,
        excitation.gz_reph,
        fov=(fov_x, fov_y),
        matrix=(n_x, n_y),
        segments=segments,
        acceleration=acceleration,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        labels=("LIN",),
    )
    return SimpleNamespace(
        excitation=excitation, epi=epi, fov=(fov_x, fov_y)
    )


def _play_shot(
    seq,
    epi,
    *,
    origin_line: int | None,
    n_y: int,
    rev_label,
    extra_line_labels=(),
    invert_phase: bool = False,
    blip_nulled: bool = False,
    n_lines: int | None = None,
) -> None:
    """One excitation and its train, per the module's loop contract.

    ``origin_line`` is the absolute first line; ``None`` nulls the prewinder
    (the navigator's centre line). ``invert_phase`` negates every
    phase-encode event -- the opposite-PE reference. ``blip_nulled`` drops
    the blips, and ``n_lines`` truncates the train.
    """
    sign = -1.0 if invert_phase else 1.0
    count = epi.etl if n_lines is None else n_lines

    seq.add_block(epi.rf, epi.gz)
    if getattr(epi, "wait_te", None) is not None:
        seq.add_block(epi.wait_te)

    scale = 0.0 if origin_line is None else sign * (origin_line - n_y / 2) / (n_y / 2)
    if origin_line is not None:
        epi.shot_labels[0].value = int(origin_line)
    seq.add_block(
        epi.gx_pre,
        pp.scale_grad(epi.gy_pre, scale),
        *(epi.shot_labels if origin_line is not None else ()),
    )
    for line in range(count):
        rev_label.value = int(line % 2)
        events = [epi.gx[line], epi.adc, rev_label]
        blip = epi.gy_blips[line]
        if blip is not None and not blip_nulled:
            events.append(blip if not invert_phase else pp.scale_grad(blip, -1.0))
        events.extend(epi.line_labels[line] if origin_line is not None else ())
        events.extend(extra_line_labels)
        seq.add_block(*events)
    seq.add_block(epi.gx_spoil, pp.scale_grad(epi.gy_rew, scale))
    if getattr(epi, "wait_tr", None) is not None:
        seq.add_block(epi.wait_tr)


def navigator(
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    flip_angle_deg: float = 70.0,
    te: float | None = None,
    tr: float | None = None,
    segments: int = 1,
    acceleration: int = 1,
    readout_bandwidth_hz: float = 500e3,
    opposite_reference: bool = True,
    slice_order: str = "interleaved",
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Build the navigator sequence: blip-nulled lines, and the reference.

    Three centre lines with the blips dropped, labelled ``NAV`` and ``REF``
    with ``REV`` marking the reversed ones -- what an odd/even phase fit
    reads -- then, optionally, one full shot with every phase-encode event
    negated, labelled ``SET = 1``: the opposite-polarity acquisition a
    distortion correction pairs with the main scan.

    Parameters mirror :func:`main` where they overlap.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The navigator sequence.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
    kernel = _shared_kernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        slice_thickness=slice_thickness,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        segments=segments,
        acceleration=acceleration,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
    )
    epi = kernel.epi
    excitation = kernel.excitation
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )

    seq = pp.Sequence(system)
    rev_label = pp.make_label("REV", "SET", 0)
    nav_label = pp.make_label("NAV", "SET", 1)
    ref_label = pp.make_label("REF", "SET", 1)
    nav_clear = pp.make_label("NAV", "SET", 0)
    ref_clear = pp.make_label("REF", "SET", 0)
    set_label = pp.make_label("SET", "SET", 1)
    slc_label = pp.make_label("SLC", "SET", 0)

    # Match the main scan's slice order so the navigator's phase estimate is
    # acquired under the same slice-to-slice timing.
    for i_slice in (int(i) for i in pp.calc_traversal_order(n_slices, slice_order)):
        epi.rf.freq_offset = excitation.gz.amplitude * slice_positions[i_slice]
        epi.rf.phase_offset = -2 * np.pi * epi.rf.freq_offset * epi.rf.center
        slc_label.value = int(i_slice)

        _play_shot(
            seq,
            epi,
            origin_line=None,
            n_y=n_y,
            rev_label=rev_label,
            extra_line_labels=(nav_label, ref_label, slc_label),
            blip_nulled=True,
            n_lines=_NAV_LINES,
        )
        if opposite_reference:
            _play_shot(
                seq,
                epi,
                origin_line=0,
                n_y=n_y,
                rev_label=rev_label,
                extra_line_labels=(nav_clear, ref_clear, set_label, slc_label),
                invert_phase=True,
            )

    fov_x, fov_y = kernel.fov
    seq.set_definition(key="FOV", value=[fov_x, fov_y, slice_thickness * n_slices])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_slices])
    seq.set_definition(key="Name", value="epi_2d_navigator")
    seq.set_definition(key="EchoSpacing", value=epi.esp)
    return seq


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "epi_2d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    flip_angle_deg: float = 70.0,
    te: float | None = None,
    tr: float | None = None,
    n_repetitions: int = 1,
    segments: int = 1,
    acceleration: int = 1,
    readout_bandwidth_hz: float = 500e3,
    opposite_reference: bool = True,
    slice_order: str = "interleaved",
    n_gain_calibration_readouts: int | None = None,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Create the main 2D EPI sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the linked navigator + main pair; ``seq_filename`` names the
        navigator, which carries ``NextSequence`` pointing at the main file
        written beside it. Default is False.
    seq_filename : str, optional
        Output filename for the navigator when writing. Default is
        'epi_2d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float or tuple of float, optional
        In-plane field of view in meters, (fov_x, fov_y) if a tuple. Default
        is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters. Default is
        (0.0, 0.0, 0.0).
    n_x : int, optional
        Number of readout samples. Default is 128.
    n_y : int, optional
        Number of phase-encode lines. Default is 128.
    n_slices : int, optional
        Number of slices, interleaved within each repetition. Default is 1.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 5e-3.
    slice_gap : float, optional
        Gap between adjacent slices in meters. Default is 0.0.
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 70.0.
    te : float or None, optional
        Echo time of the first line in seconds. None is as short as
        possible. Default is None.
    tr : float or None, optional
        Repetition time in seconds over one shot of one slice. None is as
        short as possible. Default is None.
    n_repetitions : int, optional
        How many times the whole volume is acquired, each carrying its
        ``REP`` counter -- an fMRI-style time series. Default is 1.
    segments : int, optional
        Interleaved shots the train is split into. Default is 1.
    acceleration : int, optional
        Uniform phase-encode undersampling factor. Default is 1.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 500e3.
    opposite_reference : bool, optional
        Include the opposite-phase-encode reference in the navigator when
        writing the pair. Default is True.
    slice_order : str, optional
        Order the slices are acquired in, one of ``pp.calc_traversal_order``'s
        schemes. ``'interleaved'`` (the default) maximises the time between
        neighbouring slices.
    n_gain_calibration_readouts : int or None, optional
        Written as the ``NumGainCalibrationReadouts`` definition. None is
        one per slice. Default is None.
    spoiling_cycles : float, optional
        Cycles of dephasing left at the end of each shot. Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The main EPI sequence object.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
    kernel = _shared_kernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        slice_thickness=slice_thickness,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        segments=segments,
        acceleration=acceleration,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
    )
    epi = kernel.epi
    excitation = kernel.excitation
    fov_x, fov_y = kernel.fov
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )

    seq = pp.Sequence(system)
    rev_label = pp.make_label("REV", "SET", 0)
    slc_label = pp.make_label("SLC", "SET", 0)
    rep_label = pp.make_label("REP", "SET", 0)

    # Slices interleave by default so neighbours are excited as far apart in
    # time as the ordering allows; the frequency offset stays tied to the
    # physical slice, so only the acquisition order changes.
    slices = [int(i) for i in pp.calc_traversal_order(n_slices, slice_order)]
    for repetition in range(n_repetitions):
        rep_label.value = int(repetition)
        for segment in range(segments):
            for i_slice in slices:
                epi.rf.freq_offset = (
                    excitation.gz.amplitude * slice_positions[i_slice]
                )
                epi.rf.phase_offset = (
                    -2 * np.pi * epi.rf.freq_offset * epi.rf.center
                )
                slc_label.value = int(i_slice)
                _play_shot(
                    seq,
                    epi,
                    origin_line=segment,
                    n_y=n_y,
                    rev_label=rev_label,
                    extra_line_labels=(slc_label, rep_label),
                )

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
    seq.set_definition(key="Name", value="epi_2d")
    seq.set_definition(key="TE", value=epi.echo_times[len(epi.order) // 2])
    seq.set_definition(key="EchoSpacing", value=epi.esp)
    seq.set_definition(key="EPIFactor", value=epi.etl)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=n_slices if n_gain_calibration_readouts is None else n_gain_calibration_readouts,
    )

    if write_seq:
        write_pair(
            seq,
            seq_filename,
            system=system,
            fov=fov,
            n_x=n_x,
            n_y=n_y,
            n_slices=n_slices,
            slice_thickness=slice_thickness,
            slice_gap=slice_gap,
            flip_angle_deg=flip_angle_deg,
            te=te,
            tr=tr,
            segments=segments,
            acceleration=acceleration,
            readout_bandwidth_hz=readout_bandwidth_hz,
            opposite_reference=opposite_reference,
            spoiling_cycles=spoiling_cycles,
        )

    return seq


def write_pair(main_seq: pp.Sequence, seq_filename: str, **navigator_kwargs) -> tuple[str, str]:
    """Write navigator and main as a linked pair.

    The navigator goes to ``seq_filename`` carrying ``NextSequence`` with the
    main file's name; the main goes beside it as ``<stem>_main.seq``. The
    chain is what the interpreter's sequence collection follows, one file per
    subsequence.

    Parameters
    ----------
    main_seq : pulserver.pypulseq.Sequence
        The main acquisition.
    seq_filename : str
        Where the navigator is written.
    **navigator_kwargs
        Forwarded to :func:`navigator`.

    Returns
    -------
    tuple of str
        The navigator and main paths, in chain order.
    """
    path = Path(seq_filename)
    main_path = path.with_name(path.stem + "_main.seq")

    lead = navigator(**navigator_kwargs)
    lead.set_definition(key="NextSequence", value=main_path.name)
    write_sequence(lead, str(path), offline=True)
    write_sequence(main_seq, str(main_path), offline=True)
    return str(path), str(main_path)


class Epi2D(SequencePlugin):
    """The 2D EPI behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=-1.0,
                    min=-1.0,
                    max=150.0,
                    incr=0.1,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 25.0, 30.0, 40.0, 60.0],
                ),
                UIParam.TR: DropdownFloatParam(
                    value=-1.0,
                    min=-1.0,
                    max=10000.0,
                    incr=1.0,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 50.0, 100.0, 200.0, 500.0],
                ),
                UIParam.FLIP: DropdownFloatParam(
                    value=70.0,
                    min=10.0,
                    max=90.0,
                    incr=1.0,
                    unit="deg",
                    options=[45.0, 60.0, 70.0, 80.0, 90.0],
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
                    options=[2.0, 3.0, 4.0, 5.0, 8.0],
                ),
                UIParam.SLICE_SPACING: DropdownFloatParam(
                    value=5.0,
                    min=1.0,
                    max=20.0,
                    incr=0.5,
                    unit="mm",
                    options=[2.0, 3.0, 4.0, 5.0, 8.0],
                ),
                UIParam.NX: DropdownIntParam(
                    value=128, min=16, max=256, incr=1, options=[64, 96, 128, 192]
                ),
                UIParam.NY: DropdownIntParam(
                    value=128, min=16, max=256, incr=1, options=[64, 96, 128, 192]
                ),
                UIParam.NSLICES: DropdownIntParam(
                    value=1, min=1, max=128, incr=1, options=[1, 10, 20, 30, 40]
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=500e3, min=100e3, max=1000e3, incr=1000.0, unit="Hz"
                ),
                UIParam.RY: TypeinFloatParam(
                    value=1.0, min=1.0, max=8.0, incr=1.0, unit=""
                ),
                UIParam.NUM_FRAMES: DropdownIntParam(
                    value=1, min=1, max=1024, incr=1, options=[1, 10, 100, 300, 600]
                ),
                UIParam.FOV_OFFSET_X: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Y: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.FOV_OFFSET_Z: OffFloatParam(value=0.0, min=-500.0, max=500.0, unit="mm"),
                UIParam.user_name(0): Description(text="Segments"),
                UIParam.user_value(0): TypeinFloatParam(
                    value=1.0, min=1.0, max=32.0, incr=1.0, unit=""
                ),
                UIParam.user_name(1): Description(text="Opposite-PE reference"),
                UIParam.user_value(1): TypeinFloatParam(
                    value=1.0, min=0.0, max=1.0, incr=1.0, unit=""
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = _main_kwargs(system, protocol)
        try:
            kernel = _shared_kernel(
                system,
                **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        shots = kwargs.get("segments", 1) * kwargs.get("n_slices", 1)
        duration = (
            kwargs.get("n_repetitions", 1) * shots * kernel.epi.seq.duration()[0]
        )
        return {
            "valid": True,
            "duration": duration,
            "info": (
                f"TA = {duration:.1f} s, ETL = {kernel.epi.etl}, "
                f"ESP = {kernel.epi.esp * 1e3:.2f} ms"
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
        """Build both sequences and write the linked pair at ``output_path``."""
        del offline  # the chain is followed from files, so both are written
        kwargs = _main_kwargs(system, protocol)
        seq = main(**kwargs)
        write_pair(
            seq,
            output_path,
            system=system,
            **{
                name: value
                for name, value in kwargs.items()
                if name in _NAVIGATOR_ARGUMENTS
            },
        )


_KERNEL_ARGUMENTS = frozenset(
    (
        "fov",
        "n_x",
        "n_y",
        "slice_thickness",
        "flip_angle_deg",
        "te",
        "tr",
        "segments",
        "acceleration",
        "readout_bandwidth_hz",
        "spoiling_cycles",
    )
)

_NAVIGATOR_ARGUMENTS = frozenset(
    (
        "fov",
        "n_x",
        "n_y",
        "n_slices",
        "slice_thickness",
        "slice_gap",
        "flip_angle_deg",
        "te",
        "tr",
        "segments",
        "acceleration",
        "readout_bandwidth_hz",
        "opposite_reference",
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
        segments=max(1, round(params.user_float(prot, 0, 1.0))),
        opposite_reference=bool(round(params.user_float(prot, 1, 1.0))),
        n_repetitions=params.param_int(prot, UIParam.NUM_FRAMES),
    )


PLUGIN = Epi2D()


def get_default_protocol(system):
    """Bridge entry point: the plugin's default protocol."""
    return PLUGIN.get_default_protocol(system)


def validate_protocol(system, protocol):
    """Bridge entry point: protocol feasibility and scan duration."""
    return PLUGIN.validate_protocol(system, protocol)


def make_sequence(system, protocol, output_path):
    """Bridge entry point: write the linked ``.seq`` pair."""
    return PLUGIN.make_sequence(system, protocol, output_path)


_ARG_MAP = [
    ("--te-ms", UIParam.TE, float, "Echo time [ms], or a negative TEPreset"),
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
    ("--frames", UIParam.NUM_FRAMES, int, "Volumes in the time series"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    ("--offset-y-mm", UIParam.FOV_OFFSET_Y, float, "Volume offset along phase encode [mm]"),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slice [mm]"),
    ("--segments", UIParam.user_value(0), float, "Interleaved shots per plane"),
    (
        "--no-opposite-reference",
        UIParam.user_value(1),
        lambda value: 0.0 if float(value) else 1.0,
        "Pass 1 to drop the opposite-PE reference from the navigator",
    ),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate a linked 2D EPI navigator + main .seq pair offline.",
            default_output="epi_2d.seq",
        )
    )
