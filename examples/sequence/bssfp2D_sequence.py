"""Balanced SSFP 2D Cartesian, multi-slice.

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

#: SLR design of the selective pulses, held here rather than left at the design
#: module's default so a script can retune the excitation without touching the
#: loop. The selection amplitude follows as
#: ``time_bw_product / (duration * thickness)``, which is also what a slice
#: offset is converted against.
PULSE_DURATION = 0.6e-3
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
    seq_filename: str = "bssfp_2d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    flip_angle_deg: float = 45.0,
    tr: float | None = None,
    readout_bandwidth_hz: float = 125e3,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    n_dummy: int = 10,
    n_gain_calibration_readouts: int | None = None,
) -> pp.Sequence:
    """Create a balanced SSFP 2D Cartesian sequence.

    The steady-state pair of constraints, both solved rather than padded: every
    axis returns to k = 0 between one pulse centre and the next, and TE sits at
    exactly TR/2 -- :class:`design.BssfpReadout2D`, whose repetition is rewind,
    excite, read. The train is entered through the half-flip catalyst pulse, half
    a TR ahead of the first excitation and opposite in phase, and every
    subsequent excitation alternates phase; ``ONCE`` marks the catalyst (1), the
    steady state (0) and the closing rewind (2). Slices are played as complete
    trains one after another, because a steady state does not survive
    interleaving. :mod:`pulserver.app.cartesian2D_recon` reads the result back.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'bssfp_2d.seq'.
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
        Number of slices, each acquired as its own complete train. Default
        is 1.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 5e-3.
    slice_gap : float, optional
        Gap between adjacent slices in meters. Default is 0.0.
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 45.0.
    tr : float or None, optional
        Repetition time in seconds. The echo time is always TR/2. None is as
        short as possible. Default is None.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. A balanced train wants a
        window long enough beside the pulse for TE to sit at TR/2, hence the
        lower default. Default is 125e3.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired, in (0.5, 1]. Default
        is 1.0.
    acceleration : int, optional
        Uniform phase-encode undersampling factor. Default is 1.
    n_acs : int, optional
        Number of fully sampled autocalibration lines, acquired ahead of the
        rest. Default is 24.
    n_dummy : int, optional
        Repetitions played without acquiring after the catalyst pulse, while
        the oscillating transient settles. Default is 10.
    n_gain_calibration_readouts : int or None, optional
        Written as the ``NumGainCalibrationReadouts`` definition. None is
        one per slice. Default is None.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The bSSFP sequence object.

    Examples
    --------
    >>> from pulserver.app import bssfp2D_sequence
    >>> seq = bssfp2D_sequence(n_x=64, n_y=16, n_slices=1, readout_bandwidth_hz=50e3, n_dummy=0)
    >>> seq.num_trs, seq.num_segments
    (1, 2)

    The waveform figures below are one design, prescribed to be *legible*
    rather than diagnostic: the shortest TR the readout admits, so nothing
    waits; a long readout, so its flat top dominates the repetition; and three
    slices of eight lines each, so the whole traversal fits on a page.

    .. plot::
       :include-source:
       :nofigs:
       :context:

       from pulserver.app import bssfp2D_sequence

       seq = bssfp2D_sequence(
           n_x=256, n_y=8, n_slices=3, slice_gap=1e-3, tr=None, n_acs=0,
           n_dummy=0,
       )

    **The excitation**, and the magnetisation it leaves behind: the pulse's
    own ``B1`` envelope beside the profile it writes across the selected
    axis.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_rf(plot_now=False)

    **One canonical TR.** One slice's whole traversal comes out as a single
    canonical TR here, so the figure is zoomed to its first 15 ms -- three
    repetitions, each returning every gradient axis to zero before the next
    excitation, which is what balanced means.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot(
           tr="worst_case",
           time_range=(0.0, 0.015),
           time_disp="ms",
           grad_disp="mT/m",
           plot_now=False,
       )

    **The segments**, which are the interpreter's units of playout. Each is
    drawn as the instance carrying the most gradient energy -- the one the
    safety checks were run against -- over the span of the scan where it
    plays.

    .. plot::
       :include-source:
       :context: close-figs

       for index in range(seq.num_segments):
           seq.plot(
               segment_idx=index, time_disp="ms", grad_disp="mT/m", plot_now=False
           )

    **What the scan covers.** A 2D scan encodes its third axis in the
    frequency each slice is excited at rather than in a gradient, so that
    is what stands in for ``kz``: three slices as three rows, eight phase
    encodes across each.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_kspace(plane="yf", color_by="order", plot_now=False)

    **Mechanical resonance.** The repetition is periodic, so its gradients
    have energy only at multiples of ``1 / T_TR``. Those lines are what a
    forbidden band is judged against, and the verdict panel is the whole of
    the acoustic check: every line that falls inside a band, against the
    amplitude that band allows.

    .. plot::
       :include-source:
       :context: close-figs

       gamma = 42.576e3  # Hz/m per mT/m
       seq.calculate_gradient_spectrum(
           tr="worst_case",
           resonance_lines=True,
           bands=[(550.0, 700.0, 3.0 * gamma), (1150.0, 1300.0, 3.0 * gamma)],
       )

    **Peripheral nerve stimulation**, under the rheobase/chronaxie model the
    scanner's own gate applies, over the same repetition played back to back.
    This design asks for the shortest timing the hardware admits, so its
    prewinders and rewinders ramp as fast as they are allowed to. A lower
    ``MAX_SLEW``, a longer echo time, or a narrower readout bandwidth each
    bring the response down.

    .. plot::
       :include-source:
       :context: close-figs

       seq.calculate_pns(
           {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333},
           tr="worst_case",
       )
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    kernel = Bssfp2DKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        flip_angle_deg=flip_angle_deg,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        partial_fourier=partial_fourier,
        acceleration=acceleration,
        n_acs=n_acs,
        n_dummy=n_dummy,
    )
    excitation = kernel.excitation
    bssfp = kernel.readout
    fov_x, fov_y = kernel.fov
    sampled_lines = kernel.sampled_lines

    acs_start = max(0, n_y // 2 - n_acs // 2)
    acs_stop = min(n_y, acs_start + n_acs)
    last_calibration_line = kernel.n_calibration - 1
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )

    seq = pp.Sequence(system)
    lin_label, slc_label, ima_label, seg_label = bssfp.adc_labels
    nominal_amplitude = bssfp.rf.amplitude
    rf_center = float(bssfp.rf.center)

    # One complete train per slice: catalyst, settle, acquire, close. The ky
    # of every shot is remembered because the next repetition's rewind undoes
    # it.
    for i_slice in range(n_slices):
        # `slice_positions` is in ascending order, so the loop index is the
        # geometric index a reconstruction stacks by.
        slc_label.value = i_slice
        freq_hz = excitation.gz.amplitude * slice_positions[i_slice]
        bssfp.rf.freq_offset = freq_hz

        # The catalyst: half the flip, half a TR early, and *opposite* in
        # phase to the first excitation -- which alternation makes phase pi --
        # so the magnetisation lands on the bisector the steady state
        # oscillates about.
        bssfp.rf.amplitude = 0.5 * nominal_amplitude
        bssfp.rf.phase_offset = -2 * np.pi * freq_hz * rf_center
        seq.add_block(bssfp.rf, bssfp.gz, *bssfp.prep_labels)
        seq.add_block(bssfp.wait_prep, *bssfp.train_labels)
        bssfp.rf.amplitude = nominal_amplitude

        previous_ky = None
        shots = [None] * n_dummy + list(sampled_lines)
        for shot, line in enumerate(shots):
            alternation = np.pi * ((shot + 1) % 2)
            bssfp.rf.phase_offset = alternation - 2 * np.pi * freq_hz * rf_center
            bssfp.adc.phase_offset = alternation

            if shot:
                seq.add_block(
                    bssfp.gx_rew,
                    pp.scale_grad(bssfp.gy_rew, previous_ky),
                    bssfp.gz_rew,
                )
            else:
                seq.add_block(bssfp.wait_rewind, bssfp.gz_rew)
            seq.add_block(bssfp.rf, bssfp.gz)

            ky = 0.0 if line is None else (line - n_y / 2) / (n_y / 2)
            if line is None:
                seq.add_block(
                    bssfp.gx,
                    pp.scale_grad(bssfp.gy_pre, ky),
                    bssfp.gz_pre,
                )
            else:
                lin_label.value = line
                ima_label.value = int(acs_start <= line < acs_stop)
                seg_label.value = int(
                    shots.index(line) - n_dummy > last_calibration_line
                )
                seq.add_block(
                    bssfp.gx,
                    bssfp.adc,
                    pp.scale_grad(bssfp.gy_pre, ky),
                    bssfp.gz_pre,
                    *bssfp.adc_labels,
                )
            previous_ky = ky

        seq.add_block(
            bssfp.gx_rew,
            pp.scale_grad(bssfp.gy_rew, previous_ky),
            bssfp.gz_rew,
            *bssfp.end_labels,
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
    seq.set_definition(key="Name", value="bssfp_2d")
    seq.set_definition(key="TE", value=bssfp.te)
    seq.set_definition(key="TR", value=bssfp.tr)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=n_slices
        if n_gain_calibration_readouts is None
        else n_gain_calibration_readouts,
    )

    seq.set_definition(key="kSpaceCenterLine", value=n_y // 2)
    seq.set_definition(key="kSpaceCenterSample", value=bssfp.center_sample)
    seq.set_definition(key="SlicePositions", value=slice_positions.tolist())
    seq.set_definition(key="SliceThickness", value=kernel.excitation.slice_thickness)
    seq.set_definition(
        key="SliceGap",
        value=slice_thickness + slice_gap - kernel.excitation.slice_thickness,
    )

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def Bssfp2DKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    flip_angle_deg: float = 45.0,
    tr: float | None = None,
    readout_bandwidth_hz: float = 125e3,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    n_dummy: int = 10,
) -> SimpleNamespace:
    """Design the repetition, and the plan that repeats it.

    Building :class:`design.BssfpReadout2D` *is* the feasibility check: it
    solves the balance and the TE = TR/2 constraint together, and raises
    ``ValueError`` when the requested TR cannot hold them.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_slices, slice_thickness, flip_angle_deg, tr, \
readout_bandwidth_hz, partial_fourier, acceleration, n_acs, n_dummy
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``readout``, ``fov``, ``sampled_lines``,
        ``n_calibration``, ``echo_time``, ``repetition_time``,
        ``bandwidth_hz`` and ``duration``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = design.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        slice_thickness,
        duration_s=PULSE_DURATION,
        rephase=False,
        time_bw_product=TIME_BW_PRODUCT,
    )
    readout = design.BssfpReadout2D(
        system,
        excitation.rf,
        excitation.gz,
        fov=(fov_x, fov_y),
        matrix=(n_x, n_y),
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        labels=("LIN", "SLC", "IMA", "SEG"),
    )

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

    # Catalyst, settle, one repetition per acquired line, closing rewind --
    # per slice, since every slice is its own train.
    train = (n_dummy + len(sampled_lines) + 1.5) * readout.tr
    duration = n_slices * train

    return SimpleNamespace(
        excitation=excitation,
        readout=readout,
        fov=(fov_x, fov_y),
        sampled_lines=sampled_lines,
        n_calibration=n_calibration,
        echo_time=readout.te,
        repetition_time=readout.tr,
        bandwidth_hz=readout.bandwidth_hz,
        duration=duration,
    )


# ======================================================================
# The scanner protocol contract
# ======================================================================


class Bssfp2D(SequencePlugin):
    """The 2D balanced SSFP behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TR: DropdownFloatParam(
                    value=-1.0,
                    min=-1.0,
                    max=20.0,
                    incr=0.05,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 3.0, 4.0, 5.0, 8.0],
                ),
                UIParam.FLIP: DropdownFloatParam(
                    value=45.0,
                    min=10.0,
                    max=90.0,
                    incr=1.0,
                    unit="deg",
                    options=[30.0, 45.0, 60.0, 70.0, 90.0],
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
                    max=64,
                    incr=1,
                    options=[1, 3, 5, 10, 20],
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=125e3,
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
                UIParam.user_name(2): Description(text="Dummy repetitions"),
                UIParam.user_value(2): TypeinFloatParam(
                    value=10.0,
                    min=0.0,
                    max=200.0,
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
            kernel = Bssfp2DKernel(
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
                f"TA = {kernel.duration:.1f} s, "
                f"TR = {kernel.repetition_time * 1e3:.2f} ms, "
                f"TE = TR/2 = {kernel.echo_time * 1e3:.2f} ms"
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
        "flip_angle_deg",
        "tr",
        "readout_bandwidth_hz",
        "partial_fourier",
        "acceleration",
        "n_acs",
        "n_dummy",
    )
)


def protocol_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
    """The prescribed quantities, plus this sequence's own user slots."""
    prot = dict_to_protocol(protocol)
    return main_kwargs(
        main,
        system,
        protocol,
        partial_fourier=params.user_float(prot, 3, 1.0),
        n_acs=params.acs_lines_from_protocol(
            prot, params.param_int(prot, UIParam.NY), 0
        ),
        n_dummy=max(0, round(params.user_float(prot, 2, 10.0))),
    )


PLUGIN = Bssfp2D()


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
        "--dummies",
        UIParam.user_value(2),
        float,
        "Unacquired repetitions after the catalyst",
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
            description="Generate a 2D balanced SSFP .seq offline.",
            default_output="bssfp_2d.seq",
        )
    )
