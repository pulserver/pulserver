"""RF-spoiled 2D Cartesian gradient echo, multi-slice.

``main`` returns the :class:`pulserver.pypulseq.Sequence`; ``PLUGIN`` is the
same sequence behind the scanner protocol contract, and running this module as
a script writes a ``.seq`` from the same controls.
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


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "gre_2d.seq",
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
    flip_angle_deg: float = 12.0,
    te: float | None = 8e-3,
    tr: float | None = 250e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_echo: float = 1.0,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    n_averages: int = 1,
    n_dummy: int = 16,
    n_gain_calibration_readouts: int | None = None,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Create an RF-spoiled 2D Cartesian gradient-echo sequence.

    One frequency-encoded line per repetition, from a slice-selective SLR
    excitation. Phase encoding may be undersampled with a fully sampled
    autocalibration block and truncated by partial Fourier; the readout may be a
    partial echo. :mod:`pulserver.app.cartesian2D_recon` reads all three back.

    The autocalibration block leads the traversal and closes a segment of its own,
    so the reconstruction can calibrate while the rest of the scan is still
    arriving -- which puts the centre of k-space in the transient, hence
    ``n_dummy``. More slices than one TR can hold are split into passes.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'gre_2d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float or tuple of float, optional
        In-plane field of view in meters. If a single value, it is used for
        both x and y. If a tuple, it is (fov_x, fov_y). Default is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters along the logical readout,
        phase and slice axes. Which way those axes point in the magnet is the
        interpreter's business, so only the offset is applied here. Default is
        (0.0, 0.0, 0.0).
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
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 12.0.
    te : float or None, optional
        Echo time in seconds. None is as short as possible. Default is 8e-3.
    tr : float or None, optional
        Repetition time in seconds, between successive excitations of the same
        slice. None is as short as possible, and puts every slice in one pass.
        Default is 250e-3.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. The achieved value is reported by
        the readout module and is generally lower. Default is 250e3.
    partial_echo : float, optional
        Fraction of the full echo acquired, in (0.5, 1]. Truncates the samples
        before the echo, which shortens the minimum TE. Default is 1.0.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired, in (0.5, 1]. Truncates
        the lines before the centre, which shortens the scan. Default is 1.0.
    acceleration : int, optional
        Uniform phase-encode undersampling factor. Default is 1.
    n_acs : int, optional
        Number of fully sampled autocalibration lines at the center of
        k-space, acquired ahead of the rest of the scan. Default is 24.
    n_averages : int, optional
        How many times the scan is acquired, written into the block table
        rather than left to the interpreter's repeat count. The dummies play
        on the first average only; every average carries its index as ``AVG``.
        Default is 1.
    n_dummy : int, optional
        Repetitions played without acquiring, before the first line of each
        pass, to bring the magnetisation to steady state. The autocalibration
        block leads the traversal, so these are what keeps the centre of
        k-space -- which sets the image contrast -- out of the transient.
        Default is 16.
    n_gain_calibration_readouts : int or None, optional
        How many readouts the scanner's automatic prescan may use to set the
        receive gain, written as the ``NumGainCalibrationReadouts``
        definition. None is one per slice. Default is None.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment in degrees. Default is 117.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left on the readout axis at the end of each
        repetition, counted across one voxel. Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The GRE sequence object.

    Examples
    --------
    >>> from pulserver.app import gre2D_sequence
    >>> seq = gre2D_sequence(n_x=32, n_y=16, n_slices=1, tr=12e-3, n_dummy=0)
    >>> seq.num_trs, seq.num_segments
    (16, 2)

    The waveform figures below are one design, prescribed to be *legible*
    rather than diagnostic: the shortest TE and TR the readout admits, so
    nothing waits; a long readout and heavy spoiling, so those lobes are
    unmistakable against the encodes; and few phase encodes, so one
    repetition fits on a page.

    .. plot::
       :include-source:
       :nofigs:
       :context:

       from pulserver.app import gre2D_sequence

       seq = gre2D_sequence(
           n_x=256, n_y=16, n_slices=1, te=None, tr=None,
           n_acs=0, n_dummy=0, spoiling_cycles=6.0,
       )

    **The excitation.** An SLR pulse under a selection gradient, and the
    slice it tips: a 12 degree flip leaves ``|Mxy| = sin 12`` inside the slab
    and nothing outside it.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_rf(title="slice-selective SLR excitation, 5 mm", plot_now=False)

    **One repetition**, which is the whole sequence: excitation and rephaser,
    the prewinders, the readout, then the spoiler and the phase-encode
    rewinder that leave the magnetisation ready for the next line.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot(tr="worst_case", time_disp="ms", grad_disp="mT/m", plot_now=False)

    **What the scan covers**, on a second design that has both axes worth
    drawing. A 2D scan encodes its third axis in the frequency each slice is
    excited at rather than in a gradient, so that is what stands in for
    ``kz``: five slices as five rows, interleaved so no two neighbours are
    excited back to back. Along ``ky`` the contiguous band at the centre is
    the autocalibration block, acquired first so the reconstruction can
    calibrate while the rest arrives, and everything outside it is the
    accelerated traversal that follows.

    .. plot::
       :include-source:
       :context: close-figs

       sampled = gre2D_sequence(
           n_x=64, n_y=48, n_slices=5, slice_gap=1e-3, tr=60e-3,
           acceleration=2, n_acs=12, n_dummy=0,
       )
       sampled.plot_kspace(plane="yf", color_by="order", plot_now=False)

    **Mechanical resonance.** The repetition is periodic, so its gradients
    have energy only at multiples of ``1 / T_TR``. Those lines are what a
    forbidden band is judged against, and the verdict panel is the whole of
    the acoustic check: every line inside a band, against the amplitude that
    band allows.

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
    prewinders and spoiler ramp as fast as they are allowed to and the
    response goes over the threshold -- which is the check earning its place
    rather than a property of the sequence. ``MAX_SLEW``, a longer TE, or a
    lower readout bandwidth each bring it down.

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

    # Designing the repetitions is also what validates TE, TR and the rest.
    kernel = GREKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        slice_order=slice_order,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        partial_echo=partial_echo,
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

    # The rest of the encoding plan: where each slice sits, and the RF-spoiling
    # phase every repetition is played with. The dummies share the schedule, so
    # the spoiling phase the first acquired line sees is the one it would have
    # seen mid-scan.
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
        """Play one TR of every slice of a pass, acquiring or not.

        ``mark`` rides the first excitation, for a label whose value has just
        changed. Label state is sticky, so it stays set until it is set again.
        """
        # Present only when a TE or TR longer than the minimum was asked for.
        wait_te = getattr(readout, "wait_te", None)
        wait_tr = getattr(readout, "wait_tr", None)
        for i_slice in slices:
            rf_phase = next(spoiling_phase)
            readout.rf.freq_offset = excitation.gz.amplitude * slice_positions[i_slice]
            readout.rf.phase_offset = (
                rf_phase - 2 * np.pi * readout.rf.freq_offset * readout.rf.center
            )
            readout.adc.phase_offset = rf_phase

            # Which slice this excitation is, by position: `slice_positions`
            # is in ascending order, so the loop index is the geometric index
            # a reconstruction stacks by, whatever order the passes visit.
            slc_label.value = i_slice
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
            if acquire:
                seq.add_block(readout.gx, readout.adc, *readout.adc_labels)
            else:
                seq.add_block(readout.gx)
            seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, ky))
            if wait_tr is not None:
                seq.add_block(wait_tr)

    for slices in kernel.passes:
        readout = kernel.readouts[len(slices)]
        lin_label, slc_label, ima_label, seg_label = readout.adc_labels

        # Steady state first: the same repetition without its ADC, so the
        # magnetisation the first acquired line sees is the one every later
        # line sees. The centre of k-space is acquired first and sets the
        # contrast, so this is what the ordering costs.
        #
        # `ONCE` is what keeps them out of the averages: 1 plays on the first
        # pass only, and the first acquired line clears it back to 0 so the
        # body repeats.
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
            # Every calibration line is imaging data too: the block is a
            # fully sampled centre of the same k-space rather than a separate
            # acquisition. Label state persists, so the flag is written every
            # repetition.
            lin_label.value = line
            ima_label.value = int(acs_start <= line < acs_stop)
            # The calibration block is segment zero and the rest segment one,
            # which is what lets a reconstruction calibrate the moment the
            # block is complete rather than at the end of the scan.
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
    seq.set_definition(key="Name", value="gre_2d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=n_slices
        if n_gain_calibration_readouts is None
        else n_gain_calibration_readouts,
    )

    # Where the centre of k-space is, and the slice geometry the excitation
    # actually produces: the gap is the prescribed spacing less one measured
    # thickness.
    seq.set_definition(key="kSpaceCenterLine", value=n_y // 2)
    seq.set_definition(
        key="kSpaceCenterSample",
        value=kernel.readouts[len(kernel.passes[0])].center_sample,
    )
    seq.set_definition(key="SlicePositions", value=slice_positions.tolist())
    seq.set_definition(key="SliceThickness", value=kernel.excitation.slice_thickness)
    seq.set_definition(
        key="SliceGap",
        value=slice_thickness + slice_gap - kernel.excitation.slice_thickness,
    )

    # Last, because it multiplies the block table: the averages are written
    # out rather than left to the interpreter's repeat count, so every
    # acquisition in the file carries the `AVG` it belongs to and the dummies
    # -- marked `ONCE` -- appear in the first average only.
    seq = pp.tile(seq, n_averages, in_place=True)

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def GREKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_order: str = "interleaved",
    flip_angle_deg: float = 12.0,
    te: float | None = 8e-3,
    tr: float | None = 250e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_echo: float = 1.0,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    n_averages: int = 1,
    n_dummy: int = 16,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """Design the repetitions, and the plan that repeats them.

    Whatever both the scan loop and the feasibility check need: the excitation,
    one readout per distinct pass size, which slices each pass holds, the
    phase-encode lines the sampling asks for, and the total scan time they add
    up to.

    Building the modules *is* the feasibility check. A TE shorter than one
    repetition can achieve makes :class:`design.LineReadout2D` raise
    ``ValueError``, as does any out-of-range matrix, fov or fraction, so the
    same call the scan loop is built from tells
    :meth:`Gre2D.validate_protocol` whether the protocol is feasible and how
    long it takes -- there is no second timing path to drift out of step with
    the sequence.

    A TR too short for every slice does *not* raise. The slices are dealt into
    as many passes as it takes, spread across the slab so the slices of one
    pass are not neighbours, and each pass gets a repetition of its own length
    ``tr / (slices in the pass)``. A pass therefore lasts exactly one TR
    whatever its size, which is what makes the requested TR exact for every
    slice rather than exact for most of them.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_slices, slice_thickness, slice_order, flip_angle_deg, te, \
tr, readout_bandwidth_hz, partial_echo, partial_fourier, acceleration, n_acs, \
n_averages, n_dummy, spoiling_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``readouts`` (keyed by pass size), ``passes`` (slice
        indices per pass, in excitation order), ``fov``, ``sampled_lines``,
        ``n_calibration`` (how many of them lead the traversal),
        ``n_averages``, ``echo_time``, ``repetition_time``, ``bandwidth_hz``
        and ``duration``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = design.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
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
            partial_echo=partial_echo,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            labels=("LIN", "SLC", "IMA", "SEG"),
        )

    # The shortest repetition the prescription admits says how many slices one
    # TR can hold, and so how many passes the slices have to be dealt into.
    shortest = readout(None)
    per_pass = n_slices if tr is None else max(1, int(tr / shortest.duration))
    n_passes = -(-n_slices // per_pass)

    # Dealt round-robin: the sizes come out differing by at most one, and the
    # slices of a pass come out n_passes apart, which is the same thing that
    # keeps a multi-slice acquisition from exciting neighbours back to back.
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

    # The autocalibration block is acquired first, so the reconstruction can
    # estimate coil sensitivities from it while the rest of the scan is still
    # running. It leads the traversal, so its length is where it ends.
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

    # One repetition per acquired line per slice, plus the dummies that bring
    # each pass to steady state; the readout has already padded itself to the
    # per-slice TR, so a pass is simply their sum. The averages repeat the
    # body and not the dummies, which is what `ONCE` says on them.
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
        echo_time=shortest.echo_time,
        repetition_time=max(
            len(group) * readouts[len(group)].duration for group in passes
        ),
        bandwidth_hz=shortest.bandwidth_hz,
        duration=duration,
    )


# ======================================================================
# The scanner protocol contract
# ======================================================================


class Gre2D(SequencePlugin):
    """The 2D gradient echo behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(
                    value=8.0,
                    min=1.0,
                    max=80.0,
                    incr=0.1,
                    unit="ms",
                    options=[TEPreset.MINIMUM, 5.0, 8.0, 15.0, 30.0],
                ),
                UIParam.TR: DropdownFloatParam(
                    value=250.0,
                    min=5.0,
                    max=5000.0,
                    incr=0.1,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 250.0, 500.0, 1000.0, 2000.0],
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
                # Where the operator put the slab. Not a widget: the scanner
                # fills these in from the prescription.
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
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take.

        Designing the repetitions through :func:`GREKernel` is the whole
        check: the same construction the sequence is built from raises
        ``ValueError`` on an infeasible TE, and otherwise reports the scan
        duration directly.
        """
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = protocol_kwargs(system, protocol)
        try:
            kernel = GREKernel(
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


#: What :func:`GREKernel` takes of what :func:`main` takes, so one reading of
#: the protocol serves both.
KERNEL_ARGUMENTS = frozenset(
    (
        "fov",
        "n_x",
        "n_y",
        "n_slices",
        "slice_thickness",
        "slice_order",
        "flip_angle_deg",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "partial_echo",
        "partial_fourier",
        "acceleration",
        "n_acs",
        "n_averages",
        "n_dummy",
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
        n_dummy=max(0, round(params.user_float(prot, 2, 16.0))),
    )


PLUGIN = Gre2D()


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
            description="Generate a 2D Cartesian gradient-echo .seq offline.",
            default_output="gre_2d.seq",
        )
    )
