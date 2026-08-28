"""RF-spoiled multi-echo 3D Cartesian gradient echo, slab-selective.

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
from pulserver.design import (
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


#: Whether the plugin wave-encodes the readout. A corkscrew gradient during
#: the readout spreads every voxel along it, so the aliasing parallel imaging
#: has to separate is spread with it and a higher acceleration comes back
#: clean. ``"phase"`` drives the phase axis, ``"partition"`` the partition
#: axis, ``"both"`` the corkscrew. :mod:`pulserver.app.wave3D_recon` is what
#: reads it back -- an ordinary Cartesian reconstruction cannot.
WAVE = None

#: Periods of the corkscrew across the readout, and the peak it reaches on
#: each axis in T/m. The amplitude is a ceiling: a sinusoid slews at its
#: amplitude times its frequency, so a fast corkscrew is bounded by the
#: gradient system rather than by what is asked of it.
WAVE_CYCLES = 8
WAVE_AMPLITUDE = 8e-3


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "gre_multiecho_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    n_echoes: int = 4,
    monopolar: bool = True,
    slab_thickness: float = 128e-3,
    flip_angle_deg: float = 12.0,
    te: float | None = None,
    tr: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    partial_fourier: float = 1.0,
    partial_fourier_z: float = 1.0,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 0,
    elliptical: bool = True,
    n_acs: int = 24,
    n_acs_z: int = 16,
    n_averages: int = 1,
    n_dummy: int = 64,
    n_gain_calibration_readouts: int = 1,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
    wave: str | None = None,
    wave_cycles: int = 8,
    wave_amplitude: float = 8e-3,
) -> pp.Sequence:
    """Create an RF-spoiled multi-echo 3D Cartesian gradient-echo sequence.

    The echo-train readout on a volume: every repetition phase-encodes one
    ``(ky, kz)`` pair and reads it at each of ``n_echoes`` echo times, monopolar
    or bipolar, each acquisition carrying its echo index as ``ECO``. Everything
    else -- the slab excitation, the autocalibration rectangle leading the
    traversal, the CAIPIRINHA lattice with its selectable kz shift per ky block
    under regular undersampling, partial Fourier, spoiling -- is
    :mod:`pulserver.app.gre3D_sequence`.
    :mod:`pulserver.app.cartesian3D_recon` reconstructs one volume per echo.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'gre_multiecho_3d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float or tuple of float, optional
        In-plane field of view in meters, (fov_x, fov_y) if a tuple. Default
        is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters along the logical
        readout, phase and slab axes. Default is (0.0, 0.0, 0.0).
    n_x : int, optional
        Number of readout samples. Default is 128.
    n_y : int, optional
        Number of phase encoding steps. Default is 128.
    n_z : int, optional
        Number of partition encoding steps. Default is 64.
    n_echoes : int, optional
        Echoes per repetition. Default is 4.
    monopolar : bool, optional
        Rewind between echoes so every one is read the same way. False
        alternates the readout sign instead. Default is True.
    slab_thickness : float, optional
        Excited slab thickness in meters, also the field of view along z.
        Default is 128e-3.
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 12.0.
    te : float or None, optional
        First echo time in seconds; the rest follow at the train's echo
        spacing. None is as short as possible. Default is None.
    tr : float or None, optional
        Repetition time in seconds, one per excitation. None is as short as
        possible. Default is None.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 250e3.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired along y, in (0.5, 1].
        Default is 1.0.
    partial_fourier_z : float, optional
        Fraction of the partition-encode extent acquired along z, in (0.5, 1].
        Truncates the partitions before the centre. Default is 1.0.
    acceleration : int, optional
        Uniform phase-encode undersampling factor along y. Default is 1.
    acceleration_z : int, optional
        Uniform partition-encode undersampling factor along z. Default is 1.
    caipi_shift : int, optional
        CAIPIRINHA shift along kz per sampled-ky block,
        0 <= caipi_shift < acceleration_z. 0 is a regular lattice. Default
        is 0.
    elliptical : bool, optional
        Restrict the phase-encode support to the inscribed ky-kz ellipse,
        dropping the corners a round object never fills. Default is True.
    n_acs : int, optional
        Autocalibration extent along y, in lines. Default is 24.
    n_acs_z : int, optional
        Autocalibration extent along z, in partitions. Default is 16.
    n_averages : int, optional
        How many times the scan is acquired, written into the block table.
        Default is 1.
    n_dummy : int, optional
        Repetitions played without acquiring before the first pair. Default
        is 64.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment in degrees. Default is 117.0.
    spoiling_cycles : float, optional
        Cycles of dephasing left on the readout axis at the end of each
        repetition. Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The multi-echo GRE sequence object.

    Examples
    --------
    >>> from pulserver.app import gre_multiecho3D_sequence
    >>> seq = gre_multiecho3D_sequence(n_x=32, n_y=16, n_z=4, n_echoes=3, n_dummy=0)
    >>> seq.num_trs, seq.num_segments
    (64, 1)

    The waveform figures below are one design, prescribed to be *legible*
    rather than diagnostic: the shortest echo spacing and TR the readout
    admits, so nothing waits; a long readout and heavy spoiling, so those lobes
    are unmistakable; and three echoes over a small partition grid, so the
    figures stay readable.

    .. plot::
       :include-source:
       :nofigs:
       :context:

       from pulserver.app import gre_multiecho3D_sequence

       seq = gre_multiecho3D_sequence(
           n_x=256, n_y=16, n_z=4, n_echoes=3, te=None, tr=None, n_acs=0,
           n_acs_z=0, n_dummy=0, spoiling_cycles=6.0,
       )

    **The excitation**, and the magnetisation it leaves behind: the pulse's
    own ``B1`` envelope beside the profile it writes across the selected
    axis.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_rf(plot_now=False)

    **One repetition**, carrying the whole echo train: one slab excitation,
    then a readout per echo with a rewinder between them, so every echo is
    read the same way, then the spoiler.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot(tr="worst_case", time_disp="ms", grad_disp="mT/m", plot_now=False)

    **What the scan covers**, as a phase-encode against partition grid.
    Every coordinate is read once per echo, so the echo panel is the T2*
    weighting the sample carries.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_kspace(plane="yz", color_by="order", plot_now=False)

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
    readouts and the rewinders between them ramp as fast as they are allowed
    to. A lower ``MAX_SLEW``, a longer echo time, or a narrower readout
    bandwidth each bring the response down.

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

    kernel = Multiecho3DKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        n_echoes=n_echoes,
        monopolar=monopolar,
        slab_thickness=slab_thickness,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        partial_fourier=partial_fourier,
        partial_fourier_z=partial_fourier_z,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        caipi_shift=caipi_shift,
        elliptical=elliptical,
        n_acs=n_acs,
        n_acs_z=n_acs_z,
        n_averages=n_averages,
        n_dummy=n_dummy,
        spoiling_cycles=spoiling_cycles,
        wave=wave,
        wave_cycles=wave_cycles,
        wave_amplitude=wave_amplitude,
    )
    readout = kernel.readout
    fov_x, fov_y = kernel.fov
    pairs = kernel.pairs
    last_calibration_pair = kernel.n_calibration - 1

    # The wave-free calibration pass is repetitions of its own, and the
    # spoiling schedule has to cover every one of them.
    n_wave_calibration = kernel.n_calibration if wave is not None else 0
    rf_phases = pp.make_rf_spoiling_schedule(
        len(pairs) + n_dummy + n_wave_calibration,
        increment=np.deg2rad(rf_spoiling_increment_deg),
    )

    seq = pp.Sequence(system)
    spoiling_phase = iter(rf_phases)
    lin_label, par_label, ima_label, seg_label, eco_label = readout.adc_labels
    wait_te = getattr(readout, "wait_te", None)
    wait_tr = getattr(readout, "wait_tr", None)

    def corkscrew(scale: float) -> tuple:
        """The wave events at ``scale``, or nothing when the wave is off.

        Scaling rather than dropping them is what keeps a wave-free readout
        the same block as a wave-encoded one: the amplitude changes, the shape
        and the definition do not, so the sequence stays one repeating unit
        and its segments are what they would have been.
        """
        return tuple(
            pp.scale_grad(event, scale)
            for event in (
                getattr(readout, "gy_wave", None),
                getattr(readout, "gz_wave", None),
            )
            if event is not None
        )

    def repetition(
        ky: float, kz: float, acquire: bool, marks=(), wave_scale: float = 1.0
    ) -> None:
        """Play one TR, acquiring or not."""
        rf_phase = next(spoiling_phase)
        readout.rf.phase_offset = rf_phase
        readout.adc.phase_offset = rf_phase

        seq.add_block(readout.rf, readout.gz, *marks)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(
            readout.gx_pre,
            pp.scale_grad(readout.gy_pre, ky),
            pp.scale_grad(readout.gz_pre, kz),
        )
        for i_echo in range(n_echoes):
            if monopolar and i_echo:
                seq.add_block(readout.gx_flyback)
            lobe = readout.gx if monopolar or i_echo % 2 == 0 else readout.gx_rev
            if acquire:
                eco_label.value = i_echo
                seq.add_block(
                    lobe, readout.adc, *corkscrew(wave_scale), *readout.adc_labels
                )
            else:
                seq.add_block(lobe, *corkscrew(wave_scale))
        seq.add_block(
            readout.gx_spoil,
            pp.scale_grad(readout.gy_rew, ky),
            pp.scale_grad(readout.gz_rew, kz),
        )
        if wait_tr is not None:
            seq.add_block(wait_tr)

    for i_dummy in range(n_dummy):
        repetition(
            0.0,
            0.0,
            acquire=False,
            marks=(pp.make_label("ONCE", "SET", 1),) if i_dummy == 0 else (),
        )

    # Labels whose value has changed, waiting for the next repetition to carry
    # them. Sticky state means one carries each change however many follow.
    pending = [pp.make_label("ONCE", "SET", 0)] if n_dummy else []

    # A wave-encoded line carries no coil information a sensitivity solve can
    # use: every voxel is smeared along the readout, which is the point of it.
    # So with the wave on the calibration rectangle is acquired again ahead of
    # the scan with the corkscrew scaled away, flagged calibration-only --
    # the wave-free copy is not part of the k-space the imaging train fills.
    if wave is not None:
        pending.append(pp.make_label("REF", "SET", 1))
        ima_label.value = 0
        seg_label.value = 0
        for line, partition in pairs[: kernel.n_calibration]:
            lin_label.value = line
            par_label.value = partition
            repetition(
                (line - n_y / 2) / (n_y / 2),
                (partition - n_z / 2) / (n_z / 2),
                acquire=True,
                marks=tuple(pending),
                wave_scale=0.0,
            )
            pending = []
        pending = [pp.make_label("REF", "SET", 0)]

    for index, (line, partition) in enumerate(pairs):
        ky = (line - n_y / 2) / (n_y / 2)
        kz = (partition - n_z / 2) / (n_z / 2)
        lin_label.value = line
        par_label.value = partition
        # With the wave on the rectangle was already acquired wave-free, so
        # these lines are imaging data and nothing else.
        ima_label.value = 0 if wave is not None else int(index <= last_calibration_pair)
        seg_label.value = 1 if wave is not None else int(index > last_calibration_pair)

        repetition(ky, kz, acquire=True, marks=tuple(pending))
        pending = []

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset),
        system=system,
        compat=False,
    ).apply_to_sequence(seq, in_place=True)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_z])
    seq.set_definition(key="Name", value="gre_multiecho_3d")
    seq.set_definition(key="TE", value=kernel.echo_times)
    seq.set_definition(key="TR", value=kernel.repetition_time)
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    seq.set_definition(key="kSpaceCenterLine", value=n_y // 2)
    seq.set_definition(key="kSpaceCenterPartition", value=n_z // 2)
    seq.set_definition(key="kSpaceCenterSample", value=kernel.readout.center_sample)
    seq.set_definition(key="SliceThickness", value=kernel.excitation.slice_thickness)
    seq = pp.tile(seq, n_averages, in_place=True)

    if write_seq:
        write_sequence(seq, seq_filename, offline=True)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def Multiecho3DKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 64,
    n_echoes: int = 4,
    monopolar: bool = True,
    slab_thickness: float = 128e-3,
    flip_angle_deg: float = 12.0,
    te: float | None = None,
    tr: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    partial_fourier: float = 1.0,
    partial_fourier_z: float = 1.0,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 0,
    elliptical: bool = True,
    n_acs: int = 24,
    n_acs_z: int = 16,
    n_averages: int = 1,
    n_dummy: int = 64,
    spoiling_cycles: float = 4.0,
    wave: str | None = None,
    wave_cycles: int = 8,
    wave_amplitude: float = 8e-3,
) -> SimpleNamespace:
    """Design the repetition, and the plan that repeats it.

    :func:`pulserver.app.gre3D_sequence.GRE3DKernel` with the echo train of
    :func:`pulserver.app.gre_multiecho2D_sequence.MultiechoKernel`: the readout
    carries ``n_echoes`` and the train's polarity, and the echo times are
    read off the built blocks.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov, n_x, n_y, n_z, n_echoes, monopolar, slab_thickness, \
flip_angle_deg, te, tr, readout_bandwidth_hz, partial_fourier, \
partial_fourier_z, acceleration, acceleration_z, caipi_shift, elliptical, n_acs, n_acs_z, \
n_averages, n_dummy, spoiling_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        As :func:`pulserver.app.gre3D_sequence.GRE3DKernel` returns, with
        ``echo_times`` (one per echo) in place of ``echo_time``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = design.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        slab_thickness,
        duration_s=PULSE_DURATION,
        is_slab=True,
        time_bw_product=TIME_BW_PRODUCT,
    )

    readout = design.LineReadout3D(
        system,
        excitation.rf,
        excitation.gz,
        fov=(fov_x, fov_y, slab_thickness),
        matrix=(n_x, n_y, n_z),
        te=te,
        tr=tr,
        n_echoes=n_echoes,
        flyback=monopolar,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        labels=("LIN", "PAR", "IMA", "SEG", "ECO"),
        wave=wave,
        wave_cycles=wave_cycles,
        wave_amplitude=wave_amplitude,
    )

    echo_spacing = pp.calc_duration(readout.gx)
    if monopolar and n_echoes > 1:
        echo_spacing += pp.calc_duration(readout.gx_flyback)
    echo_times = [
        readout.echo_time + i_echo * echo_spacing for i_echo in range(n_echoes)
    ]

    pairs, n_calibration = pp.calc_sampled_pairs(
        (n_y, n_z),
        (acceleration, acceleration_z),
        (n_acs, n_acs_z),
        partial_fourier=(partial_fourier, partial_fourier_z),
        caipi_shift=caipi_shift,
        elliptical=elliptical,
        order="calibration_first",
    )

    # With the wave on the calibration rectangle is acquired again, wave-free,
    # so it is played once more than the traversal already asks for.
    n_wave_calibration = n_calibration if wave is not None else 0
    duration = (
        n_dummy + n_wave_calibration + n_averages * len(pairs)
    ) * readout.duration

    return SimpleNamespace(
        excitation=excitation,
        readout=readout,
        fov=(fov_x, fov_y),
        pairs=pairs,
        n_calibration=n_calibration,
        n_averages=n_averages,
        echo_times=echo_times,
        repetition_time=readout.duration,
        bandwidth_hz=readout.bandwidth_hz,
        duration=duration,
    )


# ======================================================================
# The scanner protocol contract
# ======================================================================


class GreMultiecho3D(SequencePlugin):
    """The multi-echo 3D gradient echo behind the scanner protocol contract."""

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
                    value=-1.0,
                    min=-1.0,
                    max=5000.0,
                    incr=0.1,
                    unit="ms",
                    options=[TRPreset.MINIMUM, 15.0, 30.0, 50.0, 100.0],
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
                    value=2.0,
                    min=0.5,
                    max=10.0,
                    incr=0.1,
                    unit="mm",
                    options=[1.0, 1.5, 2.0, 3.0, 5.0],
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
                    value=64,
                    min=8,
                    max=256,
                    incr=1,
                    options=[32, 64, 96, 128, 192],
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
                UIParam.RZ: TypeinFloatParam(
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
                UIParam.user_name(0): Description(text="ACS lines (y)"),
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
                    value=64.0,
                    min=0.0,
                    max=512.0,
                    incr=1.0,
                    unit="TR",
                ),
                UIParam.user_name(3): Description(text="Partial Fourier (y)"),
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
                UIParam.user_name(5): Description(text="Partial Fourier (z)"),
                UIParam.user_value(5): TypeinFloatParam(
                    value=1.0,
                    min=0.75,
                    max=1.0,
                    incr=0.05,
                    unit="",
                ),
                UIParam.user_name(6): Description(text="ACS partitions (z)"),
                UIParam.user_value(6): TypeinFloatParam(
                    value=16.0,
                    min=0.0,
                    max=256.0,
                    incr=1.0,
                    unit="lines",
                ),
                UIParam.user_name(7): Description(text="CAIPI shift (kz per ky)"),
                UIParam.user_value(7): TypeinFloatParam(
                    value=0.0,
                    min=0.0,
                    max=8.0,
                    incr=1.0,
                    unit="",
                ),
                UIParam.user_name(8): Description(text="Elliptical sampling"),
                UIParam.user_value(8): TypeinFloatParam(
                    value=1.0, min=0.0, max=1.0, incr=1.0, unit=""
                ),
                # The corkscrew's two controls, in the first slots this
                # sequence has left. They describe something that is only
                # there when ``WAVE`` is set, so that is when they appear.
                **(
                    {
                        UIParam.user_name(9): Description(text="Wave cycles"),
                        UIParam.user_value(9): TypeinFloatParam(
                            value=float(WAVE_CYCLES),
                            min=1.0,
                            max=64.0,
                            incr=1.0,
                            unit="",
                        ),
                        UIParam.user_name(10): Description(text="Wave amplitude"),
                        UIParam.user_value(10): TypeinFloatParam(
                            value=WAVE_AMPLITUDE * 1e3,
                            min=0.5,
                            max=40.0,
                            incr=0.5,
                            unit="mT/m",
                        ),
                    }
                    if WAVE is not None
                    else {}
                ),
            }
        )

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = protocol_kwargs(system, protocol)
        try:
            kernel = Multiecho3DKernel(
                system,
                **{
                    name: value
                    for name, value in kwargs.items()
                    if name in KERNEL_ARGUMENTS
                },
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
        seq = main(**protocol_kwargs(system, protocol))
        write_sequence(seq, output_path, offline=offline)


KERNEL_ARGUMENTS = frozenset(
    (
        "fov",
        "n_x",
        "n_y",
        "n_z",
        "n_echoes",
        "monopolar",
        "slab_thickness",
        "flip_angle_deg",
        "te",
        "tr",
        "readout_bandwidth_hz",
        "partial_fourier",
        "partial_fourier_z",
        "acceleration",
        "acceleration_z",
        "caipi_shift",
        "elliptical",
        "n_acs",
        "n_acs_z",
        "n_averages",
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
        n_z=n_z,
        slab_thickness=n_z * partition_thickness,
        n_echoes=max(1, round(params.user_float(prot, 1, 4.0))),
        monopolar=bool(round(params.user_float(prot, 4, 1.0))),
        partial_fourier=params.user_float(prot, 3, 1.0),
        partial_fourier_z=params.user_float(prot, 5, 1.0),
        n_acs=params.acs_lines_from_protocol(
            prot, params.param_int(prot, UIParam.NY), 0
        ),
        n_dummy=max(0, round(params.user_float(prot, 2, 64.0))),
        n_acs_z=max(0, round(params.user_float(prot, 6, 16.0))),
        caipi_shift=max(0, round(params.user_float(prot, 7, 0.0))),
        elliptical=bool(round(params.user_float(prot, 8, 1.0))),
        wave=WAVE,
        wave_cycles=max(1, round(params.user_float(prot, 9, float(WAVE_CYCLES)))),
        wave_amplitude=params.user_float(prot, 10, WAVE_AMPLITUDE * 1e3) * 1e-3,
    )


PLUGIN = GreMultiecho3D()


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
    ("--te-ms", UIParam.TE, float, "First echo time [ms], or a negative TEPreset"),
    ("--tr-ms", UIParam.TR, float, "Repetition time [ms], or a negative TRPreset"),
    ("--flip-deg", UIParam.FLIP, float, "Flip angle [deg]"),
    ("--fov-mm", UIParam.FOV, float, "Readout FOV [mm]"),
    ("--phase-fov-mm", UIParam.PHASE_FOV, float, "Phase-encode FOV [mm]"),
    (
        "--partition-thickness-mm",
        UIParam.SLICE_THICKNESS,
        float,
        "Partition thickness [mm]; the slab is this times the partition count",
    ),
    ("--nx", UIParam.NX, int, "Readout matrix size"),
    ("--ny", UIParam.NY, int, "Phase-encode matrix size"),
    ("--nz", UIParam.NSLICES, int, "Partition count"),
    ("--bandwidth-hz", UIParam.BANDWIDTH, float, "Requested receiver bandwidth [Hz]"),
    ("--ry", UIParam.RY, float, "Phase-encode undersampling factor along y"),
    ("--rz", UIParam.RZ, float, "Partition-encode undersampling factor along z"),
    ("--nex", UIParam.NEX, float, "Number of signal averages"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    (
        "--offset-y-mm",
        UIParam.FOV_OFFSET_Y,
        float,
        "Volume offset along phase encode [mm]",
    ),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slab [mm]"),
    ("--acs-lines", UIParam.user_value(0), float, "Number of ACS lines along y"),
    ("--echoes", UIParam.user_value(1), float, "Echoes per repetition"),
    (
        "--partial-fourier",
        UIParam.user_value(3),
        float,
        "Acquired phase-encode fraction along y in (0.5, 1]",
    ),
    (
        "--bipolar",
        UIParam.user_value(4),
        lambda value: 0.0 if float(value) else 1.0,
        "Pass 1 for a bipolar train (0, the default value, keeps it monopolar)",
    ),
    (
        "--partial-fourier-z",
        UIParam.user_value(5),
        float,
        "Acquired partition-encode fraction along z in (0.5, 1]",
    ),
    (
        "--acs-partitions",
        UIParam.user_value(6),
        float,
        "Number of ACS partitions along z",
    ),
    ("--caipi-shift", UIParam.user_value(7), float, "CAIPIRINHA kz shift per ky block"),
    (
        "--no-elliptical",
        UIParam.user_value(8),
        lambda value: 0.0 if float(value) else 1.0,
        "Pass 1 to sample the full ky-kz rectangle instead of the ellipse",
    ),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=ARG_MAP,
            description="Generate a multi-echo 3D Cartesian gradient-echo .seq offline.",
            default_output="gre_multiecho_3d.seq",
        )
    )
