"""3D gradient-echo EPI, slab-selective, exported as a linked pair.

``main`` returns the main :class:`pulserver.pypulseq.Sequence`; ``PLUGIN``
writes the linked pair.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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

NAVIGATOR_LINES = 3

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

#: Swap the slab excitation for a spectral-spatial, slab- *and* water-selective
#: pulse. An EPI train reads for a long time, so fat -- a few ppm off water --
#: shifts far along the phase-encode axis and ghosts the image; exciting water
#: only removes the source rather than correcting for it. A script-level toggle
#: rather than a UI control: it reshapes the excitation the whole sequence is
#: timed around, so it belongs to whoever runs the script.
SPSP_EXCITATION = False

#: Chemical shift of the fat methylene resonance from water, in ppm. Held in
#: ppm rather than hertz so it is field-strength independent: the water-only
#: excitation converts it against ``system.B0`` when the pulse is built, so the
#: same script targets fat at 1.5 T and 3 T alike.
FAT_SHIFT_PPM = -3.4

#: Offer the fMRI multiphase mode: a time series of ``UIParam.NUM_FRAMES``
#: frames, each carrying its ``REP`` counter. A script-level toggle because a
#: multiphase scan is a different study than a single volume. When ``False`` the
#: frame count is forced to one and the multiphase control is dropped from the
#: protocol, so the console never shows it.
ENABLE_MULTIPHASE = False


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "epi_3d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 32,
    slab_thickness: float = 96e-3,
    flip_angle_deg: float = 20.0,
    te: float | None = None,
    tr: float | None = None,
    n_repetitions: int = 1,
    segments: int = 1,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 1,
    partial_fourier: float = 1.0,
    partial_fourier_z: float = 1.0,
    n_acs: int = 24,
    n_acs_z: int = 16,
    readout_bandwidth_hz: float = 500e3,
    opposite_reference: bool = True,
    partition_order: str = "center_out",
    n_dummy: int = 2,
    n_gain_calibration_readouts: int = 1,
    spoiling_cycles: float = 4.0,
    spsp: bool = False,
) -> pp.Sequence:
    """Create the main 3D EPI sequence.

    The volume counterpart of :mod:`pulserver.app.epi2D_sequence`: one
    :class:`design.EpiReadout3D` train per ``(segment, shell)``, blips on the
    read ramps, ``REV`` on every line, the partition carried as ``PAR``. A shell
    is one partition for a plain stack of trains, or a band of ``Rz`` partitions
    the CAIPI sawtooth walks for segmented blipped-CAIPI. The
    navigator -- blip-nulled ``NAV``/``REF`` lines at the centre partition plus
    the opposite-phase-encode ``SET = 1`` reference -- is its own sequence,
    linked ahead of the main file through ``NextSequence``, exercising the
    Sequence Collection path. :mod:`pulserver.app.epi3D_recon` reads both back.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the linked navigator + main pair; ``seq_filename`` names the
        navigator. Default is False.
    seq_filename : str, optional
        Output filename for the navigator when writing. Default is
        'epi_3d.seq'.
    system : pypulseq.Opts, optional
        System limits. Default is `pp.Opts()`.
    fov : float or tuple of float, optional
        In-plane field of view in meters. Default is 220e-3.
    fov_offset : tuple of float, optional
        Where the prescribed volume sits, in meters. Default is
        (0.0, 0.0, 0.0).
    n_x : int, optional
        Number of readout samples. Default is 128.
    n_y : int, optional
        Number of phase-encode lines. Default is 128.
    n_z : int, optional
        Number of partitions, one shot each per segment. Default is 32.
    slab_thickness : float, optional
        Excited slab thickness in meters, also the field of view along z.
        Default is 96e-3.
    flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 20.0.
    te : float or None, optional
        Echo time of the first line in seconds. None is as short as
        possible. Default is None.
    tr : float or None, optional
        Repetition time in seconds over one shot. None is as short as
        possible. Default is None.
    n_repetitions : int, optional
        Volumes in the time series, each carrying ``REP``. Default is 1.
    segments : int, optional
        Interleaved shots per partition. Default is 1.
    acceleration : int, optional
        Uniform phase-encode undersampling factor along y. Default is 1.
    acceleration_z : int, optional
        Partition undersampling factor along z, ``Rz``. Above 1 the train
        becomes segmented blipped-CAIPI: each shot walks a shell of ``Rz``
        partitions with the CAIPI sawtooth, and ``n_z // Rz`` shells tile the
        :func:`pp.make_caipirinha_mask` lattice. Default is 1.
    caipi_shift : int, optional
        Partitions the CAIPI sawtooth climbs per acquired line, ``delta_z``.
        Used only when ``acceleration_z`` is above 1. Default is 1.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired along y, in ``(0.5, 1]``.
        The train is shortened and its origin slid to the trailing lines, so
        the leading (early) lines are dropped and conjugate symmetry covers
        them. Default is 1.0.
    partial_fourier_z : float, optional
        Fraction of the partition extent acquired along z, in ``(0.5, 1]``.
        The leading shells are dropped. Default is 1.0.
    n_acs : int, optional
        Autocalibration extent along y, in lines. With ``n_acs_z`` it bounds
        the fully sampled central rectangle a short linear train lays down
        ahead of the imaging shots whenever the scan is accelerated, so a
        parallel reconstruction has coil sensitivities to solve against.
        Default is 24.
    n_acs_z : int, optional
        Autocalibration extent along z, in partitions. Default is 16.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth in Hz. Default is 500e3.
    partition_order : str, optional
        Order the partitions are encoded in, one of
        ``pp.calc_traversal_order``'s schemes. ``'center_out'`` (the default)
        puts the centre of k-space on the first, steady-state echoes.
    opposite_reference : bool, optional
        Include the opposite-phase-encode reference when writing the pair.
        Default is True.
    n_dummy : int, optional
        Shots played without acquiring, before the first acquired one, so the
        first acquired shot sees the magnetisation every later one sees. A
        whole shot is the unit: one excitation and its train is what the
        steady state is reached by. Default is 2.
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.
    spoiling_cycles : float, optional
        Cycles of dephasing left at the end of each shot. Default is 4.0.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The main EPI sequence object.

    Examples
    --------
    >>> from pulserver.app import epi3D_sequence
    >>> seq = epi3D_sequence(n_x=32, n_y=32, n_z=4, n_dummy=0)
    >>> seq.num_trs, seq.num_segments
    (4, 1)

    The waveform figures below are one design, prescribed to be *legible*
    rather than diagnostic: the shortest TE and TR the readout admits, so
    nothing waits; heavy spoiling, so that lobe is unmistakable; and a short
    train over four partitions, so one shot and the whole traversal both stay
    readable.

    .. plot::
       :include-source:
       :nofigs:
       :context:

       from pulserver.app import epi3D_sequence

       seq = epi3D_sequence(
           n_x=128, n_y=32, n_z=4, te=None, tr=None, n_acs=0, n_acs_z=0,
           n_dummy=0, spoiling_cycles=6.0,
       )

    **The excitation**, and the magnetisation it leaves behind: the pulse's
    own ``B1`` envelope beside the profile it writes across the selected
    axis.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot_rf(plot_now=False)

    **One shot**, which is the whole repetition: the excitation, then the
    oscillating readout with the phase blips riding its ramps, then the
    spoiler.

    .. plot::
       :include-source:
       :context: close-figs

       seq.plot(tr="worst_case", time_disp="ms", grad_disp="mT/m", plot_now=False)

    **What the scan covers**, on a second design that is accelerated in both
    encoded directions, because that is when the sampling is worth a picture.
    Each cell is one ``(ky, kz)`` position the scan declares it encodes --
    light where a readout reached it, dark where none did. The red path is one
    shot: it steps two lines at a time along ``ky`` and climbs a partition
    with each step, wrapping at the top of its shell. That climb is what the
    CAIPI blips on ``Gz`` do, and it is what puts the aliased copies a
    partition apart instead of on top of one another. The arcs between samples
    are the blips themselves -- the k-space path of a triangular blip is a
    pair of parabolas.

    .. plot::
       :include-source:
       :context: close-figs

       sampled = epi3D_sequence(
           n_x=64, n_y=32, n_z=12, te=None, tr=None, n_acs=0, n_acs_z=0,
           n_dummy=0, acceleration=2, acceleration_z=2, caipi_shift=1,
       )
       sampled.plot_kspace(plane="yz", lattice=True, plot_now=False)

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
    This design asks for the shortest timing the hardware admits, so its blips
    and readout ramps run at the slew limit. A lower ``MAX_SLEW``, a longer
    echo time, or a narrower readout bandwidth each bring the response down.

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
    kernel = SharedKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=slab_thickness,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        segments=segments,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        caipi_shift=caipi_shift,
        partial_fourier=partial_fourier,
        partial_fourier_z=partial_fourier_z,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        spsp=spsp,
    )
    epi = kernel.epi
    fov_x, fov_y = kernel.fov

    seq = pp.Sequence(system)
    rev_label = pp.make_label("REV", "SET", 0)
    rep_label = pp.make_label("REP", "SET", 0)

    # One shot per (segment, shell): a shell is a band of ``acceleration_z``
    # partitions the CAIPI sawtooth walks within the train, so ``n_z // Rz``
    # shells tile the lattice. With no z acceleration a shell is one partition
    # and this is a plain stack of EPI trains. Partial Fourier along z keeps the
    # trailing shells only; the retained shells run centre-out by default, so
    # the first echoes -- the ones the steady state and the contrast follow --
    # sit at the centre of k-space. Partial Fourier along y rode into the train
    # length, so it needs only its origin ``first_y`` here.
    kept = list(range(kernel.first_shell, kernel.n_shells))
    shells = [kept[i] for i in pp.calc_traversal_order(len(kept), partition_order)]
    # Steady state first: whole shots without acquiring, before the first
    # acquired one.
    #
    # `ONCE` is what keeps them out of the averages: 1 plays on the first pass
    # only, and the first acquired shot clears it back to 0 so the body
    # repeats.
    mark = pp.make_label("ONCE", "SET", 1)
    for _ in range(n_dummy):
        ShotKernel(
            seq,
            epi,
            origin=(kernel.first_y, shells[0] * acceleration_z),
            grid=(n_y, n_z),
            rev_label=rev_label,
            acquire=False,
            first_extra=(mark,) if mark is not None else (),
        )
        mark = None

    clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
    for repetition in range(n_repetitions):
        rep_label.value = int(repetition)
        # Imaging only: coil maps come from the separate calibration sequence,
        # so this file is one clean repeating unit.
        for segment in range(segments):
            for shell in shells:
                ShotKernel(
                    seq,
                    epi,
                    origin=(
                        kernel.first_y + segment * acceleration,
                        shell * acceleration_z,
                    ),
                    grid=(n_y, n_z),
                    rev_label=rev_label,
                    extra_line_labels=(rep_label,),
                    first_extra=(clear_once,) if clear_once is not None else (),
                )
                clear_once = None

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
    seq.set_definition(key="Name", value="epi_3d")
    seq.set_definition(key="TE", value=epi.echo_times[len(epi.order) // 2])
    seq.set_definition(key="EchoSpacing", value=epi.esp)
    seq.set_definition(key="EPIFactor", value=epi.etl)
    seq.set_definition(
        key="NumGainCalibrationReadouts", value=n_gain_calibration_readouts
    )

    if write_seq:
        write_pair(
            seq,
            seq_filename,
            system=system,
            fov=fov,
            fov_offset=fov_offset,
            spsp=spsp,
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            slab_thickness=slab_thickness,
            flip_angle_deg=flip_angle_deg,
            te=te,
            tr=tr,
            segments=segments,
            acceleration=acceleration,
            acceleration_z=acceleration_z,
            caipi_shift=caipi_shift,
            partial_fourier=partial_fourier,
            partial_fourier_z=partial_fourier_z,
            n_acs=n_acs,
            n_acs_z=n_acs_z,
            readout_bandwidth_hz=readout_bandwidth_hz,
            opposite_reference=opposite_reference,
            spoiling_cycles=spoiling_cycles,
        )

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def SlabExcitationKernel(
    system: pp.Opts,
    flip_angle_deg: float,
    thickness_m: float,
    spsp: bool = False,
):
    """The slab excitation, spectral-spatial when ``spsp`` is set.

    Returns ``(excitation, rf, gz)``. The selection gradient carries its own
    rephaser folded onto the end -- as a slab excitation does -- so the 3D
    train's z prewinder block, which already encodes the partition, never
    holds a second z gradient.
    """
    if spsp:
        # Fat's offset is a frequency, so it is field dependent; ppm times the
        # Larmor frequency is what makes the script B0 independent. A spectral
        # passband about that wide leaves water in the passband and fat in the
        # stopband.
        fat_offset_hz = FAT_SHIFT_PPM * 1e-6 * system.gamma * system.B0
        excitation = design.SpspExcitation(
            system,
            flip_angle_deg,
            thickness_m=thickness_m,
            spectral_bandwidth_hz=abs(fat_offset_hz),
            freq_offset_hz=0.0,
            is_slab=True,
        )
        return excitation, excitation.rf, excitation.gz
    excitation = design.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        thickness_m,
        duration_s=PULSE_DURATION,
        is_slab=True,
        time_bw_product=TIME_BW_PRODUCT,
    )
    return excitation, excitation.rf, excitation.gz


def SharedKernel(
    system: pp.Opts,
    *,
    fov,
    n_x: int,
    n_y: int,
    n_z: int,
    slab_thickness: float,
    flip_angle_deg: float,
    te: float | None = None,
    tr: float | None = None,
    segments: int = 1,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 1,
    partial_fourier: float = 1.0,
    partial_fourier_z: float = 1.0,
    readout_bandwidth_hz: float = 500e3,
    spoiling_cycles: float = 4.0,
    spsp: bool = False,
) -> SimpleNamespace:
    """The slab excitation and train both sequences are built from.

    The partition axis is undersampled by ``acceleration_z`` with the CAIPI
    sawtooth of :func:`pp.calc_epi_order`: with no z acceleration the train is
    plain segmented EPI, and above it the ``'caipi'`` shell that tiles a
    :func:`pp.make_caipirinha_mask` lattice.

    Partial Fourier drops the leading (early) end of each phase-encode axis and
    lets conjugate symmetry cover it: ``partial_fourier`` shortens the train and
    slides its ``ky`` origin (``first_y``) up to the retained trailing fraction,
    ``partial_fourier_z`` drops the leading shells (``first_shell``). The centre
    of k-space, which the two fractions keep, is what the reconstruction fills
    from.

    Coil sensitivities are calibrated from a separate low-resolution Cartesian
    gradient echo -- :func:`calibration`, a standalone sequence in the linked
    collection -- rather than from an autocalibration block folded into this
    train, so the imaging file is one clean repeating unit and the calibration
    keeps EPI distortion out of the maps.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov
    scheme = "caipi" if acceleration_z > 1 else "linear"

    # Partial Fourier along y shortens the train and shifts its origin: keep the
    # trailing fraction of the echoes, dropping whole lattice steps so the
    # retained lattice stays aligned.
    etl_full = -(-n_y // (acceleration * segments))
    etl = max(1, round(partial_fourier * etl_full))
    first_y = max(0, n_y - etl * acceleration * segments)

    # Partial Fourier along z drops the leading shells.
    n_shells = n_z // acceleration_z
    first_shell = n_shells - max(1, round(partial_fourier_z * n_shells))

    excitation, exc_rf, exc_gz = SlabExcitationKernel(
        system, flip_angle_deg, slab_thickness, spsp
    )
    epi = design.EpiReadout3D(
        system,
        exc_rf,
        exc_gz,
        fov=(fov_x, fov_y, slab_thickness),
        matrix=(n_x, n_y, n_z),
        scheme=scheme,
        etl=etl,
        segments=segments,
        acceleration=acceleration,
        partition_acceleration=acceleration_z,
        caipi_shift=caipi_shift,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        labels=("LIN", "PAR"),
    )
    return SimpleNamespace(
        excitation=excitation,
        epi=epi,
        first_y=first_y,
        first_shell=first_shell,
        n_shells=n_shells,
        fov=(fov_x, fov_y),
    )


def ShotKernel(
    seq,
    epi,
    *,
    origin: tuple[int, int] | None,
    grid: tuple[int, int],
    rev_label,
    extra_line_labels=(),
    invert_phase: bool = False,
    blip_nulled: bool = False,
    n_lines: int | None = None,
    acquire: bool = True,
    first_extra=(),
) -> None:
    """One excitation and its train, per the module's loop contract."""
    n_y, n_z = grid
    sign = -1.0 if invert_phase else 1.0
    count = epi.etl if n_lines is None else n_lines

    seq.add_block(epi.rf, epi.gz, *first_extra)
    if getattr(epi, "wait_te", None) is not None:
        seq.add_block(epi.wait_te)

    if origin is None:
        ky_scale, kz_scale = 0.0, 0.0
    else:
        line, partition = origin
        ky_scale = sign * (line - n_y / 2) / (n_y / 2)
        kz_scale = (partition - n_z / 2) / (n_z / 2)
        epi.shot_labels[0].value = int(line)
        epi.shot_labels[1].value = int(partition)
    seq.add_block(
        epi.gx_pre,
        pp.scale_grad(epi.gy_pre, ky_scale),
        pp.scale_grad(epi.gz_pre, kz_scale),
        *(epi.shot_labels if acquire and origin is not None else ()),
    )
    for line in range(count):
        rev_label.value = int(line % 2)
        events = [epi.gx[line]]
        if acquire:
            events.extend((epi.adc, rev_label))
        for blip in (epi.gy_blips[line], epi.gz_blips[line]):
            if blip is not None and not blip_nulled:
                events.append(blip if not invert_phase else pp.scale_grad(blip, -1.0))
        if acquire:
            events.extend(epi.line_labels[line] if origin is not None else ())
            events.extend(extra_line_labels)
        seq.add_block(*events)
    seq.add_block(
        epi.gx_spoil,
        pp.scale_grad(epi.gy_rew, ky_scale),
        pp.scale_grad(epi.gz_rew, kz_scale),
    )
    if getattr(epi, "wait_tr", None) is not None:
        seq.add_block(epi.wait_tr)


def NavigatorKernel(
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 32,
    slab_thickness: float = 96e-3,
    flip_angle_deg: float = 20.0,
    te: float | None = None,
    tr: float | None = None,
    segments: int = 1,
    acceleration: int = 1,
    acceleration_z: int = 1,
    caipi_shift: int = 1,
    readout_bandwidth_hz: float = 500e3,
    opposite_reference: bool = True,
    spoiling_cycles: float = 4.0,
    spsp: bool = False,
) -> pp.Sequence:
    """Build the navigator sequence for the 3D EPI.

    Blip-nulled ``NAV``/``REF`` lines at the centre partition, then the
    opposite-phase-encode ``SET = 1`` shot. Parameters mirror :func:`main`.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The navigator sequence.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
    kernel = SharedKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        slab_thickness=slab_thickness,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        segments=segments,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        caipi_shift=caipi_shift,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        spsp=spsp,
    )
    epi = kernel.epi

    seq = pp.Sequence(system)
    rev_label = pp.make_label("REV", "SET", 0)
    nav_label = pp.make_label("NAV", "SET", 1)
    ref_label = pp.make_label("REF", "SET", 1)
    nav_clear = pp.make_label("NAV", "SET", 0)
    ref_clear = pp.make_label("REF", "SET", 0)
    set_label = pp.make_label("SET", "SET", 1)

    ShotKernel(
        seq,
        epi,
        origin=None,
        grid=(n_y, n_z),
        rev_label=rev_label,
        extra_line_labels=(nav_label, ref_label),
        blip_nulled=True,
        n_lines=NAVIGATOR_LINES,
    )
    if opposite_reference:
        ShotKernel(
            seq,
            epi,
            origin=(0, n_z // 2),
            grid=(n_y, n_z),
            rev_label=rev_label,
            extra_line_labels=(nav_clear, ref_clear, set_label),
            invert_phase=True,
        )

    fov_x, fov_y = kernel.fov
    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_z])
    seq.set_definition(key="Name", value="epi_3d_navigator")
    seq.set_definition(key="EchoSpacing", value=epi.esp)
    # The prescription moves this sequence with the imaging it calibrates, and
    # ``compat=False`` because these readouts sample across their read ramps:
    # the k they traced is what a reconstruction needs to put them back on the
    # grid. A saturation band placed at design time carries NOPOS/NOROT and is
    # left where it was put.
    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset),
        system=system,
        compat=False,
    ).apply_to_sequence(seq, in_place=True)

    return seq


def CalibrationKernel(
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 32,
    slab_thickness: float = 96e-3,
    flip_angle_deg: float = 20.0,
    n_acs: int = 24,
    n_acs_z: int = 16,
    readout_bandwidth_hz: float = 500e3,
    spoiling_cycles: float = 4.0,
    spsp: bool = False,
) -> pp.Sequence:
    """A low-resolution Cartesian gradient echo over the centre of k-space.

    The autocalibration scan for the parallel-imaging reconstruction: a fully
    sampled ``n_acs x n_acs_z`` block, one line per repetition, marked ``REF``
    (``ACQ_IS_PARALLEL_CALIBRATION``) so a reconstruction reads it as coil
    calibration only -- it never becomes imaging data. A plain gradient echo
    rather than an EPI train, so the maps carry no EPI distortion, and its own
    sequence in the linked collection so the imaging file stays one clean
    repeating unit. Parameters mirror :func:`main`.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The calibration sequence, or ``None`` when no calibration is asked for.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    acs_lines = pp.calc_calibration_lines(n_y, n_acs)
    acs_partitions = pp.calc_calibration_lines(n_z, n_acs_z)
    if not acs_lines or not acs_partitions:
        return None

    _, exc_rf, exc_gz = SlabExcitationKernel(
        system, flip_angle_deg, slab_thickness, spsp
    )
    readout = design.LineReadout3D(
        system,
        exc_rf,
        exc_gz,
        fov=(fov_x, fov_y, slab_thickness),
        matrix=(n_x, n_y, n_z),
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        labels=("LIN", "PAR"),
    )

    seq = pp.Sequence(system)
    ref_label = pp.make_label("REF", "SET", 1)
    lin_label, par_label = readout.adc_labels
    wait_te = getattr(readout, "wait_te", None)
    wait_tr = getattr(readout, "wait_tr", None)
    # Partitions outer, lines inner: a standard centre-block raster.
    for partition in acs_partitions:
        kz = (partition - n_z / 2) / (n_z / 2)
        for line in acs_lines:
            ky = (line - n_y / 2) / (n_y / 2)
            lin_label.value = int(line)
            par_label.value = int(partition)
            seq.add_block(readout.rf, readout.gz)
            if wait_te is not None:
                seq.add_block(wait_te)
            seq.add_block(
                readout.gx_pre,
                pp.scale_grad(readout.gy_pre, ky),
                pp.scale_grad(readout.gz_pre, kz),
            )
            seq.add_block(readout.gx, readout.adc, ref_label, lin_label, par_label)
            seq.add_block(
                readout.gx_spoil,
                pp.scale_grad(readout.gy_rew, ky),
                pp.scale_grad(readout.gz_rew, kz),
            )
            if wait_tr is not None:
                seq.add_block(wait_tr)

    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_z])
    seq.set_definition(key="Name", value="epi_3d_calibration")
    # The prescription moves this sequence with the imaging it calibrates, and
    # ``compat=False`` because these readouts sample across their read ramps:
    # the k they traced is what a reconstruction needs to put them back on the
    # grid. A saturation band placed at design time carries NOPOS/NOROT and is
    # left where it was put.
    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset),
        system=system,
        compat=False,
    ).apply_to_sequence(seq, in_place=True)

    return seq


def needs_calibration(kwargs: dict) -> bool:
    """Whether the scan is undersampled enough to want a coil calibration."""
    return (
        kwargs.get("acceleration", 1) > 1
        or kwargs.get("acceleration_z", 1) > 1
        or kwargs.get("partial_fourier", 1.0) < 1.0
        or kwargs.get("partial_fourier_z", 1.0) < 1.0
    )


def write_pair(
    main_seq: pp.Sequence, seq_filename: str, *, offline: bool = True, **kwargs
) -> tuple[str, ...]:
    """Write the linked collection: calibration, then navigator, then main.

    Each sequence in the chain points ``NextSequence`` at the next, so the
    interpreter's Sequence Collection plays them in order as one scan while
    keeping each -- the low-resolution coil calibration, the phase navigator,
    the imaging -- its own well-formed repeating unit. The calibration leads
    only when the scan is accelerated enough to ask for one; otherwise the
    chain is navigator then main, as it was.

    Parameters
    ----------
    main_seq : pulserver.pypulseq.Sequence
        The imaging acquisition.
    seq_filename : str
        Where the chain's first sequence is written; the others go beside it as
        ``<stem>_navigator.seq`` and ``<stem>_main.seq``.
    offline : bool, optional
        The form every file of the chain is written in, as
        :func:`pulserver.design.write_sequence` reads it. Default is True.
    **kwargs
        Forwarded to :func:`calibration` and :func:`navigator`, each taking the
        subset it declares.

    Returns
    -------
    tuple of str
        The written paths, in chain order.
    """
    path = Path(seq_filename)
    main_path = path.with_name(path.stem + "_main.seq")
    nav_path = path.with_name(path.stem + "_navigator.seq")

    # The calibration leads the chain only when the imaging is undersampled; a
    # fully sampled scan reconstructs from itself and needs no coil maps.
    system = kwargs.get("system")
    calib = (
        CalibrationKernel(
            system=system,
            **{
                name: value
                for name, value in kwargs.items()
                if name in CALIBRATION_ARGUMENTS
            },
        )
        if needs_calibration(kwargs)
        else None
    )
    nav = NavigatorKernel(
        system=system,
        **{
            name: value for name, value in kwargs.items() if name in NAVIGATOR_ARGUMENTS
        },
    )

    # calibration -> navigator -> main; the calibration drops out when absent,
    # and the chain's first sequence is written at ``seq_filename``.
    chain: list[tuple[pp.Sequence, Path]] = []
    if calib is not None:
        chain.append((calib, path))
        chain.append((nav, nav_path))
    else:
        chain.append((nav, path))
    chain.append((main_seq, main_path))

    for index, (seq, seq_path) in enumerate(chain):
        if index + 1 < len(chain):
            seq.set_definition(key="NextSequence", value=chain[index + 1][1].name)
        write_sequence(seq, str(seq_path), offline=offline)
    return tuple(str(seq_path) for _, seq_path in chain)


# ======================================================================
# The scanner protocol contract
# ======================================================================


class Epi3D(SequencePlugin):
    """The 3D EPI behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        controls = {
            UIParam.TE: DropdownFloatParam(
                value=-1.0,
                min=-1.0,
                max=150.0,
                incr=0.1,
                unit="ms",
                options=[TEPreset.MINIMUM, 20.0, 25.0, 30.0, 40.0],
            ),
            UIParam.TR: DropdownFloatParam(
                value=-1.0,
                min=-1.0,
                max=10000.0,
                incr=1.0,
                unit="ms",
                options=[TRPreset.MINIMUM, 50.0, 60.0, 80.0, 100.0],
            ),
            UIParam.FLIP: DropdownFloatParam(
                value=20.0,
                min=5.0,
                max=90.0,
                incr=1.0,
                unit="deg",
                options=[12.0, 15.0, 20.0, 25.0, 30.0],
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
                value=3.0,
                min=0.5,
                max=10.0,
                incr=0.1,
                unit="mm",
                options=[1.5, 2.0, 3.0, 4.0, 5.0],
            ),
            UIParam.NX: DropdownIntParam(
                value=128, min=16, max=256, incr=1, options=[64, 96, 128, 192]
            ),
            UIParam.NY: DropdownIntParam(
                value=128, min=16, max=256, incr=1, options=[64, 96, 128, 192]
            ),
            UIParam.NSLICES: DropdownIntParam(
                value=32, min=4, max=128, incr=1, options=[16, 24, 32, 48, 64]
            ),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=500e3, min=100e3, max=1000e3, incr=1000.0, unit="Hz"
            ),
            UIParam.RY: TypeinFloatParam(
                value=1.0, min=1.0, max=8.0, incr=1.0, unit=""
            ),
            UIParam.RZ: TypeinFloatParam(
                value=1.0, min=1.0, max=8.0, incr=1.0, unit=""
            ),
            UIParam.NUM_FRAMES: DropdownIntParam(
                value=1, min=1, max=1024, incr=1, options=[1, 10, 100, 300, 600]
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
            UIParam.user_name(0): Description(text="Segments"),
            UIParam.user_value(0): TypeinFloatParam(
                value=1.0, min=1.0, max=32.0, incr=1.0, unit=""
            ),
            UIParam.user_name(1): Description(text="Opposite-PE reference"),
            UIParam.user_value(1): TypeinFloatParam(
                value=1.0, min=0.0, max=1.0, incr=1.0, unit=""
            ),
            UIParam.user_name(2): Description(text="CAIPI shift (kz per ky)"),
            UIParam.user_value(2): TypeinFloatParam(
                value=1.0, min=0.0, max=8.0, incr=1.0, unit=""
            ),
            UIParam.user_name(3): Description(text="ACS lines (y)"),
            UIParam.user_value(3): TypeinFloatParam(
                value=24.0, min=0.0, max=256.0, incr=1.0, unit="lines"
            ),
            UIParam.user_name(4): Description(text="ACS partitions (z)"),
            UIParam.user_value(4): TypeinFloatParam(
                value=16.0, min=0.0, max=128.0, incr=1.0, unit="lines"
            ),
            UIParam.user_name(5): Description(text="Partial Fourier (y)"),
            UIParam.user_value(5): TypeinFloatParam(
                value=1.0, min=0.75, max=1.0, incr=0.05, unit=""
            ),
            UIParam.user_name(6): Description(text="Partial Fourier (z)"),
            UIParam.user_value(6): TypeinFloatParam(
                value=1.0, min=0.75, max=1.0, incr=0.05, unit=""
            ),
        }
        # The multiphase control is shown only when the fMRI time series is on.
        if not ENABLE_MULTIPHASE:
            controls.pop(UIParam.NUM_FRAMES, None)
        return protocol_to_dict(controls)

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = protocol_kwargs(system, protocol)
        try:
            kernel = SharedKernel(
                system,
                **{
                    name: value
                    for name, value in kwargs.items()
                    if name in KERNEL_ARGUMENTS
                },
            )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        n_kept = kernel.n_shells - kernel.first_shell
        shots = kwargs.get("segments", 1) * n_kept
        per_rep = shots * kernel.epi.seq.duration()[0]
        duration = kwargs.get("n_repetitions", 1) * per_rep
        if needs_calibration(kwargs):
            calib = CalibrationKernel(
                system=system,
                **{
                    name: value
                    for name, value in kwargs.items()
                    if name in CALIBRATION_ARGUMENTS
                },
            )
            if calib is not None:
                duration += calib.duration()[0]
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
        kwargs = protocol_kwargs(system, protocol)
        seq = main(**kwargs)
        # write_pair filters to what the calibration and navigator each take;
        # pass everything so it can also read the undersampling flags.
        write_pair(seq, output_path, offline=offline, **kwargs)


KERNEL_ARGUMENTS = frozenset(
    (
        "fov",
        "spsp",
        "n_x",
        "n_y",
        "n_z",
        "slab_thickness",
        "flip_angle_deg",
        "te",
        "tr",
        "segments",
        "acceleration",
        "acceleration_z",
        "caipi_shift",
        "partial_fourier",
        "partial_fourier_z",
        "readout_bandwidth_hz",
        "spoiling_cycles",
    )
)

CALIBRATION_ARGUMENTS = frozenset(
    (
        "fov",
        "spsp",
        "fov_offset",
        "n_x",
        "n_y",
        "n_z",
        "slab_thickness",
        "flip_angle_deg",
        "n_acs",
        "n_acs_z",
        "readout_bandwidth_hz",
        "spoiling_cycles",
    )
)

NAVIGATOR_ARGUMENTS = frozenset(
    (
        "fov",
        "spsp",
        "fov_offset",
        "n_x",
        "n_y",
        "n_z",
        "slab_thickness",
        "flip_angle_deg",
        "te",
        "tr",
        "segments",
        "acceleration",
        "acceleration_z",
        "caipi_shift",
        "readout_bandwidth_hz",
        "opposite_reference",
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
        spsp=SPSP_EXCITATION,
        n_z=n_z,
        slab_thickness=n_z * partition_thickness,
        segments=max(1, round(params.user_float(prot, 0, 1.0))),
        opposite_reference=bool(round(params.user_float(prot, 1, 1.0))),
        caipi_shift=max(0, round(params.user_float(prot, 2, 1.0))),
        n_acs=max(0, round(params.user_float(prot, 3, 24.0))),
        n_acs_z=max(0, round(params.user_float(prot, 4, 16.0))),
        partial_fourier=params.user_float(prot, 5, 1.0),
        partial_fourier_z=params.user_float(prot, 6, 1.0),
        n_repetitions=(
            params.param_int(prot, UIParam.NUM_FRAMES) if ENABLE_MULTIPHASE else 1
        ),
    )


PLUGIN = Epi3D()


def get_default_protocol(system):
    """Bridge entry point: the plugin's default protocol."""
    return PLUGIN.get_default_protocol(system)


def validate_protocol(system, protocol):
    """Bridge entry point: protocol feasibility and scan duration."""
    return PLUGIN.validate_protocol(system, protocol)


def make_sequence(system, protocol, output_path):
    """Bridge entry point: write the linked ``.seq`` pair."""
    return PLUGIN.make_sequence(system, protocol, output_path)


ARG_MAP = [
    ("--te-ms", UIParam.TE, float, "Echo time [ms], or a negative TEPreset"),
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
    ("--frames", UIParam.NUM_FRAMES, int, "Volumes in the time series"),
    ("--offset-x-mm", UIParam.FOV_OFFSET_X, float, "Volume offset along readout [mm]"),
    (
        "--offset-y-mm",
        UIParam.FOV_OFFSET_Y,
        float,
        "Volume offset along phase encode [mm]",
    ),
    ("--offset-z-mm", UIParam.FOV_OFFSET_Z, float, "Volume offset along slab [mm]"),
    ("--segments", UIParam.user_value(0), float, "Interleaved shots per partition"),
    (
        "--no-opposite-reference",
        UIParam.user_value(1),
        lambda value: 0.0 if float(value) else 1.0,
        "Pass 1 to drop the opposite-PE reference from the navigator",
    ),
    ("--caipi-shift", UIParam.user_value(2), float, "CAIPIRINHA kz shift per ky block"),
    ("--acs-lines", UIParam.user_value(3), float, "Autocalibration lines along y"),
    (
        "--acs-partitions",
        UIParam.user_value(4),
        float,
        "Autocalibration partitions along z",
    ),
    (
        "--partial-fourier",
        UIParam.user_value(5),
        float,
        "Acquired y fraction in (0.5, 1]",
    ),
    (
        "--partial-fourier-z",
        UIParam.user_value(6),
        float,
        "Acquired z fraction in (0.5, 1]",
    ),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=ARG_MAP,
            description="Generate a linked 3D EPI navigator + main .seq pair offline.",
            default_output="epi_3d.seq",
        )
    )
