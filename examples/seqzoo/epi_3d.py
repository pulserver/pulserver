"""3D gradient-echo EPI, slab-selective, exported as a linked pair.

The volume counterpart of :mod:`pulserver.seqzoo.epi_2d`: one
:class:`design.EpiReadout3D` train per ``(segment, shell)``, blips on the
read ramps, ``REV`` on every line, the partition carried as ``PAR``. A shell
is one partition for a plain stack of trains, or a band of ``Rz`` partitions
the CAIPI sawtooth walks for segmented blipped-CAIPI. The
navigator -- blip-nulled ``NAV``/``REF`` lines at the centre partition plus
the opposite-phase-encode ``SET = 1`` reference -- is its own sequence,
linked ahead of the main file through ``NextSequence``, exercising the
Sequence Collection path. :mod:`pulserver.reczoo.epi_3d` reads both back.

``main`` returns the main :class:`pulserver.pypulseq.Sequence`; ``PLUGIN``
writes the linked pair.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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
    n_acs: int = 24,
    n_acs_z: int = 16,
    readout_bandwidth_hz: float = 500e3,
    spoiling_cycles: float = 4.0,
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

    An undersampled scan -- accelerated or partial Fourier -- also builds a
    ``calibration`` train: a short linear, fully sampled EPI train over the
    central ``n_acs`` lines, played once per central partition to lay down the
    autocalibration rectangle a parallel reconstruction estimates its coil
    sensitivities from. The rectangle's lines and partitions come from
    :func:`pp.calc_calibration_lines`, so the sequence and
    :mod:`pulserver.reczoo.epi_3d` read the same window.
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

    excitation = design.SpatialSelectiveExcitation(
        system, flip_angle_deg, slab_thickness, is_slab=True
    )
    epi = design.EpiReadout3D(
        system,
        excitation.rf,
        excitation.gz,
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

    # The autocalibration rectangle, acquired only when the sampling misses the
    # centre's surroundings -- a fully sampled scan calibrates from itself. Its
    # extent is the same builtin window the reconstruction reads off the mask.
    accelerated = (
        acceleration > 1
        or acceleration_z > 1
        or partial_fourier < 1.0
        or partial_fourier_z < 1.0
    )
    acs_lines = pp.calc_calibration_lines(n_y, n_acs) if accelerated else []
    acs_partitions = pp.calc_calibration_lines(n_z, n_acs_z) if accelerated else []
    calibration = None
    if acs_lines and acs_partitions:
        calibration = design.EpiReadout3D(
            system,
            excitation.rf,
            excitation.gz,
            fov=(fov_x, fov_y, slab_thickness),
            matrix=(n_x, n_y, n_z),
            scheme="linear",
            etl=len(acs_lines),
            te=te,
            tr=tr,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            labels=("LIN", "PAR"),
        )
    return SimpleNamespace(
        excitation=excitation,
        epi=epi,
        calibration=calibration,
        acs_lines=acs_lines,
        acs_partitions=acs_partitions,
        first_y=first_y,
        first_shell=first_shell,
        n_shells=n_shells,
        fov=(fov_x, fov_y),
    )


def _play_shot(
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
) -> None:
    """One excitation and its train, per the module's loop contract."""
    n_y, n_z = grid
    sign = -1.0 if invert_phase else 1.0
    count = epi.etl if n_lines is None else n_lines

    seq.add_block(epi.rf, epi.gz)
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
        *(epi.shot_labels if origin is not None else ()),
    )
    for line in range(count):
        rev_label.value = int(line % 2)
        events = [epi.gx[line], epi.adc, rev_label]
        for blip in (epi.gy_blips[line], epi.gz_blips[line]):
            if blip is not None and not blip_nulled:
                events.append(blip if not invert_phase else pp.scale_grad(blip, -1.0))
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


def navigator(
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
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
    kernel = _shared_kernel(
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
        n_acs=0,
        n_acs_z=0,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
    )
    epi = kernel.epi

    seq = pp.Sequence(system)
    rev_label = pp.make_label("REV", "SET", 0)
    nav_label = pp.make_label("NAV", "SET", 1)
    ref_label = pp.make_label("REF", "SET", 1)
    nav_clear = pp.make_label("NAV", "SET", 0)
    ref_clear = pp.make_label("REF", "SET", 0)
    set_label = pp.make_label("SET", "SET", 1)

    _play_shot(
        seq,
        epi,
        origin=None,
        grid=(n_y, n_z),
        rev_label=rev_label,
        extra_line_labels=(nav_label, ref_label),
        blip_nulled=True,
        n_lines=_NAV_LINES,
    )
    if opposite_reference:
        _play_shot(
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
    return seq


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
    n_gain_calibration_readouts: int = 1,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Create the main 3D EPI sequence.

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
    n_gain_calibration_readouts : int, optional
        Written as the ``NumGainCalibrationReadouts`` definition. Default
        is 1.
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
        spoiling_cycles=spoiling_cycles,
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
    for repetition in range(n_repetitions):
        rep_label.value = int(repetition)
        # The autocalibration rectangle leads each repetition: a fully sampled
        # centre every reconstructed volume can estimate its own coil maps
        # from. It is present only when the imaging lattice is undersampled.
        if kernel.calibration is not None:
            for partition in kernel.acs_partitions:
                _play_shot(
                    seq,
                    kernel.calibration,
                    origin=(kernel.acs_lines[0], partition),
                    grid=(n_y, n_z),
                    rev_label=rev_label,
                    extra_line_labels=(rep_label,),
                )
        for segment in range(segments):
            for shell in shells:
                _play_shot(
                    seq,
                    epi,
                    origin=(kernel.first_y + segment, shell * acceleration_z),
                    grid=(n_y, n_z),
                    rev_label=rev_label,
                    extra_line_labels=(rep_label,),
                )

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset), system=system
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
            opposite_reference=opposite_reference,
            spoiling_cycles=spoiling_cycles,
        )

    return seq


def write_pair(main_seq: pp.Sequence, seq_filename: str, **navigator_kwargs) -> tuple[str, str]:
    """Write navigator and main as a linked pair, navigator first.

    Parameters
    ----------
    main_seq : pulserver.pypulseq.Sequence
        The main acquisition.
    seq_filename : str
        Where the navigator is written; the main goes beside it as
        ``<stem>_main.seq`` and the navigator's ``NextSequence`` names it.
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


class Epi3D(SequencePlugin):
    """The 3D EPI behind the scanner protocol contract."""

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

        n_kept = kernel.n_shells - kernel.first_shell
        shots = kwargs.get("segments", 1) * n_kept
        per_rep = shots * kernel.epi.seq.duration()[0]
        if kernel.calibration is not None:
            per_rep += len(kernel.acs_partitions) * kernel.calibration.seq.duration()[0]
        duration = kwargs.get("n_repetitions", 1) * per_rep
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
        del offline
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
        "n_acs",
        "n_acs_z",
        "readout_bandwidth_hz",
        "spoiling_cycles",
    )
)

_NAVIGATOR_ARGUMENTS = frozenset(
    (
        "fov",
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


def _main_kwargs(system: pp.Opts, protocol: dict[str, dict]) -> dict:
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
        segments=max(1, round(params.user_float(prot, 0, 1.0))),
        opposite_reference=bool(round(params.user_float(prot, 1, 1.0))),
        caipi_shift=max(0, round(params.user_float(prot, 2, 1.0))),
        n_acs=max(0, round(params.user_float(prot, 3, 24.0))),
        n_acs_z=max(0, round(params.user_float(prot, 4, 16.0))),
        partial_fourier=params.user_float(prot, 5, 1.0),
        partial_fourier_z=params.user_float(prot, 6, 1.0),
        n_repetitions=params.param_int(prot, UIParam.NUM_FRAMES),
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


_ARG_MAP = [
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
    ("--offset-y-mm", UIParam.FOV_OFFSET_Y, float, "Volume offset along phase encode [mm]"),
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
    ("--acs-partitions", UIParam.user_value(4), float, "Autocalibration partitions along z"),
    ("--partial-fourier", UIParam.user_value(5), float, "Acquired y fraction in (0.5, 1]"),
    ("--partial-fourier-z", UIParam.user_value(6), float, "Acquired z fraction in (0.5, 1]"),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description="Generate a linked 3D EPI navigator + main .seq pair offline.",
            default_output="epi_3d.seq",
        )
    )
