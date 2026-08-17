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
:mod:`pulserver.app.recon.epi2D_recon` reads the navigator groups back through
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

#: Excite several slices at once with a multiband pulse instead of one at a
#: time. A script-level toggle rather than a UI control, because it changes the
#: shape of the acquisition -- a single-band reference pass, then multiband
#: shots whose blipped-CAIPI gz encoding lets a model-based reconstruction tell
#: the collapsed slices apart. The multiband factor itself is a protocol field
#: (``UIParam.MULTIBAND``), so the operator still sets it from the console; this
#: only decides whether the multiband path is taken at all. The slice count must
#: be a whole multiple of the multiband factor.
SMS_EXCITATION = False

#: Offer the fMRI multiphase mode: a time series of ``UIParam.NUM_FRAMES``
#: frames, each carrying its ``REP`` counter. A script-level toggle because a
#: multiphase scan is a different study than a single volume. When ``False`` the
#: frame count is forced to one and the multiphase control is dropped from the
#: protocol, so the console never shows it.
ENABLE_MULTIPHASE = False


def _sms_geometry(n_slices: int, n_bands: int, slice_step: float):
    """Group count, band spacing and slice FOV for a multiband acquisition.

    The ``n_slices`` slices split into ``n_slices // n_bands`` groups; the bands
    of one group are a whole group-count apart, so the excited comb interleaves
    the slices across the slab. ``slice_step`` is the slice-to-slice distance
    (thickness plus gap).
    """
    n_groups = n_slices // n_bands
    band_spacing = n_groups * slice_step
    return n_groups, band_spacing, n_bands * band_spacing


def _sms_kernel(
    system: pp.Opts,
    *,
    fov,
    n_x: int,
    n_y: int,
    slice_thickness: float,
    slice_step: float,
    n_slices: int,
    n_bands: int,
    flip_angle_deg: float,
    te: float | None = None,
    tr: float | None = None,
    segments: int = 1,
    acceleration: int = 1,
    readout_bandwidth_hz: float = 500e3,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """The multiband excitation, its blipped-CAIPI train, and the reference.

    The imaging train is a :class:`design.EpiReadout3D` whose ``n_z`` is the
    multiband factor: its partition axis is not a spatial encode but the CAIPI
    slice phase, so the ``'caipi'`` sawtooth's gz blips give band ``j`` the
    ``exp(i 2*pi*ky*j / n_bands)`` modulation a model-based separation inverts.
    Coil sensitivities come from the separate low-resolution gradient-echo
    calibration (:func:`_add_gre_calibration`), the same block the plain
    accelerated scan uses, so the maps carry no EPI distortion.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov
    n_groups, band_spacing, sms_fov_z = _sms_geometry(n_slices, n_bands, slice_step)

    sms = design.SmsExcitation(
        system,
        flip_angle_deg,
        thickness_m=slice_thickness,
        slice_gap_m=band_spacing,
        n_bands=n_bands,
    )
    # Fold the rephaser onto the selection gradient, is_slab style, so the SMS
    # train's z prewinder block never holds a second z lobe.
    gz = pp.concatenate_gradients(sms.gz, sms.gz_reph, system=system)
    epi = design.EpiReadout3D(
        system,
        sms.rf,
        gz,
        fov=(fov_x, fov_y, sms_fov_z),
        matrix=(n_x, n_y, n_bands),
        scheme="caipi",
        segments=segments,
        acceleration=acceleration,
        partition_acceleration=n_bands,
        caipi_shift=1,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
        labels=("LIN", "PAR"),
    )

    return SimpleNamespace(
        sms=sms,
        epi=epi,
        n_groups=n_groups,
        band_spacing=band_spacing,
        fov=(fov_x, fov_y),
    )


def _sms_group_center(group: int, n_slices: int, n_bands: int, slice_step: float) -> float:
    """Where the multiband comb sits to excite one group's slices (m).

    Group ``g`` holds slices ``g, g + n_groups, ...``; their midpoint is where
    the band comb -- symmetric about its own centre -- has to be placed.
    """
    n_groups = n_slices // n_bands
    first = (group - (n_slices - 1) / 2) * slice_step
    return first + (n_bands - 1) / 2 * n_groups * slice_step


def _play_sms_shot(
    seq,
    epi,
    *,
    n_y: int,
    center_m: float,
    gz_amplitude: float,
    origin_line: int,
    rev_label,
    extra_line_labels=(),
    blip_nulled: bool = False,
    n_lines: int | None = None,
) -> None:
    """One multiband excitation and its blipped-CAIPI train.

    The comb is placed on the group's slices by the excitation frequency
    offset. The partition prewinder is skipped: the shot starts at ``kz = 0``
    and the gz blips walk the CAIPI sawtooth from there, so the partition label
    is the pure slice-phase index a reconstruction reads the modulation from.
    ``blip_nulled`` drops both blips for the phase-correction navigator, and
    ``n_lines`` truncates the train to the leading few those lines need.
    """
    count = epi.etl if n_lines is None else n_lines
    epi.rf.freq_offset = gz_amplitude * center_m
    epi.rf.phase_offset = -2 * np.pi * epi.rf.freq_offset * epi.rf.center

    seq.add_block(epi.rf, epi.gz)
    if getattr(epi, "wait_te", None) is not None:
        seq.add_block(epi.wait_te)

    if blip_nulled:
        seq.add_block(epi.gx_pre)
    else:
        ky_scale = (origin_line - n_y / 2) / (n_y / 2)
        epi.shot_labels[0].value = int(origin_line)
        epi.shot_labels[1].value = 0
        seq.add_block(epi.gx_pre, pp.scale_grad(epi.gy_pre, ky_scale), *epi.shot_labels)
    for line in range(count):
        rev_label.value = int(line % 2)
        events = [epi.gx[line], epi.adc, rev_label]
        if not blip_nulled:
            for blip in (epi.gy_blips[line], epi.gz_blips[line]):
                if blip is not None:
                    events.append(blip)
            events.extend(epi.line_labels[line])
        events.extend(extra_line_labels)
        seq.add_block(*events)
    if blip_nulled:
        seq.add_block(epi.gx_spoil)
    else:
        ky_scale = (origin_line - n_y / 2) / (n_y / 2)
        seq.add_block(epi.gx_spoil, pp.scale_grad(epi.gy_rew, ky_scale))
    if getattr(epi, "wait_tr", None) is not None:
        seq.add_block(epi.wait_tr)


def _sms_calibration(
    system: pp.Opts,
    *,
    fov,
    n_x: int,
    n_y: int,
    slice_thickness: float,
    slice_gap: float,
    n_slices: int,
    n_bands: int,
    flip_angle_deg: float,
    n_acs: int = 24,
    te: float | None = None,
    tr: float | None = None,
    segments: int = 1,
    acceleration: int = 1,
    readout_bandwidth_hz: float = 500e3,
    slice_order: str = "interleaved",
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """The multiband calibration file: phase navigator, then the GRE calibration.

    Three blip-nulled navigator lines (``NAV``) carry the odd/even phase, then
    the same low-resolution slice-selective gradient echo the plain accelerated
    scan calibrates from (:func:`_add_gre_calibration`) is played, a central
    ``n_acs`` block per slice marked ``REF`` (``ACQ_IS_PARALLEL_CALIBRATION``) so
    a reconstruction estimates each slice's coil sensitivities free of EPI
    distortion. Its own sequence in the linked collection, kept out of the
    imaging repeating unit, and played once for the whole time series.
    """
    slice_step = slice_thickness + slice_gap
    kernel = _sms_kernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        slice_thickness=slice_thickness,
        slice_step=slice_step,
        n_slices=n_slices,
        n_bands=n_bands,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        segments=segments,
        acceleration=acceleration,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
    )
    epi = kernel.epi
    gz_amplitude = float(kernel.sms.gz.amplitude)
    fov_x, fov_y = kernel.fov
    n_groups = kernel.n_groups

    seq = pp.Sequence(system)
    rev_label = pp.make_label("REV", "SET", 0)
    sms_off = pp.make_label("SMS", "SET", 0)
    ref_off = pp.make_label("REF", "SET", 0)
    nav_on = pp.make_label("NAV", "SET", 1)
    nav_off = pp.make_label("NAV", "SET", 0)

    # Phase-correction navigator: the multiband readout, blips nulled, at the
    # centre group -- its odd/even fit is the readout's, blips or not.
    _play_sms_shot(
        seq,
        epi,
        n_y=n_y,
        center_m=_sms_group_center(n_groups // 2, n_slices, n_bands, slice_step),
        gz_amplitude=gz_amplitude,
        origin_line=0,
        rev_label=rev_label,
        extra_line_labels=(nav_on, ref_off, sms_off),
        blip_nulled=True,
        n_lines=_NAV_LINES,
    )

    # Coil calibration: the shared low-res GRE, one slice at a time, marked REF.
    # ``nav_off``/``sms_off`` reset the labels the navigator set on the file.
    _add_gre_calibration(
        seq,
        system,
        fov_x=fov_x,
        fov_y=fov_y,
        n_x=n_x,
        n_y=n_y,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        slice_gap=slice_gap,
        flip_angle_deg=flip_angle_deg,
        n_acs=n_acs,
        readout_bandwidth_hz=readout_bandwidth_hz,
        slice_order=slice_order,
        spoiling_cycles=spoiling_cycles,
        extra_labels=(nav_off, sms_off),
    )

    seq.set_definition(key="FOV", value=[fov_x, fov_y, slice_thickness * n_slices])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_slices])
    seq.set_definition(key="Name", value="sms_epi_2d_calibration")
    seq.set_definition(key="EchoSpacing", value=epi.esp)
    return seq


def _sms_main(
    system: pp.Opts,
    *,
    fov,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int,
    n_y: int,
    slice_thickness: float,
    slice_gap: float,
    n_slices: int,
    n_bands: int,
    flip_angle_deg: float,
    n_acs: int = 24,
    te: float | None = None,
    tr: float | None = None,
    n_repetitions: int = 1,
    segments: int = 1,
    acceleration: int = 1,
    readout_bandwidth_hz: float = 500e3,
    slice_order: str = "interleaved",
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """The multiband imaging file: blipped-CAIPI shots, one per (segment, group).

    Imaging only -- the shots are multiband (``SMS``), the group in ``SLC``,
    and the blipped-CAIPI gz encoding is what a model-based separation inverts
    against the calibration. The phase navigator and the coil calibration are
    their own sequence in the linked collection (:func:`_sms_calibration`), so
    this file is one clean repeating unit; ``MultibandFactor`` records it.
    ``n_acs`` is accepted for a uniform call from :func:`main` but unused here,
    the imaging carrying no calibration lines.
    """
    del n_acs
    slice_step = slice_thickness + slice_gap
    kernel = _sms_kernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        slice_thickness=slice_thickness,
        slice_step=slice_step,
        n_slices=n_slices,
        n_bands=n_bands,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        segments=segments,
        acceleration=acceleration,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
    )
    epi = kernel.epi
    gz_amplitude = float(kernel.sms.gz.amplitude)
    fov_x, fov_y = kernel.fov
    n_groups = kernel.n_groups

    seq = pp.Sequence(system)
    rev_label = pp.make_label("REV", "SET", 0)
    slc_label = pp.make_label("SLC", "SET", 0)
    rep_label = pp.make_label("REP", "SET", 0)
    sms_on = pp.make_label("SMS", "SET", 1)
    ref_off = pp.make_label("REF", "SET", 0)
    nav_off = pp.make_label("NAV", "SET", 0)

    # Multiband imaging: one shot per (segment, group), the group in SLC.
    groups = [int(g) for g in pp.calc_traversal_order(n_groups, slice_order)]
    for repetition in range(n_repetitions):
        rep_label.value = int(repetition)
        for segment in range(segments):
            for group in groups:
                slc_label.value = int(group)
                _play_sms_shot(
                    seq,
                    epi,
                    n_y=n_y,
                    center_m=_sms_group_center(
                        group, n_slices, n_bands, slice_step
                    ),
                    gz_amplitude=gz_amplitude,
                    origin_line=segment,
                    rev_label=rev_label,
                    extra_line_labels=(slc_label, rep_label, sms_on, ref_off, nav_off),
                )

    pp.TransformFOV(
        translation=tuple(offset * 1e3 for offset in fov_offset), system=system
    ).apply_to_sequence(seq, in_place=True)

    slab_thickness = n_slices * slice_step - slice_gap
    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_slices])
    seq.set_definition(key="Name", value="sms_epi_2d")
    seq.set_definition(key="TE", value=epi.echo_times[len(epi.order) // 2])
    seq.set_definition(key="EchoSpacing", value=epi.esp)
    seq.set_definition(key="EPIFactor", value=epi.etl)
    seq.set_definition(key="MultibandFactor", value=n_bands)
    return seq


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

    # The slice rephaser runs straight off the selection lobe: with the TE
    # wait when there is one, in the prewinder block otherwise -- the block
    # the module itself puts it in. Without it the slice never refocuses.
    rephaser = [epi.gz_reph] if getattr(epi, "gz_reph", None) is not None else []

    seq.add_block(epi.rf, epi.gz)
    if getattr(epi, "wait_te", None) is not None:
        seq.add_block(epi.wait_te, *rephaser)
        rephaser = []

    scale = 0.0 if origin_line is None else sign * (origin_line - n_y / 2) / (n_y / 2)
    if origin_line is not None:
        epi.shot_labels[0].value = int(origin_line)
    seq.add_block(
        epi.gx_pre,
        pp.scale_grad(epi.gy_pre, scale),
        *rephaser,
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


def _add_gre_calibration(
    seq: pp.Sequence,
    system: pp.Opts,
    *,
    fov_x: float,
    fov_y: float,
    n_x: int,
    n_y: int,
    n_slices: int,
    slice_thickness: float,
    slice_gap: float,
    flip_angle_deg: float,
    n_acs: int,
    readout_bandwidth_hz: float,
    slice_order: str,
    spoiling_cycles: float,
    extra_labels: tuple = (),
) -> list[int]:
    """Add a per-slice low-resolution slice-selective gradient echo to ``seq``.

    A fully sampled central ``n_acs`` block per slice, one line per repetition,
    marked ``REF`` (``ACQ_IS_PARALLEL_CALIBRATION``, coil calibration only) and
    ``SLC``/``LIN``. A plain gradient echo rather than an EPI train keeps EPI
    distortion out of the coil maps. ``extra_labels`` are appended to every ADC
    block, so a caller sharing the file with other data (the multiband
    calibration's phase navigator) can reset the labels those lines set.

    Returns the acquired calibration lines (empty when none are asked for), so
    the caller can decide whether the block was laid down at all.
    """
    acs_lines = pp.calc_calibration_lines(n_y, n_acs)
    if not acs_lines:
        return []

    excitation = design.SpatialSelectiveExcitation(system, flip_angle_deg, slice_thickness)
    readout = design.LineReadout2D(
        system,
        excitation.rf,
        excitation.gz,
        excitation.gz_reph,
        fov=(fov_x, fov_y),
        matrix=(n_x, n_y),
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=spoiling_cycles,
    )
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )

    ref_label = pp.make_label("REF", "SET", 1)
    slc_label = pp.make_label("SLC", "SET", 0)
    lin_label = pp.make_label("LIN", "SET", 0)
    for i_slice in (int(i) for i in pp.calc_traversal_order(n_slices, slice_order)):
        readout.rf.freq_offset = excitation.gz.amplitude * slice_positions[i_slice]
        readout.rf.phase_offset = -2 * np.pi * readout.rf.freq_offset * readout.rf.center
        slc_label.value = int(i_slice)
        for line in acs_lines:
            ky = (line - n_y / 2) / (n_y / 2)
            lin_label.value = int(line)
            seq.add_block(readout.rf, readout.gz)
            seq.add_block(
                readout.gx_pre, pp.scale_grad(readout.gy_pre, ky), readout.gz_reph
            )
            seq.add_block(
                readout.gx, readout.adc, ref_label, slc_label, lin_label, *extra_labels
            )
            seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, ky))
    return acs_lines


def calibration(
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    flip_angle_deg: float = 70.0,
    n_acs: int = 24,
    readout_bandwidth_hz: float = 500e3,
    slice_order: str = "interleaved",
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """A low-resolution slice-selective gradient echo, one slice at a time.

    The autocalibration for the parallel-imaging reconstruction: a fully sampled
    central ``n_acs`` block per slice, every line ``REF``
    (``ACQ_IS_PARALLEL_CALIBRATION``, coil calibration only). A plain gradient
    echo rather than an EPI train keeps EPI distortion out of the coil maps, and
    its own sequence in the linked collection so the imaging stays one clean
    repeating unit. Both the accelerated plain scan and the multiband
    (:func:`_sms_calibration`) path calibrate from this block. Parameters mirror
    :func:`main`.

    Returns
    -------
    seq : pulserver.pypulseq.Sequence
        The calibration sequence, or ``None`` when no calibration is asked for.
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    seq = pp.Sequence(system)
    acs_lines = _add_gre_calibration(
        seq,
        system,
        fov_x=fov_x,
        fov_y=fov_y,
        n_x=n_x,
        n_y=n_y,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        slice_gap=slice_gap,
        flip_angle_deg=flip_angle_deg,
        n_acs=n_acs,
        readout_bandwidth_hz=readout_bandwidth_hz,
        slice_order=slice_order,
        spoiling_cycles=spoiling_cycles,
    )
    if not acs_lines:
        return None

    seq.set_definition(key="FOV", value=[fov_x, fov_y, slice_thickness * n_slices])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_slices])
    seq.set_definition(key="Name", value="epi_2d_calibration")
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
    n_bands: int = 1,
    n_acs: int = 24,
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
    n_acs : int, optional
        Autocalibration extent along y, in lines. Bounds the fully sampled
        central block a separate low-resolution gradient echo
        (:func:`calibration`) lays down per slice ahead of the imaging
        whenever the scan is accelerated, so a parallel reconstruction has
        coil sensitivities to solve against. Default is 24.
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

    # The multiband path is a different acquisition -- a calibration file
    # (phase navigator plus single-band reference) then the blipped-CAIPI
    # imaging -- written as a linked collection, so it branches off before the
    # plain train is built.
    if SMS_EXCITATION and n_bands > 1:
        sms_kwargs = {
            "fov": fov,
            "n_x": n_x,
            "n_y": n_y,
            "slice_thickness": slice_thickness,
            "slice_gap": slice_gap,
            "n_slices": n_slices,
            "n_bands": n_bands,
            "flip_angle_deg": flip_angle_deg,
            "n_acs": n_acs,
            "te": te,
            "tr": tr,
            "segments": segments,
            "acceleration": acceleration,
            "readout_bandwidth_hz": readout_bandwidth_hz,
            "slice_order": slice_order,
            "spoiling_cycles": spoiling_cycles,
        }
        seq = _sms_main(
            system,
            fov_offset=fov_offset,
            n_repetitions=n_repetitions,
            **sms_kwargs,
        )
        if test_report:
            print(seq.test_report())
        if plot:
            seq.plot()
        seq.set_definition(
            key="NumGainCalibrationReadouts",
            value=n_slices if n_gain_calibration_readouts is None else n_gain_calibration_readouts,
        )
        if write_seq:
            _write_sms_collection(seq, seq_filename, system, sms_kwargs)
        return seq

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
            n_acs=n_acs,
            readout_bandwidth_hz=readout_bandwidth_hz,
            opposite_reference=opposite_reference,
            slice_order=slice_order,
            spoiling_cycles=spoiling_cycles,
        )

    return seq


def _needs_calibration(kwargs: dict) -> bool:
    """Whether the scan is undersampled enough to want a coil calibration."""
    return kwargs.get("acceleration", 1) > 1


def write_pair(main_seq: pp.Sequence, seq_filename: str, **kwargs) -> tuple[str, ...]:
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
        calibration(
            system=system,
            **{name: value for name, value in kwargs.items() if name in _CALIBRATION_ARGUMENTS},
        )
        if _needs_calibration(kwargs)
        else None
    )
    nav = navigator(
        system=system,
        **{name: value for name, value in kwargs.items() if name in _NAVIGATOR_ARGUMENTS},
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
        write_sequence(seq, str(seq_path), offline=True)
    return tuple(str(seq_path) for _, seq_path in chain)


#: What :func:`_sms_calibration` takes of :func:`main`'s keywords.
_SMS_CALIBRATION_ARGUMENTS = frozenset(
    (
        "fov",
        "n_x",
        "n_y",
        "slice_thickness",
        "slice_gap",
        "n_slices",
        "n_bands",
        "flip_angle_deg",
        "n_acs",
        "te",
        "tr",
        "segments",
        "acceleration",
        "readout_bandwidth_hz",
        "slice_order",
        "spoiling_cycles",
    )
)


def _write_sms_collection(
    main_seq: pp.Sequence, seq_filename: str, system: pp.Opts, kwargs: dict
) -> tuple[str, str]:
    """Write the multiband collection: calibration first, then imaging.

    The calibration -- phase navigator plus single-band reference -- leads the
    chain at ``seq_filename`` and points ``NextSequence`` at the imaging file
    beside it, so the interpreter plays them in order while each stays its own
    repeating unit.
    """
    path = Path(seq_filename)
    main_path = path.with_name(path.stem + "_main.seq")
    calib = _sms_calibration(
        system,
        **{name: value for name, value in kwargs.items() if name in _SMS_CALIBRATION_ARGUMENTS},
    )
    calib.set_definition(key="NextSequence", value=main_path.name)
    write_sequence(calib, str(path), offline=True)
    write_sequence(main_seq, str(main_path), offline=True)
    return str(path), str(main_path)


class Epi2D(SequencePlugin):
    """The 2D EPI behind the scanner protocol contract."""

    def get_default_protocol(self, system: pp.Opts) -> dict[str, dict]:
        """Return the protocol the scanner UI is built from."""
        controls = {
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
                UIParam.MULTIBAND: TypeinFloatParam(
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
                UIParam.user_name(2): Description(text="ACS lines (y)"),
                UIParam.user_value(2): TypeinFloatParam(
                    value=24.0, min=0.0, max=256.0, incr=1.0, unit="lines"
                ),
        }
        # The multiphase control is shown only when the fMRI time series is on.
        if not ENABLE_MULTIPHASE:
            controls.pop(UIParam.NUM_FRAMES, None)
        return protocol_to_dict(controls)

    def validate_protocol(self, system: pp.Opts, protocol: dict[str, dict]) -> dict:
        """Report whether the protocol is feasible, and how long it will take."""
        system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
        kwargs = _main_kwargs(system, protocol)
        n_bands = kwargs.get("n_bands", 1)
        multiband = SMS_EXCITATION and n_bands > 1
        n_slices = kwargs.get("n_slices", 1)
        if multiband and n_slices % n_bands:
            return {
                "valid": False,
                "duration": None,
                "info": f"slice count {n_slices} is not a multiple of the multiband factor {n_bands}",
            }
        try:
            if multiband:
                kernel = _sms_kernel(
                    system,
                    slice_step=kwargs["slice_thickness"] + kwargs.get("slice_gap", 0.0),
                    n_slices=n_slices,
                    n_bands=n_bands,
                    **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
                )
            else:
                kernel = _shared_kernel(
                    system,
                    **{name: value for name, value in kwargs.items() if name in _KERNEL_ARGUMENTS},
                )
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}

        if multiband:
            imaging = kwargs.get("segments", 1) * kernel.n_groups
            per_rep = imaging * kernel.epi.seq.duration()[0]
            # The coil calibration is the shared low-res GRE, played once.
            calib = calibration(
                system=system,
                **{name: value for name, value in kwargs.items() if name in _CALIBRATION_ARGUMENTS},
            )
            calib_duration = calib.duration()[0] if calib is not None else 0.0
            duration = calib_duration + kwargs.get("n_repetitions", 1) * per_rep
            info = (
                f"TA = {duration:.1f} s, MB = {n_bands}, ETL = {kernel.epi.etl}, "
                f"ESP = {kernel.epi.esp * 1e3:.2f} ms"
            )
        else:
            shots = kwargs.get("segments", 1) * n_slices
            duration = (
                kwargs.get("n_repetitions", 1) * shots * kernel.epi.seq.duration()[0]
            )
            info = (
                f"TA = {duration:.1f} s, ETL = {kernel.epi.etl}, "
                f"ESP = {kernel.epi.esp * 1e3:.2f} ms"
            )
        return {"valid": True, "duration": duration, "info": info}

    def make_sequence(
        self,
        system: pp.Opts,
        protocol: dict[str, dict],
        output_path: str,
        *,
        offline: bool = False,
    ) -> None:
        """Write the acquisition as a linked collection: a navigator+main pair,
        or a calibration+main pair when the SMS path is on."""
        del offline  # the chain is followed from files, so both are written
        kwargs = _main_kwargs(system, protocol)
        seq = main(**kwargs)
        if SMS_EXCITATION and kwargs.get("n_bands", 1) > 1:
            system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)
            _write_sms_collection(seq, output_path, system, kwargs)
            return
        # write_pair filters to what the calibration and navigator each take;
        # pass everything so it can also read the undersampling flag.
        write_pair(seq, output_path, system=system, **kwargs)


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

_CALIBRATION_ARGUMENTS = frozenset(
    (
        "fov",
        "n_x",
        "n_y",
        "n_slices",
        "slice_thickness",
        "slice_gap",
        "flip_angle_deg",
        "n_acs",
        "readout_bandwidth_hz",
        "slice_order",
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
        "slice_order",
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
        n_acs=max(0, round(params.user_float(prot, 2, 24.0))),
        n_bands=max(1, round(params.param_float_optional(prot, UIParam.MULTIBAND, 1.0))),
        n_repetitions=(
            params.param_int(prot, UIParam.NUM_FRAMES) if ENABLE_MULTIPHASE else 1
        ),
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
    ("--multiband", UIParam.MULTIBAND, float, "Simultaneous-multislice band count (SMS)"),
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
    ("--acs-lines", UIParam.user_value(2), float, "Autocalibration lines along y"),
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
