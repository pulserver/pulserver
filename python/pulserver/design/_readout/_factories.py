"""Factories for public readout modules.

Each factory returns a stateful :class:`~pulserver.Module` holding one whole
shot -- prewinders, echo train, ADCs, labels, rewinders. Dimensionality is
inferred from the length of ``matrix``, so the same call builds the 2D or the
3D member of a family. Design once outside the loop, then re-index per shot::

    readout = make_line_readout(system, (0.22, 0.22), (128, 128))
    # One shot of this readout, played 128 times: the sequence claims every
    # library row from that up front.
    seq = Sequence(system, 128, readout.set_state(lin_idx=0))
    for ky in range(128):
        readout.set_state(lin_idx=ky)
        for block in readout:
            seq.add_block(*block)

Keyword arguments not listed below are forwarded verbatim to the underlying
readout class, whose docstring documents them in full.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from .bssfp import Bssfp2D, Bssfp3D
from .epi import Epi2D, Epi2DFlyback, Epi3D, Epi3DFlyback
from .fse import Fse2D, Fse3D
from .line import Line2D, Line3D
from .noncartesian import (
    Arbitrary,
    NonCartesian2D,
    NonCartesian3D,
    Projection,
    Radial,
    Rosette,
    Spiral,
    StackOfTrajectories,
)
from .zte import Zte


def _dimensions(matrix) -> int:
    if isinstance(matrix, int):
        return 1
    if not isinstance(matrix, Sequence):
        raise TypeError("matrix must be an integer or a sequence")
    return len(matrix)


def _reject_axis_keywords(kwargs) -> None:
    axes = sorted({"ro_axis", "pe_axis", "par_axis", "phase_axis"} & set(kwargs))
    if axes:
        joined = ", ".join(axes)
        raise TypeError(f"readout axes are fixed to x/y/z; remove: {joined}")


def _readout_width(fov_x: float, nx: int, oversamp: float, pf: float) -> float:
    n_full = round(oversamp * nx)
    n_post = n_full // 2
    n_samples = max(n_post + 1, round(pf * n_full))
    return n_samples / (oversamp * fov_x)


def _unpack_bssfp_excitation(excitation):
    """Return the caller-designed ``(rf, gz, gz_reph)`` bSSFP input triple."""
    try:
        rf = excitation.rf
        gradients = tuple(excitation.gradients)
        rephasers = tuple(excitation.rephasers)
    except AttributeError:
        try:
            rf, gradient, rephaser = excitation
        except (TypeError, ValueError) as error:
            try:
                rf = excitation.rf
            except AttributeError:
                raise TypeError(
                    "excitation must be an RfPulse or an (rf, gradient, rephaser) triple"
                ) from error
            gradients = ()
            rephasers = ()
        else:
            gradients = (gradient,)
            rephasers = (rephaser,)
    if not gradients and not rephasers:
        return rf, None, None
    if len(gradients) != 1 or len(rephasers) != 1:
        raise ValueError("bSSFP excitation must contain exactly one selection gradient and one rephaser")
    gradient, rephaser = gradients[0], rephasers[0]
    if getattr(gradient, "channel", None) != "z" or getattr(rephaser, "channel", None) != "z":
        raise ValueError("bSSFP excitation selection gradient and rephaser must both be on z")
    return rf, gradient, rephaser


def make_bssfp_readout(
    system,
    fov,
    matrix,
    excitation,
    *,
    segment: int = 1,
    bandwidth_hz_px: float = 100_000.0,
):
    """Create an optimized continuous-gradient bSSFP/TRUFI acquisition.

    This is a complete bSSFP encoding train, not a balanced isolated line.
    The caller owns RF design through ``excitation``; this factory reshapes
    its supplied selection gradient and rephaser into the Pulseq
    ``writeTrufi`` construction: an alpha/2 startup pulse, alternating 0/pi
    RF and receiver phase, and phase encodes whose readout and slice/slab
    gradients share ramps across consecutive TRs.
    Every emitted train starts with ``SET ONCE=1`` and ends with ``SET
    ONCE=2`` so an interpreter can identify the startup and ramp-down.

    The 2D variant contains the complete ``ky`` acquisition in one module.
    For 3D, ``segment`` divides the complete ``ky``/``kz`` grid into that many
    equal-length, structurally identical shots; append each with a different
    ``segment_idx``.  The 3D form deliberately does not make one monolithic
    block train that would exceed an EPIC segment's practical size.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits used both to design ``excitation`` and to build the
        bSSFP encoding; the factory does not mutate them.
    fov : tuple of float
        ``(fov_x_m, fov_y_m)`` for 2D or ``(fov_x_m, fov_y_m, fov_z_m)`` for
        3D, in metres.
    matrix : tuple of int
        ``(nx, ny)`` for 2D or ``(nx, ny, nz)`` for 3D.
    excitation : SequenceModule or tuple
        Caller-designed RF module.  A slice/slab-selective module, normally
        from :func:`make_slice_selective_pulse`, must contain exactly one z
        selection gradient and one z rephaser.  A nonselective module, such
        as :func:`make_hard_pulse`, is accepted for 3D only.  An explicit
        ``(rf, selection_gradient, rephaser)`` triple is also accepted.
    segment : int, optional
        Number of equal 3D shots (default 1).  It must divide ``ny*nz``.  2D
        accepts only the default value.
    bandwidth_hz_px : float, optional
        Receiver bandwidth per readout pixel (default 100 kHz).

    Returns
    -------
    Bssfp2D or Bssfp3D
        Stateful module.  ``tr_s`` and ``te_s`` are the steady-state timing;
        ``lines_per_segment`` and ``num_segments`` describe the 3D split.

    Raises
    ------
    ValueError
        If dimensions are invalid or a 3D segment is uneven.

    Examples
    --------
    Design the RF independently, then append a complete optimized 2D TRUFI
    train::

        excitation = design.make_slice_selective_pulse(
            np.deg2rad(35), 5e-3, duration=0.6e-3, system=system)
        bssfp = design.make_bssfp_readout(
            system, (0.22, 0.22), (128, 128), excitation)
        bssfp.set_state(phase_offset_rad=0.0)
        for block in bssfp:
            seq.add_block(*block)

    Partition a slab-selective 3D volume into four equal shots.  All four calls have the
    same block structure; only LIN/PAR labels, phase encodes, and the global
    alternating RF phase advance::

        excitation = design.make_slice_selective_pulse(
            np.deg2rad(25), 0.16, duration=0.6e-3, system=system)
        bssfp = design.make_bssfp_readout(
            system, (0.22, 0.22, 0.16), (128, 64, 16), excitation, segment=4)
        for segment_idx in range(bssfp.num_segments):
            bssfp.set_state(segment_idx=segment_idx)
            for block in bssfp:
                seq.add_block(*block)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       excitation = design.make_slice_selective_pulse(np.deg2rad(35), 5e-3, duration=0.6e-3, system=system)
       train = design.make_bssfp_readout(system, (0.22, 0.22), (64, 12), excitation)
       train.plot()
       plt.gcf().suptitle("optimized 2D bSSFP: alpha/2 prep, continuous TRs, ramp-down")

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       excitation = design.make_slice_selective_pulse(np.deg2rad(25), 0.12, duration=0.6e-3, system=system)
       train = design.make_bssfp_readout(system, (0.22, 0.22, 0.12), (48, 4, 4), excitation, segment=2)
       train.set_state(segment_idx=1).plot()
       plt.gcf().suptitle("optimized 3D bSSFP: one of two equal segments")

    A nonselective 3D variant has no slab gradient; it uses the same
    segmented encoding and makes no slice/slab selection claim::

        excitation = design.make_hard_pulse(np.deg2rad(25), duration=0.5e-3, system=system)
        bssfp = design.make_bssfp_readout(
            system, (0.22, 0.22, 0.16), (128, 64, 16), excitation, segment=4)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       excitation = design.make_hard_pulse(np.deg2rad(25), duration=0.5e-3, system=system)
       train = design.make_bssfp_readout(system, (0.22, 0.22, 0.12), (48, 4, 4), excitation, segment=2)
       train.set_state(segment_idx=1).plot()
       plt.gcf().suptitle("optimized nonselective 3D bSSFP: one of two equal segments")
    """
    dimensions = _dimensions(matrix)
    if dimensions not in (2, 3):
        raise ValueError("bSSFP readout matrix must have two or three dimensions")
    if len(fov) != dimensions:
        raise ValueError("fov and matrix must have the same dimensionality")
    rf, gradient, rephaser = _unpack_bssfp_excitation(excitation)
    if dimensions == 2:
        if gradient is None:
            raise ValueError("2D bSSFP requires a slice-selective excitation with z selection and rephasing gradients")
        if int(segment) != 1:
            raise ValueError("2D bSSFP is one complete train; segment is supported for 3D only")
        return Bssfp2D(
            system,
            fov,
            matrix,
            excitation_rf=rf,
            excitation_gz=gradient,
            excitation_rephaser=rephaser,
            bandwidth_hz_px=bandwidth_hz_px,
        )
    return Bssfp3D(
        system,
        fov,
        matrix,
        excitation_rf=rf,
        excitation_gz=gradient,
        excitation_rephaser=rephaser,
        segments=int(segment),
        bandwidth_hz_px=bandwidth_hz_px,
    )


def make_line_readout(system, fov, matrix, num_echoes: int = 1, **kwargs):
    """Create a Cartesian line readout: one phase-encode step per shot.

    The plain GRE/spin-warp readout, and the base every Cartesian sequence is
    timed against. ``num_echoes > 1`` extends it into a multi-echo train
    (bipolar by default, ``flyback=True`` for constant polarity), all at the
    same phase encode -- multi-echo GRE, Dixon, T2* mapping.

    ``spoil_position`` selects the steady state: ``"post"`` (default) for
    spoiled GRE / SSFP-FID, ``"pre"`` for SSFP-Echo/PSIF, ``"none"`` for a
    fully balanced isolated-line readout.  For an optimized continuous-gradient
    bSSFP/TRUFI train, use :func:`make_bssfp_readout` instead.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits (derated in place unless ``derate=False``).
    fov : tuple of float
        ``(fov_x_m, fov_y_m)`` for 2D, ``(fov_x_m, fov_y_m, fov_z_m)`` for 3D.
    matrix : tuple of int
        ``(nx, ny)`` or ``(nx, ny, nz)``; its length selects 2D or 3D.
    num_echoes : int, optional
        Readouts per shot, all at the same phase encode (default 1).
    **kwargs
        Forwarded to :class:`~pulserver.design._readout.line.Line2D` /
        ``Line3D``: ``bandwidth_hz_px``, ``oversamp``, ``pf``, ``flyback``,
        ``spoil_position``, ``spoil_cycles``, ``esp_s``, ``slice_rephasing``
        and ``derate``. Pass ``slice_rephasing=excitation.rephasers`` to
        absorb the excitation's rephasing moment into this readout's
        prewinder, and pair it with ``excitation.without_rephasers()``.

    Returns
    -------
    Line2D or Line3D
        Readout module; ``.duration``, ``.esp`` and ``.t_first_echo_s`` give
        the timing needed to solve TE and TR.

    Raises
    ------
    ValueError
        If ``matrix`` is neither two- nor three-dimensional.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_line_readout(system, (0.22, 0.22), (128, 128))
    >>> readout.n_samples
    128
    >>> round(readout.duration * 1e3, 2)
    2.51

    A three-echo bipolar train, and one shot appended per ``ky``::

        readout = design.make_line_readout(system, (0.22, 0.22), (128, 128), num_echoes=3)
        for ky in range(128):
            readout.set_state(lin_idx=ky, phase_offset_rad=phases[ky])
            for block in readout:
                seq.add_block(*block)

    Balanced, SSFP-FID, SSFP-echo, and both multiecho traversal conventions.
    The spoiled cases deliberately use 12 cycles so the bridged spoiler is
    visually unmistakable:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       design.make_line_readout(
           system, (0.22, 0.22), (128, 128), spoil_position="none").plot()
       plt.gcf().suptitle("balanced")

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       design.make_line_readout(
           system, (0.22, 0.22), (128, 128),
           spoil_position="post", spoil_cycles=12).plot()
       plt.gcf().suptitle("SSFP-FID: 12-cycle post-spoiler")

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       design.make_line_readout(
           system, (0.22, 0.22), (128, 128),
           spoil_position="pre", spoil_cycles=12).plot()
       plt.gcf().suptitle("SSFP-echo: 12-cycle pre-spoiler")

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       design.make_line_readout(
           system, (0.22, 0.22), (128, 128), num_echoes=3,
           spoil_position="none").plot()
       plt.gcf().suptitle("three echoes: bipolar")

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       design.make_line_readout(
           system, (0.22, 0.22), (128, 128), num_echoes=3,
           flyback=True, spoil_position="none").plot()
       plt.gcf().suptitle("three echoes: monopolar/flyback")

    See Also
    --------
    make_epi_readout : the same encoding acquired in one shot.
    make_phase_encoding : the standalone phase-encode gradient.
    """
    _reject_axis_keywords(kwargs)
    if "spoil_factor" in kwargs:
        raise TypeError("spoil_factor was replaced by spoil_cycles")
    spoil_cycles = float(kwargs.pop("spoil_cycles", 1.0))
    oversamp = float(kwargs.get("oversamp", 1.0))
    pf = float(kwargs.get("pf", 1.0))
    k_width = _readout_width(float(fov[0]), int(matrix[0]), oversamp, pf)
    kwargs["spoil_factor"] = spoil_cycles * int(matrix[0]) / float(fov[0]) / k_width
    dimensions = _dimensions(matrix)
    if dimensions == 2:
        return Line2D(system, fov, matrix, num_echoes, **kwargs)
    if dimensions == 3:
        return Line3D(system, fov, matrix, num_echoes, **kwargs)
    raise ValueError("line readout matrix must have two or three dimensions")


def make_epi_readout(system, fov, matrix, num_shots, sampling_mask, *, flyback: bool = False, **kwargs):
    """Create an EPI readout: a whole echo train after one excitation.

    Traverses many phase-encode lines per excitation, which is what makes
    single-shot and segmented EPI fast -- at the cost of T2* decay and
    off-resonance distortion along the train.

    ``sampling_mask`` is the sequence of within-shot phase-encode indices the
    train visits, in train order; its length *is* the echo train length. Take
    it from a sampling plan (:meth:`~pulserver.ScanLoop.from_relative_shifts`,
    :func:`make_skipped_caipi_order`) rather than constructing it by hand.

    ``flyback=False`` (default) alternates readout polarity with no gap, so
    every gradient lobe is sampled; ``flyback=True`` keeps one polarity and
    rewinds between lines, trading acquisition time for immunity to
    odd/even (Nyquist ghost) inconsistency.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits (derated in place unless ``derate=False``).
    fov : tuple of float
        ``(fov_x_m, fov_y_m[, fov_z_m])`` in metres.
    matrix : tuple of int
        ``(nx, ny)`` or ``(nx, ny, nz)``; its length selects 2D or 3D.
    num_shots : int
        Number of shots the full encoding is split across (bookkeeping only --
        the train length comes from ``sampling_mask``).
    sampling_mask : array_like of int
        Within-shot phase-encode index visited at each echo, in train order.
    flyback : bool, optional
        Constant-polarity flyback train instead of a bipolar one.
    **kwargs
        Forwarded to the ``Epi2D``/``Epi3D`` class: ``bandwidth_hz_px``,
        ``oversamp``, ``ramp_sample``, ``start_polarity_positive``,
        ``blip_duration_s``, ``caipi_axis``, ``caipi_blip_area``,
        ``slice_rephasing`` and ``derate``. Pass
        ``slice_rephasing=excitation.rephasers`` to absorb the excitation's
        rephasing moment into this readout's prewinder, and pair it with
        ``excitation.without_rephasers()``.

    Returns
    -------
    Epi2D, Epi2DFlyback, Epi3D or Epi3DFlyback
        Readout module; ``.etl``, ``.esp`` and ``.duration`` describe the
        train.

    Raises
    ------
    ValueError
        If ``matrix`` is neither two- nor three-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_epi_readout(system, (0.22, 0.22), (64, 64), 1, np.arange(64))
    >>> readout.etl
    64

    Drive a segmented blipped-CAIPI plan::

        order = _lowlevel.make_skipped_caipi_order((64, 16), acceleration=(2, 2), caipi_shift=1, segments=2)
        readout = design.make_epi_readout(system, fov, matrix, order.n_shots, order.relative(0)[:, 0])
        for shot in range(order.n_shots):
            readout.set_state(lin_idx=int(order[shot][0, 0]))
            for block in readout:
                seq.add_block(*block)

    Short 2D bipolar, 2D flyback, and 3D bipolar trains, so the in-plane
    blips, flyback rewind, and partition blips are legible:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       design.make_epi_readout(system, (0.22, 0.22), (32, 8), 1, np.arange(8)).plot()

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       design.make_epi_readout(system, (0.22, 0.22), (32, 8), 1, np.arange(8), flyback=True).plot()

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       mask3d = np.column_stack((np.arange(8), np.arange(8) % 4))
       design.make_epi_readout(system, (0.22, 0.22, 0.12), (32, 8, 4), 1, mask3d).plot()

    See Also
    --------
    make_skipped_caipi_order : build ``sampling_mask``.
    """
    _reject_axis_keywords(kwargs)
    dimensions = _dimensions(matrix)
    classes = {(2, False): Epi2D, (2, True): Epi2DFlyback, (3, False): Epi3D, (3, True): Epi3DFlyback}
    try:
        factory = classes[(dimensions, bool(flyback))]
    except KeyError as error:
        raise ValueError("EPI matrix must have two or three dimensions") from error
    return factory(system, fov, matrix, num_shots, sampling_mask, **kwargs)


def make_fse_readout(system, fov, matrix, echo_train_length, refocusing, **kwargs):
    """Create an FSE/TSE readout: a CPMG refocusing train, one line per echo.

    Refocusing pulses replace EPI's gradient reversals, so the train decays
    with T2 rather than T2* and is immune to off-resonance distortion. The
    price is RF power and a T2-weighted point spread function -- which is why
    the echo ordering (:func:`make_radial_order` and friends) matters as much
    as the train itself.

    ``refocusing`` is the refocusing pulse *module*, normally from
    :func:`make_refocusing_pulse`; an explicit ``(rf, gradient)`` pair is also
    accepted. The train rescales that one envelope per echo, so a variable
    flip schedule (:func:`make_traps_schedule`) costs nothing extra.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits (derated in place unless ``derate=False``).
    fov : tuple of float
        ``(fov_x_m, fov_y_m[, fov_z_m])`` in metres.
    matrix : tuple of int
        ``(nx, ny)`` or ``(nx, ny, nz)``; its length selects 2D or 3D.
    echo_train_length : int
        Refocusing pulses -- and encoded lines -- per shot.
    refocusing : SequenceModule or tuple
        Slice-selective refocusing module, or an ``(rf, gradient)`` pair.
    **kwargs
        Forwarded to ``Fse2D``/``Fse3D``: ``bandwidth_hz_px``, ``oversamp``,
        ``pf``, ``esp_s``, ``refoc_flip_scale``, ``refoc_phase_rad``,
        ``spoil_cycles`` and ``derate``.

    Returns
    -------
    Fse2D or Fse3D
        Readout module; ``.esp``, ``.t_first_echo`` and
        ``.t_exc_center_to_train_start`` place the train relative to the 90.

    Raises
    ------
    TypeError
        If ``refocusing`` is neither an RF module nor an ``(rf, gradient)``
        pair.
    ValueError
        If ``matrix`` is neither two- nor three-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> refocusing = design.make_refocusing_pulse(slice_thickness=5e-3, system=system)
    >>> readout = design.make_fse_readout(system, (0.22, 0.22), (128, 128), 16, refocusing)
    >>> bool(round(readout.esp * 1e3, 2) > 0)
    True

    Combine a variable flip schedule with a centre-out echo ordering and
    append one ordered train::

        from pulserver import ScanLoop

        mask = np.ones(128, dtype=bool)
        sampling = ScanLoop.from_mask(mask, train_length=16, ordering="radial")
        readout.set_state(lin_idx=sampling[0][:, 0])
        for block in readout:
            seq.add_block(*block)

    Four echoes of both 2D and nonselective 3D CPMG trains, with their
    crushers and phase encodes:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       refocusing = design.make_refocusing_pulse(slice_thickness=5e-3, system=system)
       design.make_fse_readout(system, (0.22, 0.22), (128, 128), 4, refocusing).plot()

    .. plot::
       :include-source: false

       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       nonselective = design.make_refocusing_pulse(system=system)
       design.make_fse_readout(system, (0.22, 0.22, 0.12), (128, 128, 32), 4,
                           nonselective).plot()

    See Also
    --------
    make_refocusing_pulse, make_traps_schedule, make_radial_order
    """
    _reject_axis_keywords(kwargs)
    obsolete = {"crusher_cycles", "x_spoil_factor", "z_spoil_factor"} & set(kwargs)
    if obsolete:
        raise TypeError("use spoil_cycles instead of " + ", ".join(sorted(obsolete)))
    try:
        rf = refocusing.rf
        gradient = refocusing.gradients[0] if refocusing.gradients else SimpleNamespace(amplitude=0.0)
    except AttributeError:
        try:
            rf, gradient = refocusing
        except (TypeError, ValueError) as error:
            raise TypeError("refocusing must be an RF module or an (rf, gradient) pair") from error
    dimensions = _dimensions(matrix)
    kwargs["crusher_cycles"] = float(kwargs.pop("spoil_cycles", 4.0))
    kwargs["spoil_voxel_size"] = (
        float(fov[2]) / int(matrix[2])
        if dimensions == 3
        else getattr(refocusing, "slice_thickness", float(fov[0]) / int(matrix[0]))
    )
    if dimensions == 2:
        return Fse2D(system, fov, matrix, echo_train_length, rf, gradient, **kwargs)
    if dimensions == 3:
        return Fse3D(system, fov, matrix, echo_train_length, rf, gradient, **kwargs)
    raise ValueError("FSE matrix must have two or three dimensions")


_TRAIN_KEYS = ("num_echoes", "rotation_label", "slice_rephasing")
_STACK_KEYS = (*_TRAIN_KEYS, "phase_axis", "phase_label")


def _split_kwargs(kwargs, train_keys=_TRAIN_KEYS):
    """Separate base-waveform design kwargs from acquisition-train kwargs."""
    _reject_axis_keywords(kwargs)
    train = {key: kwargs.pop(key) for key in list(kwargs) if key in train_keys}
    return kwargs, train


def make_noncartesian_2d_readout(system, trajectory, matrix, **kwargs):
    """Create a 2D readout from a k-space path you designed yourself.

    The escape hatch from the named families: give it the *shape* of any
    planar trajectory and it solves a slew- and amplitude-feasible gradient
    for it (:func:`traj2grad`), bridges the prewinder and rewinder, and wraps
    the result as one shot. Use it when the path is not a rotated copy of a
    radial, spiral or rosette base waveform.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    trajectory : array_like
        K-space path of shape ``(n, 2)`` in cycles/m. Only its geometry is
        used, not its sample spacing.
    matrix : int
        Nominal image matrix, which sets the ADC sample count.
    **kwargs
        Base-waveform keywords (``bandwidth_hz_px``, ``oversamp``, ``axes``,
        ``solver_oversampling``, ``derate``) and train keywords
        (``num_echoes``, ``rotation_label``, ``slice_rephasing``).

    Returns
    -------
    NonCartesian2D
        Readout module wrapping the designed waveform.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> theta = np.linspace(0, 6 * np.pi, 512)
    >>> path = np.stack([np.linspace(0, 250, 512) * np.cos(theta),
    ...                  np.linspace(0, 250, 512) * np.sin(theta)], axis=-1)
    >>> readout = design.make_noncartesian_2d_readout(system, path, 128)
    >>> bool(readout.duration > 0)
    True

    Append one shot explicitly::

        seq = pp.Sequence(readout.system)
        readout.set_state(lin_idx=0)
        for block in readout:
            seq.add_block(*block)

    An analytic path, and the gradient waveform designed for it:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       theta = np.linspace(0, 6 * np.pi, 512)
       radius = np.linspace(0, 250.0, 512)
       path = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)
       from _figures import noncartesian_readout_figure
       noncartesian_readout_figure(
           design.make_noncartesian_2d_readout(system, path, 128),
           title="arbitrary 2D")

    See Also
    --------
    make_noncartesian_3d_readout : the volumetric counterpart.
    traj2grad : the underlying path-to-gradient solver.
    """
    _reject_axis_keywords(kwargs)
    design, train = _split_kwargs(kwargs)
    return NonCartesian2D(Arbitrary(system, trajectory, matrix=matrix, **design), **train)


def make_noncartesian_3d_readout(system, trajectory, matrix, **kwargs):
    """Create a 3D readout from a k-space path you designed yourself.

    The volumetric counterpart of :func:`make_noncartesian_2d_readout`: the
    supplied path already visits all three axes, so the shot is *not* a
    rotated copy of a planar base waveform. Per-shot orientation, if any, is
    still applied as a rotation.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    trajectory : array_like
        K-space path of shape ``(n, 3)`` in cycles/m.
    matrix : int
        Nominal image matrix, which sets the ADC sample count.
    **kwargs
        Base-waveform and train keywords, as for
        :func:`make_noncartesian_2d_readout`.

    Returns
    -------
    NonCartesian3D
        Readout module wrapping the designed waveform.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> t = np.linspace(0, 1, 512)
    >>> path = np.stack([200 * t * np.cos(20 * t), 200 * t * np.sin(20 * t), 200 * t], axis=-1)
    >>> readout = design.make_noncartesian_3d_readout(system, path, 64)
    >>> bool(readout.duration > 0)
    True

    Append one shot explicitly::

        seq = pp.Sequence(readout.system)
        readout.set_state(lin_idx=0)
        for block in readout:
            seq.add_block(*block)

    A conical path that visits all three axes within one shot:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       t = np.linspace(0, 1, 512)
       path = np.stack([200 * t * np.cos(20 * t), 200 * t * np.sin(20 * t), 200 * t], axis=-1)
       from _figures import noncartesian_readout_figure
       noncartesian_readout_figure(
           design.make_noncartesian_3d_readout(system, path, 64),
           title="arbitrary 3D (canonical kx'/ky' component)")

    See Also
    --------
    make_noncartesian_2d_readout : the planar counterpart.
    """
    _reject_axis_keywords(kwargs)
    design, train = _split_kwargs(kwargs)
    return NonCartesian3D(Arbitrary(system, trajectory, matrix=matrix, **design), **train)


def make_radial_readout(system, fov, matrix, **kwargs):
    """Create a 2D radial readout: one centre-crossing spoke per shot.

    Every spoke passes through the k-space centre, so motion and
    undersampling artefacts spread as streaks instead of coherent ghosts, and
    the centre is oversampled for free — the reason radial is preferred for
    free-breathing and self-navigated work.

    One base spoke is designed here; the per-shot angle is applied as a
    rotation, so pair it with :func:`make_radial_tilt` or
    :func:`calc_golden_angles`.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        Encoded field of view (m).
    matrix : int
        Nominal image matrix.
    **kwargs
        Base-waveform keywords (``bandwidth_hz_px``, ``oversamp``,
        ``derate``) and train keywords (``num_echoes``,
        ``rotation_label``, ``slice_rephasing``).

    Returns
    -------
    NonCartesian2D
        Readout module wrapping the base spoke.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_radial_readout(system, 0.22, 128)
    >>> bool(readout.duration > 0)
    True

    Play a golden-angle acquisition with explicit block iteration::

        pattern = _lowlevel.make_radial_tilt(1000, scheme="golden")
        seq = pp.Sequence(readout.system)
        for view, rotation in enumerate(pattern.to_rotations()):
            readout.set_state(lin_idx=view, rotation=rotation)
            for block in readout:
                seq.add_block(*block)

    The module's blocks, as they are played:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import noncartesian_readout_figure
       noncartesian_readout_figure(design.make_radial_readout(system, 0.22, 128), title="radial")

    See Also
    --------
    make_radial_projection_readout : the same spoke steered over the sphere.
    make_radial_stack_readout : the same spoke with Cartesian kz encoding.
    make_zte_readout : the zero-echo-time radial variant.
    """
    _reject_axis_keywords(kwargs)
    design, train = _split_kwargs(kwargs)
    return NonCartesian2D(Radial(system, fov, matrix, **design), **train)


def make_radial_projection_readout(system, fov, matrix, **kwargs):
    """Create a 3D radial (kooshball) readout: one spoke per direction.

    The same centre-crossing spoke as :func:`make_radial_readout`, but steered
    over the whole sphere rather than one plane, so every shot is a
    projection through the volume. Isotropic resolution comes from the
    direction set alone — pair it with
    :func:`make_golden_means_3d_tilt` or
    :func:`make_spiral_phyllotaxis_tilt` and
    :meth:`~pulserver.ScanLoop.to_rotations`.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        Isotropic encoded field of view (m).
    matrix : int
        Nominal image matrix.
    **kwargs
        As for :func:`make_radial_readout`.

    Returns
    -------
    Projection
        Readout module wrapping the base spoke.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_radial_projection_readout(system, 0.22, 128)
    >>> bool(readout.duration > 0)
    True

    Cover the sphere with golden-means directions, appending each shot::

        pattern = _lowlevel.make_golden_means_3d_tilt(2000)
        seq = pp.Sequence(readout.system)
        for view, rotation in enumerate(pattern.to_rotations()):
            readout.set_state(lin_idx=view, rotation=rotation)
            for block in readout:
                seq.add_block(*block)

    The module's blocks, as they are played:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import noncartesian_readout_figure
       noncartesian_readout_figure(
           design.make_radial_projection_readout(system, 0.22, 128),
           title="radial projection (canonical kx'/ky')")

    See Also
    --------
    make_radial_readout, make_radial_stack_readout
    """
    design, train = _split_kwargs(kwargs)
    return Projection(Radial(system, fov, matrix, **design), **train)


def make_radial_stack_readout(system, fov, matrix, fov_z, nz, **kwargs):
    """Create a stack-of-stars readout: rotated spokes, Cartesian kz.

    Radial in-plane, Cartesian through-plane. The hybrid keeps radial's
    motion robustness and centre oversampling where they matter — in the
    imaging plane — while leaving kz conventionally encoded, so partial
    Fourier, parallel imaging and slab selection all behave as in a Cartesian
    3D acquisition.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        In-plane encoded field of view (m).
    matrix : int
        In-plane nominal image matrix.
    fov_z : float
        Through-plane field of view (m).
    nz : int
        Number of partitions.
    **kwargs
        As for :func:`make_radial_readout`, plus ``phase_axis`` and
        ``phase_label`` for the partition-encoding gradient.

    Returns
    -------
    StackOfTrajectories
        Readout module; ``set_state(lin_idx=..., par_idx=...)`` selects the
        spoke and the partition.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_radial_stack_readout(system, 0.22, 128, 0.12, 32)
    >>> readout.nz
    32

    Append one rotated partition::

        seq = pp.Sequence(readout.system)
        readout.set_state(lin_idx=0, par_idx=16, rotation=rotation)
        for block in readout:
            seq.add_block(*block)

    The extra partition-encoding lobe on z is what distinguishes this from the planar readout:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import noncartesian_readout_figure
       noncartesian_readout_figure(
           design.make_radial_stack_readout(system, 0.22, 128, 0.12, 32),
           title="stack of stars (canonical kx'/ky')")

    See Also
    --------
    make_radial_readout, make_radial_projection_readout
    """
    design, train = _split_kwargs(kwargs, _STACK_KEYS)
    return StackOfTrajectories(Radial(system, fov, matrix, **design), fov_z, nz, **train)


def make_rosette_readout(system, fov, matrix, **kwargs):
    """Create a 2D rosette readout: one petal set per shot.

    A rosette petal repeatedly crosses the k-space centre during a single
    readout, so the centre is sampled at many closely spaced times. That makes
    the trajectory self-navigating and lets a reconstruction resolve spectral
    or T2* information from one shot — the reason it is used for
    spectroscopic and quantitative imaging.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        Encoded field of view (m).
    matrix : int
        Nominal image matrix.
    **kwargs
        Base-waveform keywords (``petals``, ``angular_frequency_ratio``,
        ``echo_spacing_s``, ``bandwidth_hz_px``, ``oversamp``, ``axes``,
        ``derate``) and train keywords (``num_echoes``, ``rotation_label``,
        ``slice_rephasing``).

    Returns
    -------
    NonCartesian2D
        Readout module wrapping the base petal set.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_rosette_readout(system, 0.22, 128)
    >>> bool(readout.duration > 0)
    True

    Append one rotated petal set::

        seq = pp.Sequence(readout.system)
        readout.set_state(lin_idx=0, rotation=rotation)
        for block in readout:
            seq.add_block(*block)

    The module's blocks, as they are played:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import noncartesian_readout_figure
       noncartesian_readout_figure(design.make_rosette_readout(system, 0.22, 128), title="rosette")

    See Also
    --------
    make_rosette_projection_readout, make_rosette_stack_readout
    make_spiral_readout : single-pass alternative with faster coverage.
    """
    design, train = _split_kwargs(kwargs)
    return NonCartesian2D(Rosette(system, fov, matrix, **design), **train)


def make_rosette_projection_readout(system, fov, matrix, **kwargs):
    """Create a 3D rosette readout: one petal set per sphere direction.

    The planar petal set of :func:`make_rosette_readout` steered over the
    sphere, so each shot both crosses the k-space centre many times *and*
    contributes a different projection — dense centre sampling for
    self-navigation without spending shots on it.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        Isotropic encoded field of view (m).
    matrix : int
        Nominal image matrix.
    **kwargs
        As for :func:`make_rosette_readout`.

    Returns
    -------
    Projection
        Readout module wrapping the base petal set.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_rosette_projection_readout(system, 0.22, 128)
    >>> bool(readout.duration > 0)
    True

    Append one projection::

        seq = pp.Sequence(readout.system)
        readout.set_state(lin_idx=0, rotation=rotation)
        for block in readout:
            seq.add_block(*block)

    The module's blocks, as they are played:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import noncartesian_readout_figure
       noncartesian_readout_figure(
           design.make_rosette_projection_readout(system, 0.22, 128),
           title="rosette projection (canonical kx'/ky')")

    See Also
    --------
    make_rosette_readout, make_rosette_stack_readout
    """
    design, train = _split_kwargs(kwargs)
    return Projection(Rosette(system, fov, matrix, **design), **train)


def make_rosette_stack_readout(system, fov, matrix, fov_z, nz, **kwargs):
    """Create a stack-of-rosettes readout: rotated petals, Cartesian kz.

    In-plane rosette encoding with conventional partition encoding along kz —
    the volumetric form used when the spectral/T2* information a rosette
    provides is wanted per slice rather than per projection.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        In-plane encoded field of view (m).
    matrix : int
        In-plane nominal image matrix.
    fov_z : float
        Through-plane field of view (m).
    nz : int
        Number of partitions.
    **kwargs
        As for :func:`make_rosette_readout`, plus ``phase_axis`` and
        ``phase_label``.

    Returns
    -------
    StackOfTrajectories
        Readout module.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_rosette_stack_readout(system, 0.22, 128, 0.12, 16)
    >>> readout.nz
    16

    Append one rotated partition::

        seq = pp.Sequence(readout.system)
        readout.set_state(lin_idx=0, par_idx=8, rotation=rotation)
        for block in readout:
            seq.add_block(*block)

    The module's blocks, as they are played:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import noncartesian_readout_figure
       noncartesian_readout_figure(
           design.make_rosette_stack_readout(system, 0.22, 128, 0.12, 16),
           title="stack of rosettes (canonical kx'/ky')")

    See Also
    --------
    make_rosette_readout, make_rosette_projection_readout
    """
    design, train = _split_kwargs(kwargs, _STACK_KEYS)
    return StackOfTrajectories(Rosette(system, fov, matrix, **design), fov_z, nz, **train)


def make_spiral_readout(system, fov, matrix, design_interleaves, **kwargs):
    """Create a 2D spiral readout: one interleaf per shot.

    A spiral covers a disc of k-space in a single, gradient-efficient sweep,
    so far fewer shots are needed than for Cartesian or radial sampling at the
    same resolution — at the cost of sensitivity to off-resonance blurring
    over the long readout.

    ``design_interleaves`` sets how many interleaves the full disc is split
    into and therefore the length of one readout. The base interleaf is
    designed at the system's slew and gradient limits; per-shot angles are
    applied as rotations.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        Encoded field of view (m).
    matrix : int
        Nominal image matrix.
    design_interleaves : int
        Number of interleaves the disc is split into.
    **kwargs
        Base-waveform keywords (``density``, ``variable_density_power``,
        ``transition_radius``, ``variant``, ``bandwidth_hz_px``, ``oversamp``,
        ``axes``, ``derate``) and train keywords (``num_echoes``,
        ``rotation_label``, ``slice_rephasing``).

    Returns
    -------
    NonCartesian2D
        Readout module wrapping the base interleaf.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_spiral_readout(system, 0.22, 128, 16)
    >>> bool(readout.duration > 0)
    True

    Append one rotated interleaf::

        seq = pp.Sequence(readout.system)
        readout.set_state(lin_idx=0, rotation=rotation)
        for block in readout:
            seq.add_block(*block)

    Constant-, variable-, and dual-density outward paths:

    .. plot::
       :include-source: false

       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       from _figures import trajectory_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       trajectory_figure([
           ("constant density", design.make_spiral_readout(
               system, 0.22, 128, 16, density="constant")),
           ("variable density", design.make_spiral_readout(
               system, 0.22, 128, 16, density="variable",
               inner_design_interleaves=8, outer_design_interleaves=24)),
           ("dual density", design.make_spiral_readout(
               system, 0.22, 128, 16, density="dual",
               inner_design_interleaves=8, outer_design_interleaves=24,
               transition_radius=0.45)),
       ], title="outward spiral density")

    Outward, inward, and in-out variable-density traversal:

    .. plot::
       :include-source: false

       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       from _figures import trajectory_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       common = dict(density="variable", inner_design_interleaves=8,
                     outer_design_interleaves=24)
       trajectory_figure([
           (variant, design.make_spiral_readout(
               system, 0.22, 128, 16, variant=variant, **common))
           for variant in ("outward", "inward", "in_out")
       ], title="variable-density traversal")

    The outward VDS waveform and a three-echo outward VDS train:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       common = dict(density="variable", inner_design_interleaves=8,
                     outer_design_interleaves=24)
       for title, echoes in (("outward VDS", 1), ("outward VDS, 3 echoes", 3)):
           design.make_spiral_readout(
               system, 0.22, 128, 16, num_echoes=echoes, **common).plot()
           plt.gcf().suptitle(title)

    See Also
    --------
    make_spiral_projection_readout, make_spiral_stack_readout
    make_rosette_readout : centre-crossing alternative for spectral encoding.
    """
    design, train = _split_kwargs(kwargs)
    return NonCartesian2D(Spiral(system, fov, matrix, design_interleaves, **design), **train)


def make_spiral_projection_readout(system, fov, matrix, design_interleaves, **kwargs):
    """Create a 3D spiral-projection readout: one interleaf per direction.

    The planar spiral interleaf of :func:`make_spiral_readout` steered over
    the sphere, so a whole disc of k-space is covered per shot and the shot
    count for an isotropic volume drops by roughly the interleaf's coverage
    factor compared with radial projections.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        Isotropic encoded field of view (m).
    matrix : int
        Nominal image matrix.
    design_interleaves : int
        Number of interleaves the disc is split into.
    **kwargs
        As for :func:`make_spiral_readout`.

    Returns
    -------
    Projection
        Readout module wrapping the base interleaf.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_spiral_projection_readout(system, 0.22, 128, 16)
    >>> bool(readout.duration > 0)
    True

    Append one projection::

        seq = pp.Sequence(readout.system)
        readout.set_state(lin_idx=0, rotation=rotation)
        for block in readout:
            seq.add_block(*block)

    The module's blocks, as they are played:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import noncartesian_readout_figure
       noncartesian_readout_figure(
           design.make_spiral_projection_readout(system, 0.22, 128, 16),
           title="spiral projection (canonical kx'/ky')")

    See Also
    --------
    make_spiral_readout, make_spiral_stack_readout
    """
    design, train = _split_kwargs(kwargs)
    return Projection(Spiral(system, fov, matrix, design_interleaves, **design), **train)


def make_spiral_stack_readout(system, fov, matrix, design_interleaves, fov_z, nz, **kwargs):
    """Create a stack-of-spirals readout: rotated interleaves, Cartesian kz.

    The workhorse volumetric spiral: efficient in-plane coverage with
    conventional partition encoding, so slab selection, partial Fourier and
    parallel imaging along kz behave exactly as in a Cartesian 3D acquisition.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        In-plane encoded field of view (m).
    matrix : int
        In-plane nominal image matrix.
    design_interleaves : int
        Number of interleaves the disc is split into.
    fov_z : float
        Through-plane field of view (m).
    nz : int
        Number of partitions.
    **kwargs
        As for :func:`make_spiral_readout`, plus ``phase_axis`` and
        ``phase_label``.

    Returns
    -------
    StackOfTrajectories
        Readout module; ``set_state(lin_idx=..., par_idx=...)`` selects the
        interleaf and the partition.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_spiral_stack_readout(system, 0.22, 128, 16, 0.12, 32)
    >>> readout.nz
    32

    Append one rotated partition::

        seq = pp.Sequence(readout.system)
        readout.set_state(lin_idx=0, par_idx=16, rotation=rotation)
        for block in readout:
            seq.add_block(*block)

    The module's blocks, as they are played:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import noncartesian_readout_figure
       noncartesian_readout_figure(
           design.make_spiral_stack_readout(system, 0.22, 128, 16, 0.12, 32),
           title="stack of spirals (canonical kx'/ky')")

    See Also
    --------
    make_spiral_readout, make_spiral_projection_readout
    """
    design, train = _split_kwargs(kwargs, _STACK_KEYS)
    return StackOfTrajectories(Spiral(system, fov, matrix, design_interleaves, **design), fov_z, nz, **train)


def make_zte_readout(system, fov, matrix, view_order, excitation, **kwargs):
    """Create a ZTE readout: continuous gradient, excite-and-acquire per view.

    In zero-echo-time imaging the encoding gradient is already on when the
    hard pulse is played, so acquisition starts within microseconds of
    excitation -- short enough to capture tissues that have vanished by the
    time a conventional TE elapses. Between views the gradient is *stepped*,
    never ramped down, which is also what makes ZTE acoustically quiet.

    Because the gradient never returns to zero, the whole ordered view set is
    one module: ``view_order`` is the segment, and consecutive directions must
    be close enough for the slew limit to bridge them. Orderings such as
    :func:`make_spiral_phyllotaxis_tilt` are chosen precisely for that.

    ``excitation`` is a non-selective pulse module (or bare RF event) -- a
    selective pulse is meaningless here, since the gradient is already
    encoding.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits (derated in place unless ``derate=False``).
    fov : float or sequence of float
        Isotropic encoded field of view (m).
    matrix : int or sequence of int
        Isotropic nominal image matrix.
    view_order : array_like
        Ordered in-plane angles (rad), or ``(N, 2)`` / ``(N, 3)`` spoke
        directions; normalised internally.
    excitation : SequenceModule or SimpleNamespace
        Non-selective RF module, or the RF event itself.
    **kwargs
        Forwarded to ``Zte``: ``bandwidth_hz_px``, ``oversamp``,
        ``dead_time_s``, ``tr_s``, ``derate``, ``rotation_encoded``.

        ``rotation_encoded=True`` is what makes a long segment playable: it
        moves the view-to-view direction change out of the waveform and into a
        per-block rotation, so the segment costs two gradient shapes instead of
        one per view. It requires a constant-rotation view order -- see
        :func:`pulserver.design.make_rotated_projection_sampling`, whose
        ``'disk'`` scheme produces one.

    Returns
    -------
    Zte
        Readout module covering the whole ordered view set;
        ``.num_missing_samples`` reports the unacquired centre points the
        reconstruction must recover.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> excitation = design.make_hard_pulse(np.deg2rad(3), duration=8e-6, system=system)
    >>> readout = design.make_zte_readout(system, 0.22, 128, _lowlevel.calc_uniform_angles(64), excitation)
    >>> readout.num_views
    64

    Append the complete continuous-gradient view segment::

        seq = pp.Sequence(readout.system)
        readout.set_state(lin_idx=np.arange(readout.num_views))
        for block in readout:
            seq.add_block(*block)

    The gradient is stepped, never ramped to zero, between views:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       from _figures import zte_readout_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       excitation = design.make_hard_pulse(np.deg2rad(3), duration=8e-6, system=system)
       zte_readout_figure(
           design.make_zte_readout(system, 0.22, 64, _lowlevel.calc_uniform_angles(8), excitation))

    See Also
    --------
    make_radial_readout : the conventional-TE radial counterpart.
    make_spiral_phyllotaxis_tilt : slew-friendly 3D view ordering.
    """
    rf = getattr(excitation, "rf", excitation)
    return Zte(system, fov, matrix, view_order, rf, **kwargs)
