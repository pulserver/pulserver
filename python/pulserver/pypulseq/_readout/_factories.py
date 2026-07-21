"""Factories for public readout modules.

Each factory returns a stateful :class:`~pulserver.Module` holding one whole
shot -- prewinders, echo train, ADCs, labels, rewinders. Dimensionality is
inferred from the length of ``matrix``, so the same call builds the 2D or the
3D member of a family. Design once outside the loop, then re-index per shot::

    readout = make_line_readout(system, (0.22, 0.22), (128, 128))
    for ky in range(128):
        readout(seq, pe_idx=ky)

Keyword arguments not listed below are forwarded verbatim to the underlying
readout class, whose docstring documents them in full.
"""

from __future__ import annotations

from collections.abc import Sequence

from .epi import Epi2D, Epi2DFlyback, Epi3D, Epi3DFlyback
from .fse import Fse2D, Fse3D
from .line import Line2D, Line3D
from .noncartesian import NonCartesian2D, NonCartesian3D, Radial, Rosette, Spiral
from .zte import Zte


def _dimensions(matrix) -> int:
    if isinstance(matrix, int):
        return 1
    if not isinstance(matrix, Sequence):
        raise TypeError("matrix must be an integer or a sequence")
    return len(matrix)


def make_line_readout(system, fov, matrix, num_echoes: int = 1, **kwargs):
    """Create a Cartesian line readout: one phase-encode step per shot.

    The plain GRE/spin-warp readout, and the base every Cartesian sequence is
    timed against. ``num_echoes > 1`` extends it into a multi-echo train
    (bipolar by default, ``flyback=True`` for constant polarity), all at the
    same phase encode -- multi-echo GRE, Dixon, T2* mapping.

    ``spoil_position`` selects the steady state: ``"post"`` (default) for
    spoiled GRE / SSFP-FID, ``"pre"`` for SSFP-Echo/PSIF, ``"none"`` for a
    fully balanced (bSSFP) readout.

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
        Forwarded to :class:`~pulserver.pypulseq._readout.line.Line2D` /
        ``Line3D``: ``bandwidth_hz_px``, ``oversamp``, ``pf``, ``flyback``,
        ``spoil_position``, ``spoil_factor``, ``esp_s``, ``ro_axis``,
        ``pe_axis``, ``par_axis``, ``derate``.

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
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = pp.make_line_readout(system, (0.22, 0.22), (128, 128))
    >>> readout.n_samples
    128
    >>> round(readout.duration * 1e3, 2)
    2.51

    A three-echo bipolar train, and one shot appended per ``ky``::

        readout = pp.make_line_readout(system, (0.22, 0.22), (128, 128), num_echoes=3)
        for ky in range(128):
            readout(seq, pe_idx=ky, rf_phase_rad=phases[ky])

    See Also
    --------
    make_epi_readout : the same encoding acquired in one shot.
    make_phase_encoding : the standalone phase-encode gradient.
    """
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
    it from a sampling plan (:func:`from_relative_shifts`,
    :func:`skipped_caipi`) rather than constructing it by hand.

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
        ``ro_axis``, ``pe_axis``, ``derate``.

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
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = pp.make_epi_readout(system, (0.22, 0.22), (64, 64), 1, np.arange(64))
    >>> readout.etl
    64

    Drive a segmented blipped-CAIPI plan::

        plan = pp.skipped_caipi((64, 16), acceleration=(2, 2), caipi_shift=1, segments=2)
        readout = pp.make_epi_readout(system, fov, matrix, plan.n_shots, plan.relative(0)[:, 0])
        for shot in range(plan.n_shots):
            readout(seq, pe_start=int(plan[shot][0, 0]))

    See Also
    --------
    from_relative_shifts, skipped_caipi : build ``sampling_mask``.
    """
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
    the echo ordering (:func:`fse_radial_order` and friends) matters as much
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
    refocusing : Module or tuple
        Slice-selective refocusing module, or an ``(rf, gradient)`` pair.
    **kwargs
        Forwarded to ``Fse2D``/``Fse3D``: ``bandwidth_hz_px``, ``oversamp``,
        ``pf``, ``esp_s``, ``refoc_flip_scale``, ``refoc_phase_rad``,
        ``crusher_cycles``, ``ro_axis``, ``pe_axis``, ``derate``.

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
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> refocusing = pp.make_refocusing_pulse(slice_thickness=5e-3, system=system)
    >>> readout = pp.make_fse_readout(system, (0.22, 0.22), (128, 128), 16, refocusing)
    >>> bool(round(readout.esp * 1e3, 2) > 0)
    True

    Combine a variable flip schedule with a centre-out echo ordering::

        flips = pp.make_traps_schedule(16, np.deg2rad(120))
        shots = pp.fse_radial_order(coords, 16)

    See Also
    --------
    make_refocusing_pulse, make_traps_schedule, fse_radial_order
    """
    try:
        rf = refocusing.rf
        gradient = refocusing.gradients[0]
    except (AttributeError, IndexError):
        try:
            rf, gradient = refocusing
        except (TypeError, ValueError) as error:
            raise TypeError("refocusing must be an RF module or an (rf, gradient) pair") from error
    dimensions = _dimensions(matrix)
    if dimensions == 2:
        return Fse2D(system, fov, matrix, echo_train_length, rf, gradient, **kwargs)
    if dimensions == 3:
        return Fse3D(system, fov, matrix, echo_train_length, rf, gradient, **kwargs)
    raise ValueError("FSE matrix must have two or three dimensions")


def make_radial_readout(system, fov, matrix, **kwargs):
    """Create a 2D radial readout: one centre-crossing spoke per shot.

    Every spoke passes through the k-space centre, so motion and
    undersampling artefacts spread as streaks instead of coherent ghosts, and
    the centre is oversampled for free -- the reason radial is preferred for
    free-breathing and self-navigated work.

    One base spoke is designed here; the per-shot angle is applied as a
    rotation, so pair it with :func:`radial_2d` or :func:`golden_angles`.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        Encoded field of view (m).
    matrix : int
        Nominal image matrix.
    **kwargs
        Forwarded to the underlying ``Radial`` gradient and
        ``NonCartesian2D`` train (``bandwidth_hz_px``, ``oversamp``,
        ``num_echoes``, ``rotation_label``, ``derate``, ...).

    Returns
    -------
    NonCartesian2D
        Readout module wrapping the base spoke.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = pp.make_radial_readout(system, 0.22, 128)
    >>> bool(readout.duration > 0)
    True

    Play a golden-angle acquisition::

        pattern = pp.radial_2d(1000, scheme="golden")
        for angle in pattern.support[:, 0]:
            readout(seq, rotation=Rotation.from_euler("z", angle))

    See Also
    --------
    radial_2d, golden_angles : spoke angle plans.
    make_zte_readout : the zero-echo-time radial variant.
    """
    return NonCartesian2D(Radial(system, fov, matrix, **kwargs))


def make_spiral_readout(system, fov, matrix, design_interleaves, **kwargs):
    """Create a 2D spiral readout: one interleaf per shot.

    A spiral covers a disc of k-space in a single, gradient-efficient sweep,
    so far fewer shots are needed than for Cartesian or radial sampling at the
    same resolution -- at the cost of sensitivity to off-resonance blurring
    over the long readout.

    ``design_interleaves`` sets how many interleaves the full disc is split
    into and therefore the length of one readout. The base interleaf is
    designed by :func:`pulserver.pypulseq.arbgrad.spiral` at the system's slew
    and gradient limits; per-shot angles are applied as rotations.

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
        Forwarded to the underlying ``Spiral`` gradient and
        ``NonCartesian2D`` train.

    Returns
    -------
    NonCartesian2D
        Readout module wrapping the base interleaf.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = pp.make_spiral_readout(system, 0.22, 128, 16)
    >>> bool(readout.duration > 0)
    True

    See Also
    --------
    pulserver.pypulseq.arbgrad : the underlying waveform design.
    make_rosette_readout : centre-crossing alternative for spectral encoding.
    """
    return NonCartesian2D(Spiral(system, fov, matrix, design_interleaves, **kwargs))


def make_rosette_readout(system, fov, matrix, **kwargs):
    """Create a 2D rosette readout: one petal per shot.

    A rosette petal repeatedly crosses the k-space centre during a single
    readout, so the centre is sampled at many closely spaced times. That makes
    the trajectory self-navigating and lets a reconstruction resolve spectral
    or T2* information from one shot -- the reason it is used for
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
        Forwarded to the underlying ``Rosette`` gradient and
        ``NonCartesian2D`` train.

    Returns
    -------
    NonCartesian2D
        Readout module wrapping the base petal.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = pp.make_rosette_readout(system, 0.22, 128)
    >>> bool(readout.duration > 0)
    True

    See Also
    --------
    make_spiral_readout : single-pass alternative with faster coverage.
    """
    return NonCartesian2D(Rosette(system, fov, matrix, **kwargs))


def make_noncartesian_3d_readout(readout, **kwargs):
    """Promote a natively 3D non-Cartesian gradient into a readout module.

    The 2D factories bundle waveform design and train construction. For
    genuinely 3D trajectories -- kooshball radial, stack-of-spiral, a
    caller-designed path -- the waveform is built separately (typically via
    :mod:`pulserver.pypulseq.arbgrad`) and wrapped here, so the design stays
    under the caller's control.

    Per-shot orientation is applied as a rotation; get the matrices from
    :func:`directions_to_rotations`.

    Parameters
    ----------
    readout : NonCartesianGradient
        Base 3D gradient waveform object.
    **kwargs
        Forwarded to ``NonCartesian3D`` (``num_echoes``, ``rotation_label``).

    Returns
    -------
    NonCartesian3D
        Readout module wrapping the base waveform.

    Examples
    --------
    Rotate a 3D base waveform over a golden-means direction set::

        readout = pp.make_noncartesian_3d_readout(base_gradient)
        pattern = pp.golden_means_3d(2000)
        for rotation in pp.directions_to_rotations(pattern.support):
            readout(seq, rotation=rotation)

    See Also
    --------
    golden_means_3d, spiral_phyllotaxis, directions_to_rotations
    """
    return NonCartesian3D(readout, **kwargs)


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
    :func:`spiral_phyllotaxis` are chosen precisely for that.

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
    excitation : Module or SimpleNamespace
        Non-selective RF module, or the RF event itself.
    **kwargs
        Forwarded to ``Zte``: ``bandwidth_hz_px``, ``oversamp``,
        ``dead_time_s``, ``tr_s``, ``derate``.

    Returns
    -------
    Zte
        Readout module covering the whole ordered view set;
        ``.num_missing_samples`` reports the unacquired centre points the
        reconstruction must recover.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> excitation = pp.make_hard_pulse(np.deg2rad(3), duration=8e-6, system=system)
    >>> readout = pp.make_zte_readout(system, 0.22, 128, pp.uniform_angles(64), excitation)
    >>> readout.num_views
    64

    See Also
    --------
    make_radial_readout : the conventional-TE radial counterpart.
    spiral_phyllotaxis : slew-friendly 3D view ordering.
    """
    rf = getattr(excitation, "rf", excitation)
    return Zte(system, fov, matrix, view_order, rf, **kwargs)
