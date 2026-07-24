"""SLR, hard, frequency-selective, and slice-selective RF pulse factories."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pypulseq as pp

from ...pypulseq._opts import default_system
from ._base import RfPulse
from ._slr import design_slr

PulseType = Literal["st", "ex", "se", "inv", "sat"]
FilterType = Literal["ls", "pm", "min", "max", "ms"]

DEFAULT_DURATION = 4e-3
DEFAULT_TIME_BANDWIDTH_PRODUCT = 4.0
DEFAULT_SLICE_DURATION = 2e-3
RF_SPOILING_INCREMENT_DEG = 117.0


def _system_or_default(system: pp.Opts | None) -> pp.Opts:
    return default_system(system)


def _slr_sample_count(duration: float, dwell: float) -> int:
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if dwell <= 0:
        raise ValueError("dwell must be > 0")
    count = max(8, round(duration / dwell))
    return count + count % 2


def make_hard_pulse(flip_angle: float, *, system: pp.Opts | None = None, **kwargs) -> RfPulse:
    """Create a non-selective rectangular (hard) pulse module.

    The simplest excitation: a constant-amplitude block with no gradient, so
    it tips the whole excited volume. Wraps
    :func:`pypulseq.make_block_pulse` and returns it as an
    :class:`~pulserver.Module`, so it can be appended and re-offset like every
    other Pulserver RF factory.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    system : pypulseq.Opts, optional
        System limits; defaults to ``pypulseq.Opts.default``.
    **kwargs
        Forwarded to :func:`pypulseq.make_block_pulse` (``duration``,
        ``phase_offset``, ``freq_offset``, ``use``, ...).

    Returns
    -------
    RfPulse
        Module holding the RF event as ``.rf``.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> pulse = design.make_hard_pulse(np.deg2rad(90), duration=0.5e-3)
    >>> pulse.rf.t[-1]
    np.float64(0.0005)

    Append its Pulseq blocks explicitly::

        seq = pp.Sequence(pulse.system)
        pulse.set_state(phase_offset_rad=np.pi)
        for block in pulse:
            seq.add_block(*block)

    A non-selective pulse has no profile to show but its envelope:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(design.make_hard_pulse(np.deg2rad(90), duration=5e-4, system=system),
                 "none", title="make_hard_pulse(90 deg, 0.5 ms)")

    See Also
    --------
    make_slr_pulse : sharper profile at the cost of duration.
    make_slice_selective_pulse : add a selection gradient.
    """
    system = _system_or_default(system)
    event = pp.make_block_pulse(flip_angle, system=system, **kwargs)
    return RfPulse(system, event)


def make_adiabatic_pulse(
    pulse_type: str,
    *,
    system: pp.Opts | None = None,
    **kwargs,
) -> RfPulse:
    """Create an adiabatic pulse module (B1-insensitive inversion/refocusing).

    Adiabatic pulses sweep frequency and amplitude together so that, above a
    threshold B1, the achieved flip angle is independent of the transmit
    amplitude — the reason they are used for inversion and preparation in the
    presence of B1 inhomogeneity. Wraps
    :func:`pypulseq.make_adiabatic_pulse` and returns an
    :class:`~pulserver.Module`; when a ``slice_thickness`` is requested the
    selection and rephasing gradients come back on ``.gradients`` /
    ``.rephasers``.

    Parameters
    ----------
    pulse_type : str
        Adiabatic family, e.g. ``"hypsec"`` or ``"wurst"``.
    system : pypulseq.Opts, optional
        System limits; defaults to ``pypulseq.Opts.default``.
    **kwargs
        Forwarded to :func:`pypulseq.make_adiabatic_pulse` (``duration``,
        ``dwell``, ``slice_thickness``, ``return_gz``, ``use``, ...).

    Returns
    -------
    RfPulse
        Module holding the RF event and any selection gradients.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> pulse = pp.make_adiabatic_pulse("hypsec", duration=10.24e-3, dwell=10e-6)
    >>> pulse.rf.signal.shape
    (1024,)

    Above the adiabatic threshold the inversion is flat across a wide off-resonance band, and stays flat as B1 changes:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(pp.make_adiabatic_pulse("hypsec", system=system),
                 "frequency", title="adiabatic hyperbolic secant", extent=2000)

    See Also
    --------
    make_inversion_pulse : adiabatic inversion with sensible defaults.
    """
    system = _system_or_default(system)
    result = pp.make_adiabatic_pulse(pulse_type=pulse_type, system=system, **kwargs)
    if isinstance(result, tuple):
        event, gradient, rephaser = result
        return RfPulse(system, event, (gradient,), (rephaser,))
    return RfPulse(system, result)


def make_slr_pulse(
    flip_angle: float,
    *,
    duration: float = DEFAULT_DURATION,
    delay: float = 0.0,
    dwell: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    center_pos: float = 0.5,
    slice_thickness: float = 0.0,
    return_gz: bool = False,
    time_bw_product: float = DEFAULT_TIME_BANDWIDTH_PRODUCT,
    pulse_type: PulseType = "st",
    filter_type: FilterType = "ls",
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    cancel_alpha_phase: bool = False,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    system: pp.Opts | None = None,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
):
    """Create an SLR-designed pulse, without requiring SigPy.

    Shinnar-Le Roux design turns the desired *slice profile* into an RF
    envelope through an equivalent FIR filter problem, giving a far sharper
    profile than a truncated sinc at the same duration. The filter is designed
    in-package, so SigPy is not a dependency; the interface follows pypulseq's
    former SigPy pulse factory.

    ``pulse_type`` selects the profile the filter is designed for — ``"ex"``
    (90-degree excitation), ``"se"`` (spin-echo refocusing), ``"inv"``
    (inversion), ``"sat"`` (saturation), or ``"st"`` (small-tip). Using the
    wrong one gives a visibly degraded profile at large flip angles.
    ``make_sigpy_pulse`` remains as an alias for the pypulseq factory this
    replaces; new code should use this name.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    duration : float, optional
        Pulse duration (s). Bandwidth is ``time_bw_product / duration``.
    delay : float, optional
        Event delay before the RF starts (s).
    freq_offset : float, optional
        RF frequency offset (Hz).
    phase_offset : float, optional
        RF phase offset (rad).
    dwell : float, optional
        RF raster (s); ``0`` uses ``system.rf_raster_time``.
    center_pos : float, optional
        Effective-rotation position within the pulse, in ``[0, 1]``; sets the
        rephasing area when ``return_gz`` is true.
    slice_thickness : float, optional
        Slice thickness (m); required when ``return_gz`` is true.
    return_gz : bool, optional
        Also design the slice-selection and rephasing gradients.
    time_bw_product : float, optional
        Time-bandwidth product; higher means a sharper profile and more
        sidelobes for the same duration.
    pulse_type : {'st', 'ex', 'se', 'inv', 'sat'}, optional
        Profile the SLR filter is designed for.
    filter_type : {'ls', 'pm', 'min', 'max', 'ms'}, optional
        FIR design method; ``'ls'`` least-squares (default), ``'pm'``
        equiripple, ``'min'``/``'max'`` minimum/maximum phase.
    passband_ripple : float, optional
        Allowed ripple inside the passband.
    stopband_ripple : float, optional
        Allowed ripple in the stopband.
    cancel_alpha_phase : bool, optional
        Remove the SLR alpha-polynomial phase.
    max_grad : float, optional
        Gradient amplitude limit for the selection gradient; ``0`` uses
        ``system``.
    max_slew : float, optional
        Slew limit for the selection gradient; ``0`` uses ``system``.
    system : pypulseq.Opts, optional
        System limits.
    use : str, optional
        Pulseq ``use`` tag.
    freq_ppm : float, optional
        Frequency offset relative to the field strength (ppm).
    phase_ppm : float, optional
        Phase offset relative to the field strength (ppm).

    Returns
    -------
    RfPulse
        Module holding the RF event; with ``return_gz=True`` its
        ``gradients`` and ``rephasers`` hold the slice-selection pair.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> pulse = design.make_slr_pulse(np.deg2rad(90), duration=2e-3, pulse_type="ex")
    >>> pulse.rf.signal.shape
    (2000,)

    Append the RF block, and any requested selection/rephasing blocks::

        seq = pp.Sequence(pulse.system)
        for block in pulse:
            seq.add_block(*block)

    The pulse, and the profile a Bloch simulation of it produces:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(design.make_slr_pulse(np.deg2rad(90), slice_thickness=5e-3,
                                   return_gz=True, system=system),
                 "slice", title="SLR 90 deg, 5 mm, TBW 4")

    See Also
    --------
    make_slice_selective_pulse : this pulse plus its selection gradients.
    make_frequency_selective_pulse : the non-spatial, spectral counterpart.
    """
    system = _system_or_default(system)
    dwell = system.rf_raster_time if dwell == 0 else dwell
    if not 0.0 <= center_pos <= 1.0:
        raise ValueError("center_pos must lie in [0, 1]")
    if return_gz and slice_thickness <= 0:
        raise ValueError("slice_thickness must be > 0 when return_gz=True")

    n = _slr_sample_count(duration, dwell)
    waveform = design_slr(
        n,
        time_bw_product,
        pulse_type=pulse_type,
        filter_type=filter_type,
        passband_ripple=passband_ripple,
        stopband_ripple=stopband_ripple,
        cancel_alpha_phase=cancel_alpha_phase,
    )
    actual_duration = n * dwell
    result = pp.make_arbitrary_rf(
        signal=waveform,
        flip_angle=flip_angle,
        delay=delay,
        dwell=dwell,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        return_gz=return_gz,
        slice_thickness=slice_thickness,
        bandwidth=time_bw_product / actual_duration,
        time_bw_product=time_bw_product,
        max_grad=max_grad,
        max_slew=max_slew,
        system=system,
        use=use,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        center=center_pos * actual_duration,
    )
    if not return_gz:
        return RfPulse(system, result)

    rf, gz = result
    flat_area = gz.amplitude * gz.flat_time
    ramp_area = gz.area - flat_area
    rephase_area = -flat_area * (1.0 - center_pos) - 0.5 * ramp_area
    gz_reph = pp.make_trapezoid(channel="z", area=rephase_area, system=system)
    return RfPulse(system, rf, (gz,), (gz_reph,))


# Back-compatible name for pypulseq's former SigPy-backed factory.
make_sigpy_pulse = make_slr_pulse


def make_frequency_selective_pulse(
    flip_angle: float,
    bandwidth: float,
    *,
    freq_offset: float = 0.0,
    duration: float | None = None,
    time_bw_product: float = DEFAULT_TIME_BANDWIDTH_PRODUCT,
    pulse_type: PulseType = "st",
    filter_type: FilterType = "ls",
    system: pp.Opts | None = None,
    use: str = "excitation",
    **kwargs,
):
    """Create a spectrally selective SLR pulse centred at ``freq_offset``.

    Selects a frequency band rather than a slab: no gradient is played, so the
    profile acts on chemical shift (fat, water, metabolites) instead of
    position. Parameters are stated the way a spectral pulse is actually
    specified — ``bandwidth`` and ``time_bw_product`` — with the duration
    derived from them when omitted.

    Spectral pulses are long, so the RF raster defaults to 10 us: ample
    spectral resolution without solving a needlessly large FIR system.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    bandwidth : float
        Passband width (Hz).
    freq_offset : float, optional
        Centre frequency relative to the transmit frequency (Hz).
    duration : float or None, optional
        Pulse duration (s); derived as ``time_bw_product / bandwidth`` when
        omitted, and checked for consistency when given.
    time_bw_product : float, optional
        Time-bandwidth product.
    pulse_type : {'st', 'ex', 'se', 'inv', 'sat'}, optional
        SLR profile family.
    filter_type : {'ls', 'pm', 'min', 'max', 'ms'}, optional
        FIR design method.
    system : pypulseq.Opts, optional
        System limits.
    use : str, optional
        Pulseq ``use`` tag.
    **kwargs
        Forwarded to :func:`make_slr_pulse`.

    Returns
    -------
    RfPulse
        Module holding the RF event; no gradients.

    Examples
    --------
    A 250 Hz saturation band on fat at 3 T (-440 Hz from water):

    >>> import numpy as np
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> pulse = design.make_frequency_selective_pulse(
    ...     np.deg2rad(90), bandwidth=250.0, freq_offset=-440.0
    ... )
    >>> pulse.rf.freq_offset
    -440.0

    Append the pulse to a sequence::

        seq = pp.Sequence(pulse.system)
        for block in pulse:
            seq.add_block(*block)

    With no gradient the same envelope selects a band of off-resonance instead of a band of space:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(design.make_frequency_selective_pulse(np.deg2rad(90), 500.0, system=system),
                 "frequency", title="frequency-selective 90 deg, 500 Hz", extent=1200)

    See Also
    --------
    make_fat_saturation_pulse : this pulse plus its spoiler and scope labels.
    make_spsp_pulse : simultaneous spectral and spatial selectivity.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")
    if duration is None:
        duration = time_bw_product / bandwidth
    elif not math.isclose(time_bw_product / duration, bandwidth, rel_tol=0.02):
        raise ValueError("bandwidth, duration, and time_bw_product are inconsistent")
    resolved_system = _system_or_default(system)
    # Spectral pulses are typically long; 10 us retains ample spectral
    # resolution without solving unnecessarily large FIR systems at 1 us.
    kwargs.setdefault("dwell", max(10e-6, resolved_system.rf_raster_time))
    return make_slr_pulse(
        flip_angle,
        duration=duration,
        freq_offset=freq_offset,
        time_bw_product=time_bw_product,
        pulse_type=pulse_type,
        filter_type=filter_type,
        system=resolved_system,
        use=use,
        **kwargs,
    )


def make_slice_selective_pulse(
    flip_angle: float,
    slice_thickness: float,
    *,
    duration: float = DEFAULT_SLICE_DURATION,
    time_bw_product: float = DEFAULT_TIME_BANDWIDTH_PRODUCT,
    system: pp.Opts | None = None,
    use: str = "excitation",
    **kwargs,
):
    """Create a slice- or slab-selective excitation, gradients included.

    The workhorse excitation: an SLR ``"ex"`` pulse played under a selection
    gradient, followed by the rephaser that undoes the through-slice dephasing
    the pulse accrues. All three events come back in one
    :class:`~pulserver.Module`, so the whole excitation is appended in a single
    call and re-offset per slice.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    slice_thickness : float
        Slice (or slab) thickness (m). The selection gradient amplitude is
        ``bandwidth / slice_thickness``.
    duration : float, optional
        Pulse duration (s).
    time_bw_product : float, optional
        Time-bandwidth product; the pulse bandwidth is the ratio of the two.
    system : pypulseq.Opts, optional
        System limits.
    use : str, optional
        Pulseq ``use`` tag.
    **kwargs
        Forwarded to :func:`make_slr_pulse` (``center_pos``, ``filter_type``,
        ``phase_offset``, ...).

    Returns
    -------
    RfPulse
        Module with ``.rf``, ``.gradients`` (selection) and ``.rephasers``.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> pulse = design.make_slice_selective_pulse(np.deg2rad(30), 5e-3)
    >>> pulse.gradients[0].channel, len(pulse.rephasers)
    ('z', 1)

    Move the excitation to another slice, then append every block::

        seq = pp.Sequence(pulse.system)
        pulse.set_state(freq_offset_hz=pulse.gradients[0].amplitude * position_m)
        for block in pulse:
            seq.add_block(*block)

    Time-bandwidth product sets the sharpness of the slice edges — the same nominal thickness, three different profiles:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_comparison
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_comparison(
           [(f"TBW = {tbw:g}", design.make_slice_selective_pulse(
               np.deg2rad(90), 5e-3, time_bw_product=tbw, system=system))
            for tbw in (2.0, 4.0, 8.0)],
           "slice", extent=0.012, title="make_slice_selective_pulse(90 deg, 5 mm)")

    See Also
    --------
    make_slr_pulse : the underlying pulse design.
    make_slice_loop : plan the physical slice/SMS loop and its frequency offsets.
    """
    return make_slr_pulse(
        flip_angle,
        duration=duration,
        slice_thickness=slice_thickness,
        return_gz=True,
        time_bw_product=time_bw_product,
        pulse_type="ex",
        system=system,
        use=use,
        **kwargs,
    )


def make_refocusing_pulse(
    *,
    slice_thickness: float | None = None,
    duration: float = 3e-3,
    time_bw_product: float = DEFAULT_TIME_BANDWIDTH_PRODUCT,
    phase_offset: float = np.pi / 2.0,
    system: pp.Opts | None = None,
    **kwargs,
):
    """Create a nominal 180-degree SLR refocusing pulse.

    Designed with the ``"se"`` SLR profile — the one that matches a
    refocusing rather than an excitation response — and phased 90 degrees
    relative to the excitation by default, as CPMG requires.

    With ``slice_thickness=None`` the module is broadband and non-selective;
    otherwise the selection and rephasing events appear on ``gradients`` and
    ``rephasers``. FSE code typically takes ``module.rf`` and
    ``module.gradients[0]`` and rescales the envelope per echo — see
    :func:`make_traps_schedule` and :func:`make_fse_readout`.

    Parameters
    ----------
    slice_thickness : float or None, optional
        Slice thickness (m), or ``None`` for a non-selective pulse.
    duration : float, optional
        Pulse duration (s).
    time_bw_product : float, optional
        Time-bandwidth product.
    phase_offset : float, optional
        RF phase (rad); the CPMG-correct ``pi/2`` by default.
    system : pypulseq.Opts, optional
        System limits.
    **kwargs
        Forwarded to :func:`make_slr_pulse`.

    Returns
    -------
    RfPulse
        Module with ``.rf`` and, when selective, its gradients.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> refocusing = design.make_refocusing_pulse(slice_thickness=5e-3)
    >>> np.rad2deg(refocusing.rf.phase_offset)
    np.float64(90.0)

    Append the RF, selection-gradient, and rephasing blocks::

        seq = pp.Sequence(refocusing.system)
        for block in refocusing:
            seq.add_block(*block)

    Build an FSE train that reuses it at reduced flip angles::

        readout = design.make_fse_readout(system, fov, matrix, 32, refocusing)
        flips = design.make_traps_schedule(32, np.deg2rad(120))

    A refocusing pulse is judged by how completely it inverts across the slice, not by its excitation profile:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(design.make_refocusing_pulse(slice_thickness=5e-3, system=system),
                 "slice", title="refocusing 180 deg, 5 mm")

    See Also
    --------
    make_fse_readout : consumes this module to build the echo train.
    make_traps_schedule : variable refocusing flip angles.
    """
    result = make_slr_pulse(
        np.pi,
        duration=duration,
        phase_offset=phase_offset,
        slice_thickness=0.0 if slice_thickness is None else slice_thickness,
        return_gz=slice_thickness is not None,
        time_bw_product=time_bw_product,
        pulse_type="se",
        system=system,
        use="refocusing",
        **kwargs,
    )
    result.slice_thickness = slice_thickness
    return result


def make_inversion_pulse(
    *,
    adiabatic: bool = True,
    duration: float = 10.24e-3,
    time_bw_product: float = 6.0,
    dwell: float = 10e-6,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    system: pp.Opts | None = None,
):
    """Create a 180-degree inversion pulse, adiabatic by default.

    Inversion sets the starting point of every T1-weighted contrast, so the
    achieved flip angle must not vary with transmit B1 — hence the adiabatic
    hyperbolic-secant default. Set ``adiabatic=False`` for an SLR ``"inv"``
    pulse, which is shorter and deposits less power but is B1-sensitive.

    This factory is intentionally non-selective. Spatially selective
    inversion has sequence-specific adiabaticity and gradient constraints and
    is therefore not exposed as a generic preparation.

    Parameters
    ----------
    adiabatic : bool, optional
        Use a hyperbolic-secant adiabatic pulse (default True).
    duration : float, optional
        Pulse duration (s); adiabaticity needs a long pulse, hence the
        10.24 ms default.
    time_bw_product : float, optional
        Time-bandwidth product for the non-adiabatic branch.
    dwell : float, optional
        RF sample spacing (s).
    freq_offset : float, optional
        RF frequency offset (Hz).
    phase_offset : float, optional
        RF phase offset (rad).
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    RfPulse
        Non-selective module holding ``.rf``.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> inversion = design.make_inversion_pulse()
    >>> len(inversion.gradients)
    0

    Append the non-selective inversion block::

        seq = pp.Sequence(inversion.system)
        for block in inversion:
            seq.add_block(*block)

    Adiabatic and conventional non-selective inversion envelopes:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_comparison
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_comparison(
           [("adiabatic", design.make_inversion_pulse(system=system)),
            ("SLR", design.make_inversion_pulse(adiabatic=False, system=system))],
           "frequency", extent=1500,
           title="non-selective inversion: final magnetization")

    """
    system = _system_or_default(system)
    if adiabatic:
        result = pp.make_adiabatic_pulse(
            pulse_type="hypsec",
            duration=duration,
            dwell=dwell,
            freq_offset=freq_offset,
            phase_offset=phase_offset,
            system=system,
            use="inversion",
        )
        return RfPulse(system, result)
    return make_slr_pulse(
        np.pi,
        duration=duration,
        dwell=dwell,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        time_bw_product=time_bw_product,
        pulse_type="inv",
        system=system,
        use="inversion",
    )
