"""Spectral multiband and PINS multislice RF transformations."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import numpy as np
import pypulseq as pp

from ...pypulseq._opts import default_system
from ._base import RfPulse


def _rf_pulse(value) -> tuple[object, pp.Opts, tuple, tuple]:
    if isinstance(value, RfPulse):
        return value.rf, value.system, value.gradients, value.rephasers
    if getattr(value, "type", None) == "rf":
        return value, default_system(None), (), ()
    raise TypeError("pulse must be an RfPulse or a Pulseq RF event")


def _band_offsets(num_bands: int, band_offset: float) -> np.ndarray:
    num_bands = int(num_bands)
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1")
    if band_offset <= 0 and num_bands > 1:
        raise ValueError("band_offset must be > 0 when num_bands > 1")
    # Even counts cannot be both symmetric and include zero. Keep the nearest
    # symmetric pairs and place the unmatched band on the positive side.
    indices = np.arange(-((num_bands - 1) // 2), num_bands // 2 + 1)
    return indices.astype(float) * float(band_offset)


def _band_weights(num_bands: int, sideband_power, phases) -> np.ndarray:
    if np.isscalar(sideband_power):
        powers = np.full(num_bands, float(sideband_power))
        powers[(num_bands - 1) // 2] = 1.0
    else:
        powers = np.asarray(sideband_power, dtype=float)
        if powers.shape != (num_bands,):
            raise ValueError("sideband_power must be a scalar or one value per band")
    if np.any(~np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("band powers must be finite and >= 0")
    if phases is None:
        phase = np.zeros(num_bands)
    elif isinstance(phases, str) and phases == "quadratic":
        position = np.arange(num_bands) - (num_bands - 1) / 2.0
        phase = (3.4 / num_bands) * position**2
    elif isinstance(phases, str):
        raise ValueError("phases must be 'quadratic' or one value per band")
    else:
        phase = np.asarray(phases, dtype=float)
        if phase.shape != (num_bands,):
            raise ValueError("phases must be 'quadratic' or one value per band")
    return np.sqrt(powers) * np.exp(1j * phase)


def make_multiband(
    pulse,
    num_bands: int,
    band_offset: float,
    *,
    sideband_power: float | Sequence[float] = 1.0,
    phases: str | Sequence[float] | None = None,
) -> RfPulse:
    """Convert an RF pulse into equispaced spectral bands.

    ``sideband_power`` is power relative to the on-resonance band, rather
    than an amplitude multiplier; the modulation therefore uses its square
    root.  A scalar applies to every off-resonance band.  Supplying one value
    per band gives direct control of asymmetric saturation.

    The offset list always contains 0 Hz. For an even number of bands the
    otherwise-unpaired band is placed on the positive-frequency side.

    Parameters
    ----------
    pulse : RfPulse or pypulseq RF event
        Base pulse to modulate into bands.
    num_bands : int
        Number of frequency bands, including the on-resonance band.
    band_offset : float
        Frequency spacing between adjacent bands (Hz).
    sideband_power : float or sequence of float, optional
        Power of each off-resonance band relative to the on-resonance band
        (a scalar applies to every off-resonance band); the modulation uses
        its square root.
    phases : {'quadratic'} or sequence of float, optional
        Per-band phase (rad). ``'quadratic'`` is the MATLAB reference
        multiband phase schedule; ``None`` leaves every band in phase.

    Returns
    -------
    RfPulse
        Modulated pulse, with ``.band_offsets`` (Hz) and ``.band_weights``
        (complex, one per band) recording the applied schedule.

    Examples
    --------
    Convert any user-designed RF module and append the resulting blocks::

        import numpy as np
        import pulserver.design as design
        import pulserver.pypulseq as pp

        base = design.make_frequency_selective_pulse(np.deg2rad(30), 300.0)
        pulse = design.make_multiband(base, 4, 1000.0, sideband_power=2.0)
        seq = pp.Sequence(pulse.system)
        for block in pulse:
            seq.add_block(*block)

    Two left-shifted bands, two right-shifted bands, and the centred three-band
    convention used by the MATLAB reference implementation:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import frequency_profile_gallery
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       base = lambda offset: design.make_frequency_selective_pulse(
           np.deg2rad(30), 300.0, freq_offset=offset, system=system)
       frequency_profile_gallery([
           ("two bands left", design.make_multiband(base(-2000.0), 2, 1000.0)),
           ("two bands right", design.make_multiband(base(1000.0), 2, 1000.0)),
           ("three bands (MATLAB reference)", design.make_multiband(base(0.0), 3, 1000.0)),
       ])
    """
    rf, system, gradients, rephasers = _rf_pulse(pulse)
    offsets = _band_offsets(num_bands, band_offset)
    weights = _band_weights(len(offsets), sideband_power, phases)
    time = np.asarray(rf.t, dtype=float)
    modulation = np.sum(
        weights[:, None] * np.exp(2j * np.pi * offsets[:, None] * time[None, :]),
        axis=0,
    )
    event = copy.deepcopy(rf)
    event.signal = np.asarray(event.signal) * modulation
    result = RfPulse(
        system, event, tuple(copy.deepcopy(g) for g in gradients), tuple(copy.deepcopy(g) for g in rephasers)
    )
    result.band_offsets = offsets
    result.band_weights = weights
    return result


def make_multiband_frequency_selective_pulse(
    flip_angle: float,
    bandwidth: float,
    num_bands: int,
    band_offset: float,
    *,
    sideband_power: float | Sequence[float] = 1.0,
    phases: str | Sequence[float] | None = None,
    **kwargs,
) -> RfPulse:
    """Design an SLR spectral pulse and replicate it into multiple bands.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad), applied to the on-resonance band.
    bandwidth : float
        Per-band frequency-selective bandwidth (Hz).
    num_bands : int
        Number of frequency bands, including the on-resonance band.
    band_offset : float
        Frequency spacing between adjacent bands (Hz).
    sideband_power : float or sequence of float, optional
        Power of each off-resonance band relative to the on-resonance band.
    phases : {'quadratic'} or sequence of float, optional
        Per-band phase (rad); see :func:`make_multiband`.
    **kwargs
        Forwarded to :func:`~pulserver.design.make_frequency_selective_pulse`.

    Returns
    -------
    RfPulse
        Multiband frequency-selective pulse; see :func:`make_multiband`.

    Examples
    --------
    Four equispaced saturation bands, always including 0 Hz::

        import numpy as np
        import pulserver.design as design
        import pulserver.pypulseq as pp

        pulse = design.make_multiband_frequency_selective_pulse(
            np.deg2rad(30), 300.0, 4, 1000.0, sideband_power=2.0)
        seq = pp.Sequence(pulse.system)
        for block in pulse:
            seq.add_block(*block)

    The centred three-band reference convention:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(design.make_multiband_frequency_selective_pulse(
           np.deg2rad(30), 300.0, 3, 1000.0, system=system),
           "frequency", title="three-band frequency-selective pulse", extent=2600)
    """
    from ._excitation import make_frequency_selective_pulse

    base = make_frequency_selective_pulse(flip_angle, bandwidth, freq_offset=0.0, **kwargs)
    return make_multiband(
        base,
        num_bands,
        band_offset,
        sideband_power=sideband_power,
        phases=phases,
    )


def _slice_positions(num_slices: int, spacing: float, include_center: bool) -> np.ndarray:
    num_slices = int(num_slices)
    if num_slices < 1:
        raise ValueError("num_slices must be >= 1")
    if spacing <= 0:
        raise ValueError("slice_offset must be > 0")
    if include_center:
        indices = np.arange(-((num_slices - 1) // 2), num_slices // 2 + 1, dtype=float)
    else:
        indices = np.arange(-(num_slices // 2), -(num_slices // 2) + num_slices, dtype=float) + 0.5
    return indices * spacing


def _extended_blip_train(system, count: int, hard_duration: float, blip) -> object:
    times = [0.0]
    amplitudes = [0.0]
    cursor = hard_duration
    for _index in range(count - 1):
        knots = (
            cursor,
            cursor + blip.rise_time,
            cursor + blip.rise_time + blip.flat_time,
            cursor + blip.rise_time + blip.flat_time + blip.fall_time,
        )
        values = (0.0, blip.amplitude, blip.amplitude, 0.0)
        for knot, value in zip(knots, values, strict=True):
            if math.isclose(knot, times[-1], abs_tol=1e-15):
                amplitudes[-1] = value
            else:
                times.append(knot)
                amplitudes.append(value)
        cursor = knots[-1] + hard_duration
    if count == 1:
        times.append(hard_duration)
        amplitudes.append(0.0)
    return pp.make_extended_trapezoid(
        channel="z",
        times=np.asarray(times),
        amplitudes=np.asarray(amplitudes),
        system=system,
    )


def make_pins(
    pulse,
    num_slices: int,
    slice_offset: float,
    *,
    slice_thickness: float,
    include_center: bool = False,
    center_offset: float | None = None,
    max_rf: float | None = None,
) -> RfPulse:
    """Convert a slice-selective pulse into a PINS pulse train.

    The supplied pulse must carry its slice-selection gradient. The PINS
    gradient blip area is ``1 / slice_offset``. ``include_center=False``
    shifts the periodic slice comb by half a spacing, avoiding a slice at the
    nominal position; set it true for the alternative SMS convention.
    ``center_offset`` translates the complete comb and is useful for cycling
    through SMS slice groups while retaining the same band separation.

    Parameters
    ----------
    pulse : RfPulse or pypulseq RF event
        Slice-selective base pulse, carrying its one z selection gradient.
    num_slices : int
        Number of slices in the comb.
    slice_offset : float
        Center-to-center spacing between comb slices (m); sets the PINS
        gradient blip area (``1 / slice_offset``).
    slice_thickness : float
        Thickness of the base pulse's slice, used to size the subpulse train.
    include_center : bool, optional
        If True, the comb is centered on the nominal slice position; if
        False (default), it is shifted by half a spacing to avoid it.
    center_offset : float or None, optional
        Additional translation of the whole comb (m), for cycling through
        interleaved SMS groups.
    max_rf : float or None, optional
        Peak subpulse amplitude; defaults to the base pulse's peak.

    Returns
    -------
    RfPulse
        PINS pulse train, with ``.slice_positions`` (m), ``.slice_offset``
        and ``.slice_thickness`` recording the comb geometry.

    Examples
    --------
    Transform a user-designed slice-selective module and append it::

        import numpy as np
        import pulserver.design as design
        import pulserver.pypulseq as pp

        base = design.make_slice_selective_pulse(np.deg2rad(15), 3e-3)
        pulse = design.make_pins(base, 6, 18e-3, slice_thickness=3e-3)
        seq = pp.Sequence(pulse.system)
        for block in pulse:
            seq.add_block(*block)

    The sparse RF train produces the expected periodic slice profile:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       base = design.make_slice_selective_pulse(np.deg2rad(15), 3e-3, system=system)
       rf_figure(design.make_pins(base, 6, 18e-3, slice_thickness=3e-3),
                 "slice", title="PINS, six offset slices", extent=60e-3)
    """
    rf, system, gradients, _ = _rf_pulse(pulse)
    if len(gradients) != 1 or gradients[0].channel != "z":
        raise ValueError("PINS requires a pulse with one z selection gradient")
    if slice_thickness <= 0:
        raise ValueError("slice_thickness must be > 0")

    selection = gradients[0]
    bandwidth = abs(float(selection.amplitude)) * float(slice_thickness)
    time_bw_product = bandwidth * float(rf.shape_dur)
    n_subpulses = max(2, int(np.ceil(time_bw_product * slice_offset / slice_thickness)))

    # Resample the user's envelope to the PINS coefficients and preserve its
    # complex RF area. Each coefficient becomes one short rectangular pulse.
    source = np.asarray(rf.signal, dtype=complex)
    source_area = source * (float(rf.shape_dur) / len(source))
    edges = np.linspace(0, len(source_area), n_subpulses + 1)
    coefficients = np.array(
        [source_area[int(round(edges[i])) : int(round(edges[i + 1]))].sum() for i in range(n_subpulses)]
    )
    if not np.any(coefficients):
        raise ValueError("pulse has zero RF area")

    max_rf = float(np.max(np.abs(source))) if max_rf is None else float(max_rf)
    if max_rf <= 0:
        raise ValueError("max_rf must be > 0")
    dwell = system.rf_raster_time
    block_raster = max(system.grad_raster_time, system.block_duration_raster)
    hard_duration = math.ceil((np.max(np.abs(coefficients)) / max_rf) / block_raster) * block_raster
    hard_duration = max(block_raster, hard_duration)
    hard_samples = int(round(hard_duration / dwell))

    blip = pp.make_trapezoid(channel="z", area=1.0 / slice_offset, system=system)
    blip_duration = pp.calc_duration(blip)
    total_duration = n_subpulses * hard_duration + (n_subpulses - 1) * blip_duration
    output = np.zeros(int(round(total_duration / dwell)), dtype=complex)
    center_offset = 0.0 if center_offset is None else float(center_offset)
    comb_shift = (0.0 if include_center else 0.5 * slice_offset) + center_offset
    cursor = 0
    for index, coefficient in enumerate(coefficients):
        phase = np.exp(-2j * np.pi * index * comb_shift / slice_offset)
        output[cursor : cursor + hard_samples] = coefficient * phase / hard_duration
        cursor += hard_samples
        if index < n_subpulses - 1:
            cursor += int(round(blip_duration / dwell))

    event = pp.make_arbitrary_rf(
        signal=output,
        flip_angle=1.0,
        dwell=dwell,
        no_signal_scaling=True,
        freq_offset=float(rf.freq_offset),
        phase_offset=float(rf.phase_offset),
        system=system,
        use=rf.use,
    )
    gradient = _extended_blip_train(system, n_subpulses, hard_duration, blip)
    rephaser = pp.make_trapezoid(channel="z", area=-0.5 * gradient.area, system=system)
    result = RfPulse(system, event, (gradient,), (rephaser,))
    positions = _slice_positions(num_slices, slice_offset, include_center)
    result.slice_positions = positions + center_offset
    result.slice_offset = float(slice_offset)
    result.slice_thickness = float(slice_thickness)
    return result


def make_pins_slice_selective_pulse(
    flip_angle: float,
    slice_thickness: float,
    num_slices: int,
    slice_offset: float,
    *,
    include_center: bool = False,
    center_offset: float | None = None,
    **kwargs,
) -> RfPulse:
    """Design a slice-selective SLR pulse and convert it to PINS.

    ``center_offset`` translates the complete slice comb in metres, allowing
    several calls to cover interleaved SMS groups.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    slice_thickness : float
        Thickness of each slice in the comb (m).
    num_slices : int
        Number of slices in the comb.
    slice_offset : float
        Center-to-center spacing between comb slices (m).
    include_center : bool, optional
        If True, center the comb on the nominal slice position; if False
        (default), shift it by half a spacing to avoid it.
    center_offset : float or None, optional
        Additional translation of the whole comb (m).
    **kwargs
        Forwarded to :func:`~pulserver.design.make_slice_selective_pulse`.

    Returns
    -------
    RfPulse
        PINS pulse train; see :func:`make_pins`.

    Examples
    --------
    Design six slices, offset from the nominal slice by default::

        import numpy as np
        import pulserver.design as design
        import pulserver.pypulseq as pp

        pulse = design.make_pins_slice_selective_pulse(
            np.deg2rad(15), 3e-3, 6, 18e-3)
        seq = pp.Sequence(pulse.system)
        for block in pulse:
            seq.add_block(*block)

    The slice comb includes no nominal-position slice by default:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(design.make_pins_slice_selective_pulse(
           np.deg2rad(15), 3e-3, 6, 18e-3, system=system),
           "slice", title="PINS slice-selective design", extent=60e-3)
    """
    from ._excitation import make_slice_selective_pulse

    base = make_slice_selective_pulse(flip_angle, slice_thickness, **kwargs)
    return make_pins(
        base,
        num_slices,
        slice_offset,
        slice_thickness=slice_thickness,
        include_center=include_center,
        center_offset=center_offset,
    )
