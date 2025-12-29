"""Slice selection subroutine."""

__all__ = ['as_spatial_selective']

from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Union

import numpy as np
from pypulseq import Opts
from pypulseq.add_gradients import add_gradients
from pypulseq.align import align
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.scale_grad import scale_grad

from ._utils import verse


def as_spatial_selective(
    pulse: SimpleNamespace,
    slice_thickness: float,
    lobes: int = 1,
    duration: Union[float, None] = None,
    bandwidth: float = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    system: Union[Opts, None] = None,
    time_bw_product: float = 0,
    apodization: float = 0,
) -> tuple[SimpleNamespace, SimpleNamespace] | SimpleNamespace:
    """
    Transform frequency selective pulse into slice selective pulse.

    Parameters
    ----------
    pulse : SimpleNamespace
        Input RF pulse event.
    slice_thickness : float
        DESCRIPTION.
    lobes : int, optional
        DESCRIPTION. The default is 1.
    duration : Union[float, None], optional
        DESCRIPTION. The default is None.
    bandwidth : float, optional
        DESCRIPTION. The default is 0.0.
    max_grad : float, optional
        DESCRIPTION. The default is 0.0.
    max_slew : float, optional
        DESCRIPTION. The default is 0.0.
    system : Union[Opts, None], optional
        DESCRIPTION. The default is None.
    time_bw_product : float, optional
        DESCRIPTION. The default is 0.
    apodization : float, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if lobes == 1 or duration == pulse.shape_dur:
        return _as_spatial_selective(
            pulse,
            slice_thickness,
            bandwidth,
            max_grad,
            max_slew,
            system,
            time_bw_product,
        )
    return _as_adiabatic_spatial_selective(
        pulse,
        slice_thickness,
        lobes,
        duration,
        bandwidth,
        max_grad,
        max_slew,
        system,
        time_bw_product,
        apodization,
    )


def _as_spatial_selective(
    pulse: SimpleNamespace,
    slice_thickness: float,
    bandwidth: float = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    system: Union[Opts, None] = None,
    time_bw_product: float = 0,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    if system is None:
        system = Opts.default
    if max_grad > 0:
        system = copy(system)
        system.max_grad = max_grad
    if max_slew > 0:
        system = copy(system)
        system.max_slew = max_slew
    if bandwidth == 0 and time_bw_product == 0:
        raise ValueError('User must provide bandwidth or time-bandwidth product.')
    if bandwidth != 0 and time_bw_product != 0:
        raise ValueError(
            'User must provide either bandwidth or time-bandwidth product, not both.'
        )
    if bandwidth == 0:
        bandwidth = time_bw_product / pulse.shape_dur

    # Compute trapezoid params
    amplitude = bandwidth / slice_thickness
    flat_area = amplitude * pulse.shape_dur
    gz = make_trapezoid(
        channel='z', system=system, flat_time=pulse.shape_dur, flat_area=flat_area
    )

    # Compute rephasing params
    center_pos = pulse.center / pulse.shape_dur
    gz_rephase = make_trapezoid(
        channel='z',
        system=system,
        area=-flat_area * (1 - center_pos) - 0.5 * (gz.area - flat_area),
    )

    # Shift gradient within the block to align with rf pulse
    if pulse.delay > gz.rise_time:
        gz.delay = (
            np.ceil((pulse.delay - gz.rise_time) / system.grad_raster_time)
            * system.grad_raster_time
        )

    return gz, gz_rephase


def _as_adiabatic_spatial_selective(
    pulse: SimpleNamespace,
    slice_thickness: float,
    lobes: int | None = None,
    duration: Union[float, None] = None,
    bandwidth: float = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    system: Union[Opts, None] = None,
    time_bw_product: float = 0,
    apodization: float = 0,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """
    Generate a slice-selective adiabatic pulse using sinc subpulses and
    PyPulseq align + add_gradients.

    Returns
    -------
    rf_block : SimpleNamespace
        Arbitrary RF block containing the full concatenated RF waveform.
    gz_combined : SimpleNamespace
        Combined slice-selection gradient for the entire pulse.
    """
    if system is None:
        system = Opts.default

    # Get duration
    T = pulse.shape_dur

    # Case 1: lobes only
    if lobes is not None and duration is None:
        duration = T / lobes

    # Case 2: duration only
    elif duration is not None and lobes is None:
        lobes = round(T / duration)
        duration = T / lobes  # recompute exact

    # Case 3: both lobes and duration given → validate
    elif lobes is not None and duration is not None:
        raise ValueError(
            'Please specify at most number of lobes or lobes duration, not both'
        )

    # Case 4: neither given → choose default
    else:
        lobes = 50
        duration = T / lobes

    # TBW vs bandwidth
    if time_bw_product > 0 and bandwidth == 0:
        bandwidth = time_bw_product / duration
    elif bandwidth > 0 and time_bw_product == 0:
        time_bw_product = bandwidth * duration
    elif bandwidth > 0 and time_bw_product > 0:
        raise ValueError(
            'Please specify either subpulse bandwidth or its TBW, not both'
        )
    else:
        raise ValueError('Please specify subpulse bandwidth or its TBW')

    # Get dwell time
    dwell = T / pulse.signal.size

    # Extract rf_envelope
    rf_env = np.asarray(pulse.signal)
    Lenv = len(rf_env)

    # Initialize subpulses
    rf_events = []
    gz_events = []

    # Base sub-pulse
    rf_sub_base, gz_sub_base = make_sinc_pulse(
        flip_angle=np.pi / 2,  # does not matter
        apodization=apodization,
        delay=pulse.delay,
        duration=duration,
        dwell=dwell,
        center_pos=0.5,
        freq_offset=pulse.freq_offset,
        max_grad=max_grad,
        max_slew=max_slew,
        phase_offset=pulse.phase_offset,
        return_gz=True,
        slice_thickness=slice_thickness,
        system=system,
        time_bw_product=time_bw_product,
        use=pulse.use,
        freq_ppm=pulse.freq_ppm,
        phase_ppm=pulse.phase_ppm,
    )
    rf_sub_base /= rf_sub_base.sum()

    # Generate sublobes
    for ii in range(lobes):
        idx = int(np.ceil((2 * ii + 1) / 2 * pulse.signal.size / lobes)) - 1
        idx = max(0, min(idx, Lenv - 1))
        env_value = pulse.signal[idx]

        # Deep copy
        rf_sub = deepcopy(rf_sub_base)

        # Scale RF by envelope sample
        rf_sub.signal *= env_value

        # Verse the RF subpulse
        rf_sub = verse(gz_sub_base, rf_sub)

        # Flip gradient polarity every other lobe
        gz_sub = scale_grad(gz_sub_base, (-1) ** ii)

        # Store blocks
        rf_events.append(rf_sub)
        gz_events.append(gz_sub)

    # Align subpulses
    aligned_gz_events = align(right=gz_events)

    # Combine gradients into single block
    gz_combined = add_gradients(aligned_gz_events, max_grad, max_slew, system)

    # Concatenate RF signals from all blocks
    signal_combined = [
        points_to_waveform(rf.signal, system.rf_raster_time, rf.t) for rf in rf_events
    ]
    signal_combined = np.concatenate(signal_combined)

    # Pass to PyPulseq
    return (
        make_arbitrary_rf(
            signal_combined,
            np.pi,  # flip_angle, not used when no_signal_scaling=True
            1e3,  # bandwidth, not used w/o slice selection
            pulse.delay,
            dwell,
            pulse.freq_offset,
            True,  # no_signal_scaling
            0,  # max_grad, not used w/o slice selection
            0,  # max_slew, not used w/o slice selection
            pulse.phase_offset,
            False,  # return_gz
            0,  # slice_thickness
            system,
            1,  # time bandwidth product, not used w/o slice selection
            pulse.use,
            pulse.freq_ppm,
            pulse.phase_ppm,
        ),
        gz_combined,
    )
