"""Pulseq Readout design helpers."""

__all__ = [
    'fse_line_readout',
    'general_line_readout',
    'line_readout',
    'spoiled_line_readout',
]

from types import SimpleNamespace

import numpy as np

from .. import pulseq as pp


def general_line_readout(
    system: pp.Opts,
    fov_m: float,
    npix: int,
    receive_bandwidth_Hz: float = 250e3,
    oversamp: float = 1.0,
    partial_fourier_factor: float = 1.0,
    rampsamp: bool = False,
    left_ramp: bool = True,
    right_ramp: bool = True,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """
    Create basic trapezoidal readout lobe.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits.
    fov_m : float
        Field of view along readout direction in ``[m]``.
    npix : int
        Image size along readout direction.
    receive_bandwidth_Hz : float, optional
        Receive bandwidth in ``[Hz]``.
        The default is ``250e3 Hz``.
    oversamp : float, optional
        Readout oversampling factor.
        The default is ``1.0``.
    partial_fourier_factor : float, optional
        Partial Fourier Factor for asymmeric echo.
        The default is ``1.0``.
    rampsamp : bool, optional
        If ``True``, sample points on trapezoid ramps.
        The default is ``Fals``e.
    left_ramp : bool, optional
        Include ramp-up part of trapezoid in readout event.
        The default is ``True``.
    right_ramp : bool, optional
        Include ramp-down part of trapezoid in readout event.
        The default is ``True``.

    Raises
    ------
    ValueError
        If ``rampsamp`` is ``True``, and either ``left_ramp`` or ``right_ramp``
        are ``True``.

    Returns
    -------
    readout : SimpleNamespace
        Object containing readout gradient and accompanying ADC event.
    metadata : SimpleNamespace
        Object containing the following additional info
            - system : pp.Opts
                  System parameters used to generate event.
            - encoding_size : int
                  Image size along readout direction.
            - encoding_center : int
                  ADC index corresponding to k-space center along readout direction.
            - encoding_center_time : float
                  Sampling time of k-space center along readout direction in ``[s]``.
            - readout_time : float
                  Total readout time in ``[s]``.
            - pre_readout_area : float
                  Readout gradient area before echo in ``[Hz/m]``.
            - post_readout_area : float
                  Readout gradient area after echo in ``[Hz/m]``.

    """
    if rampsamp and (not (left_ramp) or not (right_ramp)):
        raise ValueError('"rampsamp" option is not compatible with cropped readout')

    # Save metadata
    metadata = SimpleNamespace(system=system, image_size=npix)

    # Get readout parameters
    readout_area, readout_time, num_samples = pp.calc_kspace_readout_params(
        fov_m,
        npix,
        receive_bandwidth_Hz,
        oversamp,
        system.adc_raster_time,
        system.grad_raster_time,
    )
    dwell_time = readout_time / num_samples

    # Update metadata
    metadata.encoding_size = num_samples

    # Make sure partial_fourier leads to integer number of samples
    act_num_samples = np.ceil(partial_fourier_factor * num_samples).astype(int).item()
    partial_fourier_factor = act_num_samples / num_samples

    # Update metadata
    metadata.readout_time = act_num_samples * dwell_time
    metadata.post_echo_area = 0.5 * readout_area
    metadata.pre_echo_area = (partial_fourier_factor - 0.5) * readout_area

    # Apply partial fourier undersampling
    readout_area = partial_fourier_factor * readout_area

    # Design readout gradient
    if rampsamp:
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            area=readout_area,
            duration=metadata.readout_time,
        )
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            area=readout_area,
            duration=pp.calc_duration(gx_read) + 2 * system.adc_dead_time,
        )
        adc_delay = system.adc_dead_time
        pre_echo_area_offset = 0.0
        post_echo_area_offset = 0.0
    else:
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            flat_area=readout_area,
            flat_time=metadata.readout_time,
        )
        adc_delay = gx_read.rise_time
        pre_echo_area_offset = 0.5 * gx_read.rise_time * gx_read.amplitude
        post_echo_area_offset = 0.5 * gx_read.fall_time * gx_read.amplitude

    # Cut ramps if ADC block only includes flat part of trapezoid
    if left_ramp is False and right_ramp is False:
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            flat_area=readout_area,
            flat_time=gx_read.flat_time + 2 * system.adc_dead_time,
        )
        _, gx_read, _ = pp.split_waveform(gx_read, system=system)
        gx_read.delay = 0.0
        adc_delay = system.adc_dead_time
        pre_echo_area_offset = 0.0
        post_echo_area_offset = 0.0
    elif left_ramp is False:
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            flat_area=readout_area,
            flat_time=gx_read.flat_time + system.adc_dead_time,
        )
        _, gx_read = pp.split_waveform_at(
            gx_read,
            time_point=gx_read.rise_time,
            system=system,
        )
        gx_read.delay = 0.0
        adc_delay = system.adc_dead_time
        pre_echo_area_offset = 0.0
    elif right_ramp is False:
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            flat_area=readout_area,
            flat_time=gx_read.flat_time + system.adc_dead_time,
        )
        adc_delay = gx_read.rise_time
        gx_read, _ = pp.split_waveform_at(
            gx_read,
            time_point=gx_read.rise_time + gx_read.flat_time,
            system=system,
        )
        post_echo_area_offset = 0.0

    # Update metadata
    metadata.pre_echo_area += pre_echo_area_offset
    metadata.post_echo_area += post_echo_area_offset

    # Design Echo filter
    adc = pp.make_adc(
        system=system, delay=adc_delay, num_samples=act_num_samples, dwell=dwell_time
    )

    # Compute echo
    metadata.encoding_center = act_num_samples - (num_samples // 2)
    metadata.encoding_center_time = adc.delay + metadata.encoding_center * adc.dwell

    # Assign parameters
    readout = SimpleNamespace(gx=gx_read, adc=adc)

    return readout, metadata


def line_readout(
    system: pp.Opts,
    fov_m: float,
    npix: int,
    receive_bandwidth_Hz: float = 250e3,
    oversamp: float = 1.0,
    partial_fourier_factor: float = 1.0,
    rampsamp: bool = False,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """
    Create balanced line readout lobe.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits.
    fov_m : float
        Field of view along readout direction in ``[m]``.
    npix : int
        Image size along readout direction.
    receive_bandwidth_Hz : float, optional
        Receive bandwidth in ``[Hz]``.
        The default is ``250e3 Hz``.
    oversamp : float, optional
        Readout oversampling factor.
        The default is ``1.0``.
    partial_fourier_factor : float, optional
        Partial Fourier Factor for asymmeric echo.
        The default is ``1.0``.
    rampsamp : bool, optional
        If ``True``, sample points on trapezoid ramps.
        The default is ``Fals``e.

    Returns
    -------
    readout : SimpleNamespace
        Object containing the following events:
            - gx_prewind : SimpleNamespace
                  Prewinder gradient event.
            - gx_rewind : SimpleNamespace
                  Rewinder gradient event.
            - gx_read : SimpleNamespace
                  Readout gradient event.
            - adc : SimpleNamespace
                  Accompanying ADC event
    metadata : SimpleNamespace
        Object containing the following additional info
            - system : pp.Opts
                  System parameters used to generate event.
            - encoding_size : int
                  Image size along readout direction.
            - encoding_center : int
                  ADC index corresponding to k-space center along readout direction.
            - encoding_center_time : float
                  Sampling time of k-space center along readout direction in ``[s]``.
            - readout_time : float
                  Total readout time in ``[s]``.

    """
    # General line readout including ramps
    block, metadata = general_line_readout(
        system,
        fov_m,
        npix,
        receive_bandwidth_Hz,
        oversamp,
        partial_fourier_factor,
        rampsamp,
    )

    # Calculate prewinder
    gx_prewind, _, _ = pp.make_extended_trapezoid_area(
        channel='x',
        system=system,
        area=-metadata.pre_echo_area,
        grad_start=0.0,
        grad_end=0.0,
    )

    # Calculate rewinder
    if metadata.pre_echo_area == metadata.post_echo_area:
        gx_rewind = gx_prewind
    else:
        gx_rewind, _, _ = pp.make_extended_trapezoid_area(
            channel='x',
            system=system,
            area=-metadata.post_echo_area,
            grad_start=0.0,
            grad_end=0.0,
        )

    # Register prewinder and rewinder events
    block.gx_read = block.gx
    block.gx_prewind = gx_prewind
    block.gx_rewind = gx_rewind
    block.__dict__.pop('gx', None)

    # Update metadata
    metadata.__dict__.pop('pre_readout_area', None)
    metadata.__dict__.pop('post_readout_area', None)

    return block, metadata


def spoiled_line_readout(
    system: pp.Opts,
    fov_m: float,
    npix: int,
    spoiling_angle_rad: float,
    spoiling_thickness_m: float | None = None,
    receive_bandwidth_Hz: float = 250e3,
    oversamp: float = 1.0,
    partial_fourier_factor: float = 1.0,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """
    Create spoiled line readout lobe.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits.
    fov_m : float
        Field of view along readout direction in ``[m]``.
    npix : int
        Image size along readout direction.
    spoiling_angle_rad : float
        Dephasing angle across given thickness in units of ``[rad]``.
    spoiling_thickness_m : float
        Thickness of spoiled slab in ``[m]``.
    receive_bandwidth_Hz : float, optional
        Receive bandwidth in ``[Hz]``.
        The default is ``250e3 Hz``.
    oversamp : float, optional
        Readout oversampling factor.
        The default is ``1.0``.
    partial_fourier_factor : float, optional
        Partial Fourier Factor for asymmeric echo.
        The default is ``1.0``.
    rampsamp : bool, optional
        If ``True``, sample points on trapezoid ramps.
        The default is ``False``.

    Returns
    -------
    readout : SimpleNamespace
        Object containing the following events:
            - gx_prewind : SimpleNamespace
                  Prewinder gradient event.
            - gx_spoil : SimpleNamespace
                  Spoiler gradient event.
            - gx_read : SimpleNamespace
                  Readout gradient event.
            - adc : SimpleNamespace
                  Accompanying ADC event
    metadata : SimpleNamespace
        Object containing the following additional info
            - system : pp.Opts
                  System parameters used to generate event.
            - encoding_size : int
                  Image size along readout direction.
            - encoding_center : int
                  ADC index corresponding to k-space center along readout direction.
            - encoding_center_time : float
                  Sampling time of k-space center along readout direction in ``[s]``.
            - readout_time : float
                  Total readout time in ``[s]``.

    """
    if spoiling_thickness_m is None:
        spoiling_thickness_m = fov_m / npix

    # General line readout without ramp-down ramp
    block, metadata = general_line_readout(
        system,
        fov_m,
        npix,
        receive_bandwidth_Hz,
        oversamp,
        partial_fourier_factor,
        right_ramp=False,
    )

    # Calculate prewinder
    gx_prewind, _, _ = pp.make_extended_trapezoid_area(
        channel='x',
        system=system,
        area=-metadata.pre_echo_area,
        grad_start=0.0,
        grad_end=0.0,
    )

    # Calculate target spoil area
    spoiling_area = pp.calc_spoil_area(spoiling_angle_rad, spoiling_thickness_m)

    # Compute residual area to achieve target area
    gx_spoil_area = spoiling_area - metadata.post_echo_area

    # Calculate spoiler
    if gx_spoil_area >= 0.0:
        gx_spoil, _, _ = pp.make_extended_trapezoid_area(
            channel='x',
            system=system,
            area=gx_spoil_area,
            grad_start=block.gx.waveform[-1],
            grad_end=0.0,
        )
    else:
        times = np.diff(block.gx.tt)
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            rise_time=times[0],
            flat_time=times[1],
            amplitude=block.gx.waveform[-1],
            delay=block.gx.delay,
        )
        _, gx_spoil = pp.split_waveform_at(
            gx_read, time_point=gx_read.delay + gx_read.rise_time + gx_read.flat_time
        )
        gx_spoil.delay = 0.0

    # Register prewinder and spoiler events
    block.gx_read = block.gx
    block.gx_prewind = gx_prewind
    block.gx_spoil = gx_spoil
    block.__dict__.pop('gx', None)

    # Update metadata
    metadata.__dict__.pop('pre_readout_area', None)
    metadata.__dict__.pop('post_readout_area', None)

    return block, metadata


def fse_line_readout(
    system: pp.Opts,
    fov_m: float,
    npix: int,
    spoiling_area: float,
    receive_bandwidth_Hz: float = 250e3,
    oversamp: float = 1.0,
    partial_fourier_factor: float = 1.0,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """
    Create fast spin echo line readout lobe.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits.
    fov_m : float
        Field of view along readout direction in ``[m]``.
    npix : int
        Image size along readout direction.
    spoiling_area : float
        Area of excitation spoiling gradient along readout axis in ``[Hz/m]``.
        It must at least match the area between echo and end of readout lobe.
    receive_bandwidth_Hz : float, optional
        Receive bandwidth in ``[Hz]``.
        The default is ``250e3 Hz``.
    oversamp : float, optional
        Readout oversampling factor.
        The default is ``1.0``.
    partial_fourier_factor : float, optional
        Partial Fourier Factor for asymmeric echo.
        The default is ``1.0``.
    rampsamp : bool, optional
        If ``True``, sample points on trapezoid ramps.
        The default is ``Fals``e.

    Returns
    -------
    readout : SimpleNamespace
        Object containing the following events:
            - gx_prewind : SimpleNamespace
                  Prewinder gradient event.
            - gx_rewind : SimpleNamespace
                  Rewinder gradient event.
            - gx_read : SimpleNamespace
                  Readout gradient event.
            - adc : SimpleNamespace
                  Accompanying ADC event
    metadata : SimpleNamespace
        Object containing the following additional info
            - system : pp.Opts
                  System parameters used to generate event.
            - encoding_size : int
                  Image size along readout direction.
            - encoding_center : int
                  ADC index corresponding to k-space center along readout direction.
            - encoding_center_time : float
                  Sampling time of k-space center along readout direction in ``[s]``.
            - readout_time : float
                  Total readout time in ``[s]``.

    """
    # General line readout without ramps
    block, metadata = general_line_readout(
        system,
        fov_m,
        npix,
        receive_bandwidth_Hz,
        oversamp,
        partial_fourier_factor,
        left_ramp=False,
        right_ramp=False,
    )

    # Get target area for prewinder
    gx_prewind_area = -metadata.pre_echo_area + spoiling_area
    gx_rewind_area = spoiling_area - metadata.post_echo_area

    if gx_prewind_area <= 0.0 or gx_rewind_area <= 0.0:
        raise ValueError('Increase spoil area!')

    # Calculate prewinder
    gx_prewind, _, _ = pp.make_extended_trapezoid_area(
        channel='x',
        system=system,
        area=gx_prewind_area,
        grad_start=0.0,
        grad_end=block.gx.waveform[0],
    )

    # Calculate rewinder
    gx_rewind, _, _ = pp.make_extended_trapezoid_area(
        channel='x',
        system=system,
        area=gx_rewind_area,
        grad_start=block.gx.waveform[-1],
        grad_end=0.0,
    )

    # Register prewinder and rewinder events
    block.gx_read = block.gx
    block.gx_prewind = gx_prewind
    block.gx_rewind = gx_rewind
    block.__dict__.pop('gx', None)

    # Update metadata
    metadata.__dict__.pop('pre_readout_area', None)
    metadata.__dict__.pop('post_readout_area', None)

    return block, metadata
