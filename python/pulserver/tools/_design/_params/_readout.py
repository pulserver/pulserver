"""Compute readout parameters"""

__all__ = ['calc_kspace_readout_params', 'calc_readout_duration']

import numpy as np


def calc_kspace_readout_params(
    fov: float,
    npix: int,
    receiver_bandwidth: float,
    oversamp: float = 2.0,
    partial_fourier_factor: float = 1.0,
    adc_raster_time: float = 1e-7,
    grad_raster_time: float = 1e-5,
    adc_samples_divisor=1,
    time_unit: float | None = None,
) -> tuple[float, float, int]:
    """
    Compute readout k-space width, raster-aligned readout duration, and number of samples.

    This wrapper computes:
      - the k-space width along the readout (kr_area = npix / fov),
      - a readout duration aligned to both gradient and ADC rasters with an actual bandwidth
        that does not exceed the requested receiver_bandwidth (i.e., dwell is rounded up),
      - the number of ADC samples after applying readout oversampling.

    Parameters
    ----------
    fov : float
        Field of view in meters along the readout direction.
    npix : int
        Nominal number of image pixels along readout (before oversampling).
    receiver_bandwidth : float
        Target receive bandwidth in Hz. The actual bandwidth returned will be
        less than or equal to this value due to raster alignment (dwell rounded up).
    oversamp : float, optional
        Readout oversampling factor (e.g., 2.0 keeps the central half of k-space),
        by default 2.0.
    partial_fourier_factor : float, optional
        Partial Fourier reduction factors for number of samples. The default is 1.0.
    adc_raster_time : float, optional
        ADC raster (smallest dwell quantum) in seconds, by default 1e-7.
    grad_raster_time : float, optional
        Gradient raster time in seconds, by default 1e-5.
    adc_samples_divisor : int, optional
        Force final number of samples to be an integer multiple of this value.
        The default is 1.
    time_unit : float or None, optional
        Discretization unit (seconds) for integer tick math inside the alignment routine.
        If None (default), chooses 1e-9 when either raster < 1e-6, else 1e-6.

    Returns
    -------
    kr_area : float
        Readout k-space width in 1/m (equals npix / fov).
        Note: This is a k-space extent, not a gradient moment; converting to gradient
        area requires gamma and the time integral of the gradient waveform.
    t_read : float
        Raster-aligned readout duration in seconds; a multiple of both rasters and
        equal to num_samples * dwell_time_actual from the alignment routine.
    num_samples : int
        Number of ADC samples used to cover the oversampled readout.

    Notes
    -----
    - The alignment routine rounds the dwell to the next feasible multiple of the ADC raster
      that also makes the total readout a multiple of the gradient raster, biasing toward
      smaller bandwidth (bandwidth_actual <= receiver_bandwidth).
    - If you need tighter control on the actual bandwidth, adjust `oversamp` or rasters,
      or evaluate both ceil/floor candidates in a custom policy.

    Examples
    --------
    >>> kr, t_read, n = calc_kspace_readout_params(
    ...     fov=0.24, npix=256, receiver_bandwidth=200e3,
    ...     oversamp=2.0, adc_raster_time=100e-9, grad_raster_time=10e-6
    ... )
    >>> kr  # ~ npix / fov
    1066.6666666666667
    >>> n   # oversampled samples
    512

    """
    if fov <= 0 or npix <= 0 or receiver_bandwidth <= 0:
        raise ValueError('fov, npix, and receiver_bandwidth must be positive.')
    if oversamp <= 0:
        raise ValueError('oversamp must be positive.')
    if adc_raster_time <= 0 or grad_raster_time <= 0:
        raise ValueError('Raster times must be positive.')
    # Compute actual number of pixels
    npix_eff = int(np.ceil(npix * partial_fourier_factor))

    # k-space step and width (extent along readout)
    delta_k = 1.0 / fov
    kr_area = npix_eff * delta_k  # [1/m]

    # Oversampled sample count
    num_samples = int(np.ceil(oversamp * npix_eff))

    # Make sure it is integer multiple of 4
    num_samples = int(adc_samples_divisor * np.ceil(num_samples / adc_samples_divisor))

    # Choose integer tick unit for exact divisibility math
    if time_unit is None:
        time_unit = 1e-9 if min(adc_raster_time, grad_raster_time) < 1e-6 else 1e-6

    # Readout duration that honors both rasters and does not exceed the target BW
    t_read, _, _ = calc_readout_duration(
        grad_raster_time=grad_raster_time,
        adc_raster_time=adc_raster_time,
        target_bw=receiver_bandwidth,
        num_samples=num_samples,
        time_unit=time_unit,
    )

    return kr_area, t_read, num_samples


def calc_readout_duration(
    grad_raster_time: float,
    adc_raster_time: float,
    target_bw: float,
    num_samples: int,
    time_unit: float = 1e-9,
    min_q: int = 1,
) -> tuple[float, float, float]:
    """
    Compute a raster-aligned ADC readout duration, biasing toward smaller bandwidth.

    The resulting readout duration T satisfies:
    - T is a multiple of both grad_raster_time and adc_raster_time
    - T = num_samples * dwell_time_actual
    - bandwidth_actual = 1 / dwell_time_actual <= target_bw (i.e., we round dwell up)

    Parameters
    ----------
    grad_raster_time : float
        Gradient raster time in seconds (e.g., 10e-6).
    adc_raster_time : float
        ADC raster (smallest dwell quantum) in seconds (e.g., 100e-9).
    target_bw : float
        Target receive bandwidth in Hz. The actual bandwidth will be
        less than or equal to this value (i.e., dwell time rounded up).
    num_samples : int
        Number of ADC samples (N >= 1).
    time_unit : float, optional
        Discretization unit (seconds) for integer tick arithmetic (default 1e-9).
        Use a unit that exactly represents your rasters (e.g., 1 ns for 100 ns and 10 us).
    min_q : int, optional
        Minimum integer multiple of the ADC raster for dwell (default 1).

    Returns
    -------
    T_readout : float
        Readout duration in seconds; multiple of grad_raster_time and adc_raster_time.
    dwell_time_actual : float
        Actual ADC dwell time (seconds), a multiple of adc_raster_time.
    bandwidth_actual : float
        Actual receive bandwidth in Hz (= 1 / dwell_time_actual), guaranteed
        to be <= target_bw.

    Notes
    -----
    Let:
      - a = adc_raster_time, g = grad_raster_time, N = num_samples.
      - dwell must be d = q * a for some integer q.
      - T = N * d must also be a multiple of g.

    In integer ticks (using `time_unit`):
      - a_ticks = round(a / time_unit), g_ticks = round(g / time_unit).
      - The constraint on q is: q must be a multiple of
            K = g_ticks / gcd(g_ticks, N * a_ticks).
      - To enforce bandwidth_actual <= target_bw (i.e., larger dwell),
        choose the smallest q that is a multiple of K and satisfies
            q >= (1/target_bw) / a.

    Examples
    --------
    >>> T, d, bw = compute_readout_duration(10e-6, 100e-9, 200e3, 1024)
    >>> T  # multiple of 10 us, equals 1024 * d
    0.01024
    >>> bw <= 200e3
    True

    """
    # Validate
    if grad_raster_time <= 0 or adc_raster_time <= 0 or target_bw <= 0:
        raise ValueError('Raster times and target_bw must be positive.')
    if not (isinstance(num_samples, (int, np.integer)) and num_samples >= 1):
        raise ValueError('num_samples must be a positive integer.')
    if time_unit <= 0:
        raise ValueError('time_unit must be positive.')
    if min_q < 1:
        raise ValueError('min_q must be >= 1.')

    a = float(adc_raster_time)
    g = float(grad_raster_time)
    N = int(num_samples)

    # Convert to integer ticks for exact divisibility checks
    a_ticks = round(a / time_unit)
    g_ticks = round(g / time_unit)
    if a_ticks <= 0 or g_ticks <= 0:
        raise ValueError(
            'time_unit too large relative to raster times; decrease time_unit.'
        )

    # Required multiple K for q so that N*q*a is multiple of g
    Na_ticks = N * a_ticks
    gcd_val = int(np.gcd(g_ticks, Na_ticks))
    K = g_ticks // gcd_val  # integer >= 1

    # Target dwell in units of 'a' (q can be fractional here)
    dt_target = 1.0 / target_bw
    q0_real = max(min_q, dt_target / a)

    # Bias toward smaller bandwidth => larger dwell => ceil to next multiple of K
    q = max(min_q, int(np.ceil(q0_real / K)) * K)

    # Compute actual dwell and readout
    dwell_time_actual = q * a
    T_readout = N * dwell_time_actual
    bandwidth_actual = 1.0 / dwell_time_actual

    # Sanity checks in tick domain (exact integers)
    T_ticks = N * q * a_ticks
    if T_ticks % g_ticks != 0:
        raise RuntimeError(
            'Computed T_readout is not a multiple of grad_raster_time (check time_unit).'
        )
    if (q * a_ticks) % a_ticks != 0:
        raise RuntimeError(
            'Computed dwell is not a multiple of adc_raster_time (unexpected).'
        )

    # Ensure bandwidth constraint direction
    if bandwidth_actual > target_bw + np.finfo(float).eps * target_bw:
        # Numerical safety: this shouldn't happen with ceil, but guard anyway
        q = q + K
        dwell_time_actual = q * a
        T_readout = N * dwell_time_actual
        bandwidth_actual = 1.0 / dwell_time_actual

    return T_readout, dwell_time_actual, bandwidth_actual
