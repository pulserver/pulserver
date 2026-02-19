"""Line readout creation subroutines."""

__all__ = ["make_line_readout"]


import numpy as np
import pypulseq as pp


def make_line_readout(
    system: pp.Opts,
    fov: float,
    npix: int,
    osf: float = 1.0,
    has_adc: bool = True,
    flyback: bool = False,
    ndim: int = 1,
) -> tuple[dict, dict] | tuple[dict, dict, dict]:
    """
    Prepare line readout for Cartesian, EPI or Radial imaging.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        Field of view in the readout direction in ``[mm]``.
    npix : int
        Matrix size in the readout direction.
    osf : float, optional
        Readout oversampling factor. The default is ``1.0``.
    has_adc : bool, optional
        If ``True``, ``read_block`` contains and ADC event.
        Otherwise, it does not (e.g., dummy shots for steady-state preparation).
        The default is ``True``.
    flyback : bool, optional
        If ``True``, design a flyback gradient. The default is ``False``.
    ndim : int, optional
        Number of k-space dimensions spanned by the readout.
        The default is ``1``.

    Returns
    -------
    read_block : dict
        Readout block dictionary with the following keys:

        * gx : SimpleNamespace
            Readout gradient event.
        * gy : SimpleNamespace
            Readout gradient event (only if ``ndim >= 2``).
        * gz : SimpleNamespace
            Readout gradient event (only if ``ndim == 3``).
        * adc : SimpleNamespace
            ADC event (only if ``has_adc`` is ``True``).

    phase_block : dict
        Readout prewinder / rewinder block with the following keys:

        * gx : SimpleNamespace
            Pre-/rewinder gradient event.
        * gy : SimpleNamespace
            Pre-/rewinder gradient event (only if ``ndim >= 2``).
        * gz : SimpleNamespace
            Pre-/rewinder gradient event (only if ``ndim == 3``).

    flyback_block : dict, optional
        Readout flyback block with the following keys:

        * gx : SimpleNamespace
            Flyback gradient event.
        * gy : SimpleNamespace
            Flyback gradient event (only if ``ndim >= 2``).
        * gz : SimpleNamespace
            Flyback gradient event (only if ``ndim == 3``).

        It is returned only if ``flyback`` is ``True``.

    Notes
    -----
    For Cartesian or EPI imaging, the ``ndim`` is equal to ``1``.
    For 2D and 3D Radial imaging, ``ndim`` should be set to ``2``
    and ``3``, respectively. The design routine will then put the maximum
    amplitude trapezoid event in ``gx, gy`` and ``gx, gy, gz``, respectively,
    to handle the rotation as amplitude scaling given by the elements
    of the rotation matrix. This trick would not work e.g., for spiral
    trajectories, where the base waveform for different axis are different.


    """
    # unit conversion (mm -> m)
    fov *= 1e-3

    # apply oversampling
    npix = int(np.ceil(osf * npix))

    # k space density
    dk = 1 / fov

    # get dwell
    dwell = system.adc_raster_time

    # calculate duration
    adc_duration = npix * dwell

    # frequency encoding gradients
    gread = pp.make_trapezoid(
        "x", flat_area=npix * dk, flat_time=adc_duration, system=system
    )

    # prepare prephaser/rephaser gradient lobe
    gphase = pp.make_trapezoid("x", area=-gread.area / 2, system=system)

    # flyback gradient
    if flyback:
        gflyback = pp.make_trapezoid("x", area=-gread.area, system=system)
    else:
        gflyback = None

    # adc event
    if has_adc:
        adc = pp.make_adc(
            num_samples=npix,
            duration=gread.flat_time,
            delay=gread.rise_time,
            system=system,
        )

    # prepare output
    if ndim == 1:
        read_block, phase_block, flyback_block = (
            {"gx": gread},
            {"gx": gphase},
            {"gx": gflyback},
        )
    if ndim == 2:
        read_block, phase_block, flyback_block = (
            {"gx": gread, "gy": gread},
            {"gx": gphase, "gy": gphase},
            {"gx": gflyback, "gy": gflyback},
        )
    if ndim == 3:
        read_block, phase_block, flyback_block = (
            {"gx": gread, "gy": gread, "gz": gread},
            {"gx": gphase, "gy": gphase, "gz": gphase},
            {"gx": gflyback, "gy": gflyback, "gz": gflyback},
        )

    if has_adc:
        read_block["adc"] = adc

    if flyback:
        return read_block, phase_block, flyback_block

    return read_block, phase_block
