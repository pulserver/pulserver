"""Spiral readout creation subroutines."""

__all__ = ["make_spiral_readout"]

from types import SimpleNamespace


import pypulseq as pp


from .._grad._make_spiral import make_spiral


def make_spiral_readout(
    system: pp.Opts,
    fov: float,
    npix: int,
    narms: int = 1,
    fs_npix: int = None,
    trans_dur: float = 0.5,
    spiral_type: str = "outward",
    mirror: bool = False,
    has_adc: bool = True,
) -> tuple[dict, SimpleNamespace]:
    """
    Prepare spiral readout.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    fov : float
        Field of View in ``[mm]``.
    npix : int
        Matrix size.
    narms : int, optional
        Number of interleaves. The default is ``1``.
    fs_npix : int, optional
        Matrix size for fully-sampled region.
        The default is ``None`` (constant density spiral).
    trans_dur : float, optional
        Fractional duration of transition region for dual-density spiral,
        referred to the duration of fully sampled region, i.e.:

        * ``npts_transition = trans_dur * npts_fully_sampled``

        It must be ``>= 0.5``. The default is ``0.5``.
    spiral_type : str, optional
        Spiral variant. Can be:

        * ``"outward"``: center-to-periphery.
        * ``inward``: periphery-to-center.

        The default is ``outward``.
    mirror : bool, optional
        Mirror spiral. The default is ``False``.
    has_adc : bool, optional
        If ``True``, ``read_block`` contains and ADC event.
        Otherwise, it does not (e.g., dummy shots for steady-state preparation).
        The default is ``True``.

    Returns
    -------
    read_block : dict
        Readout block dictionary with the following keys:

        * gx : SimpleNamespace
            X-component of readout gradient event.
        * gy : SimpleNamespace
            Y-component of readout gradient event.
        * adc : SimpleNamespace
            ADC event (only if ``has_adc`` is ``True``).

    hdr : SimpleNamespace
        Spiral header containing

        * traj : np.ndarray
            K-space trajectory of shape ``(2, len(arm))``.
        * idx : tuple
            Indexes for ADC start (``idx[0]``) and stop (``idx[1]``).

    """
    # initialize spiral gradient
    gx_spiral, gy_spiral, hdr = make_spiral(
        fov, npix, narms, 0, 0, 0, system, fs_npix, trans_dur, spiral_type, mirror
    )

    # adc event
    if has_adc:
        num_samples = gx_spiral.waveform.shape[-1] - sum(hdr.idx)
        duration = system.grad_raster_time * num_samples
        delay = system.grad_raster_time * hdr.idx[0]

        adc = pp.make_adc(
            num_samples=num_samples,
            duration=duration,
            delay=delay,
            system=system,
        )

    # prepare output
    read_block = {"gx": gx_spiral, "gy": gy_spiral}

    if has_adc:
        read_block["adc"] = adc

    return read_block, hdr
