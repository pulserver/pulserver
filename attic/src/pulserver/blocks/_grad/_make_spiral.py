"""Spiral gradient design routine."""

__all__ = ["make_spiral"]

from types import SimpleNamespace

import numpy as np

import pypulseq as pp
import pulpy.grad as pg

fufa = 0.9

def make_spiral(
    fov: float,
    npix: int,
    narms: int = 1,
    max_grad: float = 0,
    max_slew: float = 0,
    grad_raster_time: float = 0,
    system: pp.Opts = None,
    fs_npix: int = None,
    trans_dur: float = 0.5,
    spiral_type: str = "outward",
    mirror: bool = False,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """
    Create a spiral gradient event.

    See Also
    --------
    - `pypulseq.Sequence.sequence.Sequence.add_block()`
    - `pypulseq.make_arbitrary_grad.make_arbitrary_grad()`

    Parameters
    ----------
    fov : float
        Field of View in ``[mm]``.
    npix : int
        Image matrix size.
    narms : int, optional
        Number of interleaves. The default is ``1``.
    max_grad : float, optional
        Maximum gradient strength in ``[Hz/m]``.
        The default is ``0`` (use pp.Opts.default).
    max_slew : float, optional
        Maximum gradient slew-rate in ``[Hz/m/s]``.
        The default is ``0`` (use pp.Opts.default).
    grad_raster_time : float, optional
        Gradient raster time in ``[s]``. The default is ``10e-6``.
    system : pp.Opts, optional
        System limits. The default is ``None``.
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

    Returns
    -------
    grad : SimpleNamespace
        Gradient event with spiral waveform.

    """
    # system defaults
    if system is None:
        system = pp.Opts.default

    if max_grad <= 0:
        max_grad = system.max_grad

    if max_slew <= 0:
        max_slew = system.max_slew

    if grad_raster_time <= 0:
        grad_raster_time = system.grad_raster_time

    # adjust number of interleaves
    if spiral_type == "in-out" or spiral_type == "out-in":
        narms *= 2

    # convert to puply units
    fov /= 10.0  # [mm] -> [cm]
    max_grad = (
        pp.convert.convert(
            max_grad, from_unit="Hz/m", to_unit="mT/m", gamma=system.gamma
        )
        / 10
    )
    max_slew = (
        pp.convert.convert(
            max_slew, from_unit="Hz/m/s", to_unit="T/m/s", gamma=system.gamma
        )
        * 100
    )

    # calculate resolution
    res = fov / npix

    # constant density case
    if fs_npix is None:
        g0, k, _, _, dens = pg.spiral_varden(
            fov,
            res,
            grad_raster_time,
            fufa * max_slew,
            fufa * max_grad,
            0.0,
            0.0,
            narms,
        )
        g, _, _, _, _ = pg.spiral_varden(
            fov,
            res,
            grad_raster_time,
            fufa * max_slew,
            fufa * max_grad,
            0.0,
            0.0,
            narms,
            rewinder=True,
        )
    else:
        fres = fov / fs_npix
        gf, _, _, _, _ = pg.spiral_varden(
            fov, fres, grad_raster_time, fufa * max_slew, fufa * max_grad, 0.0, 0.0, 1
        )

        # number of points of fully sampled region
        densamp = gf.shape[0]
        denstrans = int(np.ceil(trans_dur * densamp))

        # actual calculation
        g0, k, _, _, dens = pg.spiral_varden(
            fov,
            res,
            grad_raster_time,
            fufa * max_slew,
            fufa * max_grad,
            densamp,
            denstrans,
            narms,
        )
        g, _, _, _, _ = pg.spiral_varden(
            fov,
            res,
            grad_raster_time,
            fufa * max_slew,
            fufa * max_grad,
            densamp,
            denstrans,
            narms,
            rewinder=True,
        )

    # compute adc points
    if spiral_type == "inward":
        npre, npost = len(g) - len(g0), 0
    else:
        npre, npost = 0, len(g) - len(g0)

    # handle spiral variant
    if spiral_type == "inward":
        g = np.flip(g, axis=0)
        k = np.flip(k, axis=0)
    if mirror:
        g = -g
        k = -k
        
    if spiral_type == "inward":
        g = np.pad(g, ((0, 16), (0, 0)))
        npost += 16
    else:
        g = np.pad(g, ((16, 0), (0, 0)))
        npre += 16
        
    # convert gradient units
    g = g.T
    g = pp.convert.convert(10 * g, from_unit="mT/m", gamma=system.gamma)  # [Hz / m]

    # normalize trajectory
    k = k.T
    kmax = ((k**2).sum(axis=0) ** 0.5).max()
    k = k / kmax / 2
    
    # make arbitrary
    gx = pp.make_arbitrary_grad("x", g[0], system=system)
    gy = pp.make_arbitrary_grad("y", g[1], system=system)

    # make header
    hdr = SimpleNamespace(traj=k, idx=(npre, npost))

    return gx, gy, hdr
