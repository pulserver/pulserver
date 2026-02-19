"""Plotting sub-routines."""

__all__ = ["display_pulse"]

import numpy as np
import matplotlib.pyplot as plt

from pulpy import sim

gamma_bar = 42.575 * 1e6  # MHz / T -> Hz / T
gamma = 2 * np.pi * gamma_bar  # rad / T / us -> rad / T / s
gamma = gamma * 1e-3  # rad / T / s -> rad / mT / s


def display_pulse(raster_time, rf, grad=None, dz=None, df=None):
    """
    Plot the RF pulse along with optional gradient and spatial/frequency ranges.

    Parameters
    ----------
    raster_time : float, optional
        Gradient and RF raster times in [s].
    rf : numpy.ndarray
        Time envelope of the RF pulse in [uT].
    grad : numpy.ndarray, optional
        Slice/slab selection waveform in [mT/m], by default None.
    dz : float, optional
        Spatial range to be plotted in [m], by default None.
    df : float, optional
        Frequency range to be plotted in [Hz], by default None.

    """
    # dimensions
    # rf -> uT
    # grad -> mT / m
    # gdt -> us
    # dz -> mm (if grad), Hz (if nograd)
    # df -> Hz
    raster_time *= 1e6
    dz *= 1e3

    # save stuff
    rf0 = rf.copy()
    rf = rf.copy()

    # get time vector
    t = np.arange(rf.shape[0]) * raster_time * 1e-6  # s

    # gradient
    if grad is None:
        nograd = True
        grad = 2 * np.pi * raster_time * 1e-6 * np.ones(rf.shape[0], dtype=float)  # rad
    else:
        nograd = False
        grad0 = grad.copy()
        grad = grad.copy()
        grad = gamma * raster_time * 1e-6 * grad  # mT / m -> rad / m
        grad = grad * 1e-3  # rad / mm

    # rf
    rf *= 1e-3  # uT -> mT
    rf *= gamma  # rad / s
    rf = rf * raster_time * 1e-6  # rad

    if nograd:
        assert (
            df is not None
        ), "Please provide frequency range for non spatially selective pulses."
    else:
        assert (
            dz is not None
        ), "Please provide spatial range for spatially selective pulses."

    # get axis
    if nograd:
        z = np.linspace(-df, df, 200)
        f = np.zeros(1)
    else:
        z = np.linspace(-dz, dz, 200)
        if df is None:
            f0 = np.zeros(1)
        else:
            f0 = np.linspace(-df, df, 100)  # Hz
        f = 2 * np.pi * f0 * raster_time * 1e-6  # rad

    # units
    # rf -> rad
    # grad -> rad / mm
    # z -> mm
    # f -> rad

    # simulate pulse
    a, b = sim.abrm_hp(rf, grad, z, f[:, None])
    a, b = a.T, b.T

    # get magnetization
    mxy = 2 * np.multiply(np.conj(a), b)
    mz = 1 - 2 * np.conj(b) * b

    if df is None:
        mxy = mxy[:, -1]
        mxy = np.ascontiguousarray(mxy)
        mz = mz[:, -1]
        mz = np.ascontiguousarray(mz)

    # spatial and frequency profile
    if nograd:
        pf = np.abs(mxy)
        pfz = mz.real
        f0 = z
    else:
        if df is None:
            pz = np.abs(mxy)
            pzz = mz.real
        else:
            pz = np.abs(mxy[:, 50])
            pf = np.abs(mxy[100])
            pzz = mz[:, 50].real
            pfz = mz[100].real

    # actual plotting
    if nograd:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(t * 1e3, rf0.real), plt.plot(t * 1e3, rf0.imag), plt.xlabel(
            "time [ms]"
        ), plt.ylabel("RF [uT]")
        plt.subplot(1, 2, 2)
        plt.plot(f0, pf), plt.xlabel("Frequency [Hz]"), plt.ylabel("M [a.u.]")
        plt.plot(f0, pfz), plt.legend(["Mxy", "Mz"])
        plt.tight_layout()
        plt.show()
    else:
        if df is None:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.plot(t * 1e3, rf0.real), plt.plot(t * 1e3, rf0.imag), plt.xlabel(
                "time [ms]"
            ), plt.ylabel("RF [uT]")
            plt.subplot(1, 3, 2)
            plt.plot(t * 1e3, grad0), plt.xlabel("time [ms]"), plt.ylabel("grad [mT/m]")
            plt.subplot(1, 3, 3)
            plt.plot(z, pz), plt.xlabel("Slice Position [mm]"), plt.ylabel("M [a.u.]")
            plt.plot(z, pzz), plt.legend(["Mxy", "Mz"])
            plt.tight_layout()
            plt.show()
        else:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.plot(t * 1e3, rf0.real), plt.plot(t * 1e3, rf0.imag), plt.xlabel(
                "time [ms]"
            ), plt.ylabel("RF [uT]")
            plt.subplot(2, 2, 3)
            plt.plot(t * 1e3, grad0), plt.xlabel("time [ms]"), plt.ylabel("Gz [mT/m]")
            plt.subplot(2, 2, 2)
            plt.plot(z, pz), plt.xlabel("Slice Position [mm]"), plt.ylabel("M [a.u.]")
            plt.plot(z, pzz), plt.legend(["Mxy", "Mz"])
            plt.subplot(2, 2, 4)
            plt.plot(f0, pf), plt.xlabel("Frequency [Hz]"), plt.ylabel("M [a.u.]")
            plt.plot(f0, pfz), plt.legend(["Mxy", "Mz"])
            plt.tight_layout()
            plt.show()

        return mxy, mz.real
