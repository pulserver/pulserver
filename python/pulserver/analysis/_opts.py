"""Vendor-neutral Opts extension for pypulseq.

This module defines :class:`Opts`, a ``pypulseq.Opts`` subclass that adds
hardware timing metadata, generic PNS model fields, and mechanical
resonance forbidden-band helpers. Vendor-specific coil tables and factory
helpers (e.g. GE coil models) live in the vendor extension packages
(see ``pulserver.gehc``).
"""

from __future__ import annotations

__all__ = ['Opts']

from collections.abc import Sequence

import pypulseq as pp


class Opts(pp.Opts):
    """``pypulseq.Opts`` subclass with PNS and mechanical resonance metadata.

    Parameters
    ----------
    gamma : float, optional
        Gyromagnetic ratio in Hz/T.
        If not provided, defaults to 42.576 MHz/T for proton imaging.
    B0 : float, optional
        Main magnetic field in tesla.
        If not provided, defaults to 3.0 T.
    max_grad : float, optional
        Maximum gradient amplitude in mT/m.
    max_slew : float, optional
        Maximum slew rate in T/m/s.
    b1_max_uT : float, optional
        RF peak limit in microtesla.
        If not provided, defaults to 20.0 uT.
    psd_rf_wait_s : float, default 0.0
        Extra RF wait time in seconds.
    psd_grd_wait_s : float, default 0.0
        Extra gradient wait time in seconds.
    chronaxie_us : float, optional
        PNS chronaxie in microseconds.
    rheobase : float, optional
        PNS rheobase parameter.
    alpha : float, optional
        PNS model alpha parameter.
    forbidden_bands : sequence of tuple, optional
        Forbidden bands in public units:
        ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m)``.
    rf_raster_time : float, default 2e-6
        RF raster time in seconds.
    grad_raster_time : float, default 4e-6
        Gradient raster time in seconds.
    adc_raster_time : float, default 2e-6
        ADC raster time in seconds.
    block_duration_raster : float, default 4e-6
        Block duration raster time in seconds.
    rf_dead_time : float, default 72e-6
        RF dead time in seconds.
    rf_ringdown_time : float, default 56e-6
        RF ringdown time in seconds.
    adc_dead_time : float, default 40e-6
        ADC dead time in seconds.
    adc_ringdown_time : float, default 0.0
        ADC ringdown time in seconds.
    segment_dead_time : float, default 12e-6
        Segment dead time in seconds.
    segment_ringdown_time : float, default 105e-6
        Segment ringdown time in seconds.

    Notes
    -----
    Public forbidden-band units are ``(Hz, mT/m)``. Internal safety APIs
    expect ``(Hz, Hz/m)``; use :meth:`forbidden_bands_hz_per_m`.
    """

    def __init__(
        self,
        *,
        gamma: float | None = None,
        B0: float | None = None,
        max_grad: float | None = None,
        max_slew: float | None = None,
        b1_max_uT: float = 20.0,
        psd_rf_wait_s: float = 0.0,
        psd_grd_wait_s: float = 0.0,
        chronaxie_us: float | None = None,
        rheobase: float | None = None,
        alpha: float | None = None,
        forbidden_bands: (
            Sequence[tuple[float, float, float] | tuple[float, float, float, str]]
            | None
        ) = None,
        rf_raster_time: float = 2e-6,
        grad_raster_time: float = 4e-6,
        adc_raster_time: float = 2e-6,
        block_duration_raster: float = 4e-6,
        rf_dead_time: float = 72e-6,
        rf_ringdown_time: float = 56e-6,
        adc_dead_time: float = 40e-6,
        adc_ringdown_time: float = 0.0,
        segment_dead_time: float = 12e-6,
        segment_ringdown_time: float = 105e-6,
    ):
        super().__init__(
            gamma=gamma,
            B0=B0,
            max_grad=max_grad,
            grad_unit='mT/m',
            max_slew=max_slew,
            slew_unit='T/m/s',
            rf_raster_time=rf_raster_time,
            grad_raster_time=grad_raster_time,
            adc_raster_time=adc_raster_time,
            block_duration_raster=block_duration_raster,
            rf_dead_time=rf_dead_time,
            rf_ringdown_time=rf_ringdown_time,
            adc_dead_time=adc_dead_time,
        )

        self.psd_rf_wait_s = float(psd_rf_wait_s)
        self.psd_grd_wait_s = float(psd_grd_wait_s)
        self.b1_max_uT = None if b1_max_uT is None else float(b1_max_uT)
        self.chronaxie_us = None if chronaxie_us is None else float(chronaxie_us)
        self.rheobase = None if rheobase is None else float(rheobase)
        self.alpha = None if alpha is None else float(alpha)

        self.adc_ringdown_time = float(adc_ringdown_time)
        self.segment_dead_time = float(segment_dead_time)
        self.segment_ringdown_time = float(segment_ringdown_time)

        self._forbidden_bands_mT_per_m: list[tuple[float, float, float, str | None]] = (
            []
        )
        if forbidden_bands is not None:
            self.set_forbidden_bands(forbidden_bands)

        if self.b1_max_uT is not None and abs(self.b1_max_uT) >= 100.0:
            raise ValueError('b1_max_uT appears too large; expected microtesla scale')

        if self.alpha is not None and self.alpha <= 0.0:
            raise ValueError('alpha must be > 0')

    def set_forbidden_bands(
        self,
        forbidden_bands: Sequence[
            tuple[float, float, float] | tuple[float, float, float, str]
        ],
    ) -> None:
        """Set forbidden bands in public units.

        Parameters
        ----------
        forbidden_bands : sequence of tuple
            Bands as ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m)`` or
            ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m, channel)`` where
            channel is one of ``'gx'``, ``'gy'``, ``'gz'``.

        Raises
        ------
        ValueError
            If bounds are invalid.
        """
        bands: list[tuple[float, float, float, str | None]] = []
        for band in forbidden_bands:
            if len(band) not in (3, 4):
                raise ValueError(
                    'Forbidden bands must be (fmin, fmax, amax) or (fmin, fmax, amax, channel)'
                )

            fmin = float(band[0])
            fmax = float(band[1])
            amax = float(band[2])
            channel: str | None = None
            if len(band) == 4:
                channel = str(band[3]).lower().strip()
                if channel not in ('gx', 'gy', 'gz'):
                    raise ValueError(
                        "Forbidden-band channel must be one of 'gx', 'gy', 'gz'"
                    )

            if fmin < 0.0 or fmax <= 0.0 or fmin >= fmax:
                raise ValueError(
                    'Forbidden-band frequencies must satisfy 0 <= fmin < fmax'
                )
            if amax < 0.0:
                raise ValueError('Forbidden-band max amplitude must be >= 0')
            bands.append((fmin, fmax, amax, channel))
        self._forbidden_bands_mT_per_m = bands

    @property
    def forbidden_bands(
        self,
    ) -> list[tuple[float, float, float] | tuple[float, float, float, str]]:
        """Forbidden bands in public units.

        Returns
        -------
        list[tuple]
            Bands as ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m)`` or
            ``(..., channel)`` when channel metadata is available.
        """
        out: list[tuple[float, float, float] | tuple[float, float, float, str]] = []
        for fmin, fmax, amax, channel in self._forbidden_bands_mT_per_m:
            if channel is None:
                out.append((fmin, fmax, amax))
            else:
                out.append((fmin, fmax, amax, channel))
        return out

    def forbidden_bands_hz_per_m(
        self, *, include_channel: bool = False
    ) -> list[tuple[float, float, float] | tuple[float, float, float, str]]:
        """Return forbidden bands converted to backend units.

        Returns
        -------
        list[tuple]
            Bands as ``(freq_min_hz, freq_max_hz, max_amplitude_hz_per_m)``.
            If ``include_channel=True``, channel-tagged entries are returned as
            ``(..., channel)`` when available.
        """
        gamma_hz_per_t = float(self.gamma)
        out: list[tuple[float, float, float] | tuple[float, float, float, str]] = []
        for fmin, fmax, amp_mT_per_m, channel in self._forbidden_bands_mT_per_m:
            amp_hz_per_m = amp_mT_per_m * 1e-3 * gamma_hz_per_t
            if include_channel and channel is not None:
                out.append((fmin, fmax, amp_hz_per_m, channel))
            else:
                out.append((fmin, fmax, amp_hz_per_m))
        return out

    def default_stim_threshold(self) -> float | None:
        """Return default PNS stimulation threshold in Hz/m/s.

        Returns
        -------
        float or None
            ``rheobase / alpha`` when both values are valid, else ``None``.
        """
        if self.rheobase is None or self.alpha is None:
            return None
        if self.alpha <= 0.0:
            return None
        return float(self.rheobase) / float(self.alpha)
