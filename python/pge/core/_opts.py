"""GE-focused Opts extension for pypulseq.

This module defines :class:`Opts`, a GE-oriented subclass of
``pypulseq.Opts`` that adds vendor timing metadata, PNS model fields,
and mechanical resonance forbidden-band helpers.
"""

from __future__ import annotations

__all__ = ['GEOpts', 'Opts']

from collections.abc import Sequence

import pypulseq as pp

# coil -> (chronaxie_us, rheobase, alpha, gmax_g_per_cm, smax_g_per_cm_per_ms)
_COIL_PNS = {
    'xrmw': (360.0, 20.0, 0.324, 33.0, 120.0),
    'xrm': (334.0, 23.4, 0.333, 50.0, 200.0),
    'whole': (370.0, 23.7, 0.344, 23.0, 77.0),
    'zoom': (354.0, 29.1, 0.309, 40.0, 150.0),
    'hrmbuhp': (359.0, 26.5, 0.370, 100.0, 200.0),
    'hrmw': (642.4, 17.9, 0.310, 70.0, 200.0),
    'magnus': (611.0, 55.2, 0.324, 300.0, 750.0),
}

# ESP model aliases that do not have explicit PNS tuples in the provided table.
# We map to the closest available family defaults.
_COIL_PNS_ALIASES = {
    'vrmw': 'xrmw',
    'hrmb': 'hrmbuhp',
    'irmw': 'hrmw',
}

# coil -> tuple(axis_x_rows, axis_y_rows, axis_z_rows)
# row format = (esp_min_us, esp_max_us, max_amp_g_per_cm)
_COIL_ESP = {
    'xrmw': (
        [],
        [],
        [(330.0, 460.0, 0.0)],
    ),
    'vrmw': (
        [],
        [],
        [(330.0, 460.0, 0.0)],
    ),
    'xrm': (
        [(410.0, 510.0, 0.0)],
        [(410.0, 510.0, 0.0)],
        [(360.0, 440.0, 0.0)],
    ),
    'hrmw': (
        [(420.0, 459.0, 0.0), (816.0, 865.0, 1.6)],
        [(420.0, 459.0, 0.0), (816.0, 876.0, 1.6)],
        [],
    ),
    'hrmb': (
        [(393.0, 445.0, 0.0), (476.0, 528.0, 0.0), (952.0, 988.0, 0.0)],
        [(393.0, 445.0, 0.0), (476.0, 504.0, 0.0)],
        [(393.0, 445.0, 0.0), (476.0, 491.0, 0.0)],
    ),
    'hrmbuhp': (
        [(350.0, 445.0, 0.0), (481.0, 488.0, 0.0)],
        [(350.0, 445.0, 0.0), (481.0, 488.0, 0.0)],
        [(418.0, 426.0, 0.0)],
    ),
    'irmw': (
        [(827.0, 896.0, 0.0)],
        [(827.0, 896.0, 0.0)],
        [],
    ),
}


def _g_per_cm_to_mT_per_m(value: float) -> float:
    """Convert gradient amplitude from G/cm to mT/m.

    Parameters
    ----------
    value : float
        Gradient amplitude in gauss per centimeter.

    Returns
    -------
    float
        Gradient amplitude in millitesla per meter.
    """
    return float(value) * 10.0


def _g_per_cm_per_ms_to_T_per_m_per_s(value: float) -> float:
    """Convert slew rate from G/cm/ms to T/m/s.

    Parameters
    ----------
    value : float
        Slew rate in gauss per centimeter per millisecond.

    Returns
    -------
    float
        Slew rate in tesla per meter per second.
    """
    return float(value) * 10.0


def _esp_us_to_band_hz(esp_min_us: float, esp_max_us: float) -> tuple[float, float]:
    """Convert ESP bounds (microseconds) into frequency-band limits (Hz).

    Uses the relation :math:`f = 1 / (2 \\cdot \\mathrm{ESP})` and returns
    ordered limits ``(f_min, f_max)``.

    Parameters
    ----------
    esp_min_us : float
        Lower ESP bound in microseconds.
    esp_max_us : float
        Upper ESP bound in microseconds.

    Returns
    -------
    tuple[float, float]
        Frequency-band limits in hertz.

    Raises
    ------
    ValueError
        If either ESP value is non-positive.
    """
    if esp_min_us <= 0.0 or esp_max_us <= 0.0:
        raise ValueError('ESP values must be positive (microseconds)')
    f1 = 1.0 / (2.0 * esp_min_us * 1e-6)
    f2 = 1.0 / (2.0 * esp_max_us * 1e-6)
    return (min(f1, f2), max(f1, f2))


class Opts(pp.Opts):
    """GE-focused ``pypulseq.Opts`` subclass with PNS and mechanical resonance metadata.

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

    @classmethod
    def from_coil_model(
        cls,
        model_name: str,
        *,
        gamma: float | None = None,
        B0: float | None = None,
        b1_max_uT: float = 20.0,
        psd_rf_wait_s: float = 0.0,
        psd_grd_wait_s: float = 0.0,
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
    ) -> "Opts":
        """Create :class:`Opts` from a known GE coil model table.

        Coil-spec fields (max gradient/slew, PNS params, forbidden bands)
        are locked to the selected model and cannot be overridden here.

        Parameters
        ----------
        model_name : str
            Coil model identifier.
        gamma : float, optional
            Gyromagnetic ratio in Hz/T.
            If not provided, defaults to 42.576 MHz/T for proton imaging.
        B0 : float, optional
            Main magnetic field in tesla.
            If not provided, defaults to 3.0 T for known models.
        b1_max_uT : float, optional
            RF peak limit in microtesla.
            If not provided, defaults to 20.0 uT.
        psd_rf_wait_s : float, default 0.0
            Extra RF wait time in seconds.
        psd_grd_wait_s : float, default 0.0
            Extra gradient wait time in seconds.
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

        Returns
        -------
        Opts
            Initialised opts object for the selected coil model.

        Raises
        ------
        ValueError
            If *model_name* is not recognised.
        """
        model = model_name.lower().strip()

        pns_key = model
        if pns_key not in _COIL_PNS:
            pns_key = _COIL_PNS_ALIASES.get(model, '')
        if pns_key not in _COIL_PNS:
            known = sorted(
                set(_COIL_PNS.keys())
                | set(_COIL_ESP.keys())
                | set(_COIL_PNS_ALIASES.keys())
            )
            raise ValueError(
                f"Unknown coil model '{model_name}'. Known models: {known}"
            )

        chronaxie_us, rheobase, alpha, gmax_gpcm, smax_gpcm_ms = _COIL_PNS[pns_key]

        max_grad_mT_per_m = _g_per_cm_to_mT_per_m(gmax_gpcm)
        max_slew_T_per_m_per_s = _g_per_cm_per_ms_to_T_per_m_per_s(smax_gpcm_ms)
        forbidden_bands = cls._bands_from_esp_model(model)

        return cls(
            gamma=gamma,
            B0=B0,
            max_grad=max_grad_mT_per_m,
            max_slew=max_slew_T_per_m_per_s,
            b1_max_uT=b1_max_uT,
            psd_rf_wait_s=psd_rf_wait_s,
            psd_grd_wait_s=psd_grd_wait_s,
            chronaxie_us=chronaxie_us,
            rheobase=rheobase,
            alpha=alpha,
            forbidden_bands=forbidden_bands,
            rf_raster_time=rf_raster_time,
            grad_raster_time=grad_raster_time,
            adc_raster_time=adc_raster_time,
            block_duration_raster=block_duration_raster,
            rf_dead_time=rf_dead_time,
            rf_ringdown_time=rf_ringdown_time,
            adc_dead_time=adc_dead_time,
            adc_ringdown_time=adc_ringdown_time,
            segment_dead_time=segment_dead_time,
            segment_ringdown_time=segment_ringdown_time,
        )

    @staticmethod
    def _bands_from_esp_model(model: str) -> list[tuple[float, float, float, str]]:
        """Build public forbidden bands from ESP tables for a model.

        Parameters
        ----------
        model : str
            Coil model key.

        Returns
        -------
        list[tuple[float, float, float, str]]
            Bands in ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m, channel)``.
        """
        rows_by_axis = _COIL_ESP.get(model)
        if rows_by_axis is None:
            return []

        out: list[tuple[float, float, float, str]] = []
        axis_names = ('gx', 'gy', 'gz')
        for axis_idx, axis_rows in enumerate(rows_by_axis):
            for esp_min_us, esp_max_us, max_amp_gpcm in axis_rows:
                fmin, fmax = _esp_us_to_band_hz(float(esp_min_us), float(esp_max_us))
                out.append(
                    (
                        fmin,
                        fmax,
                        _g_per_cm_to_mT_per_m(float(max_amp_gpcm)),
                        axis_names[axis_idx],
                    )
                )
        return out

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


# Backward-compatible alias while exposing Opts in the public namespace.
GEOpts = Opts
