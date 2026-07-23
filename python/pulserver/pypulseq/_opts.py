"""Vendor-neutral Pulseq system defaults."""

from __future__ import annotations

from collections.abc import Sequence as _Sequence

import pypulseq as _pp


class Opts(_pp.Opts):
    """PyPulseq system limits with conservative vendor-neutral rasters.

    The hardware amplitude and slew defaults remain PyPulseq's.  Timing
    defaults deliberately use the coarsest common GE/Siemens values so a
    sequence designed without explicit scanner limits is not silently tied to
    either vendor.

    Beyond upstream's limits it also carries the hardware metadata the safety
    views need — the PNS nerve-model constants and the mechanical resonance
    forbidden bands — so :meth:`pulserver.pypulseq.Sequence.pns` and
    :meth:`~pulserver.pypulseq.Sequence.grad_spectrum` can pick them up from
    the system rather than being told twice. All of it is optional; a sequence
    that never looks at PNS never needs to set any of it. Vendor coil tables
    that fill these in live in the vendor extension packages (see
    ``pulserver.gehc``).

    Parameters
    ----------
    b1_max_uT : float, optional
        RF peak limit in microtesla.
    psd_rf_wait_s, psd_grd_wait_s : float, default 0.0
        Extra RF / gradient wait times in seconds.
    chronaxie_us : float, optional
        PNS chronaxie in microseconds.
    rheobase : float, optional
        PNS rheobase in Hz/m/s.
    alpha : float, optional
        PNS coil attenuation factor; the stimulation threshold is
        ``rheobase / alpha``.
    pns_hardware : optional
        SAFE hardware description, or the path to the Siemens ``.asc`` file
        holding one, in the form :meth:`pypulseq.Sequence.calculate_pns`
        takes. Only the SAFE nerve model reads it; the Irnich model uses the
        three constants above.
    forbidden_bands : sequence of tuple, optional
        Mechanical resonance bands as
        ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m[, channel])``,
        where channel is ``'gx'``, ``'gy'`` or ``'gz'``. Read them from a
        vendor table with :func:`pulserver.io.read_esp_bands` (GE) or
        :func:`pulserver.io.read_asc_bands` (Siemens).
    adc_ringdown_time, segment_dead_time, segment_ringdown_time : float
        Additional hardware timing metadata, in seconds.

    Notes
    -----
    Public forbidden-band units are ``(Hz, mT/m)``. The C safety backend
    expects ``(Hz, Hz/m)``; use :meth:`forbidden_bands_hz_per_m`.
    """

    def __init__(
        self,
        adc_dead_time: float | None = 0.0,
        adc_raster_time: float | None = 2e-6,
        block_duration_raster: float | None = 10e-6,
        gamma: float | None = None,
        grad_raster_time: float | None = 10e-6,
        grad_unit: str = "Hz/m",
        max_grad: float | None = None,
        max_slew: float | None = None,
        rf_dead_time: float | None = 0.0,
        rf_raster_time: float | None = 2e-6,
        rf_ringdown_time: float | None = 0.0,
        adc_samples_limit: int | None = None,
        adc_samples_divisor: int | None = None,
        rise_time: float | None = None,
        slew_unit: str = "Hz/m/s",
        B0: float | None = None,
        *,
        b1_max_uT: float | None = None,
        psd_rf_wait_s: float = 0.0,
        psd_grd_wait_s: float = 0.0,
        chronaxie_us: float | None = None,
        rheobase: float | None = None,
        alpha: float | None = None,
        pns_hardware=None,
        forbidden_bands: _Sequence | None = None,
        adc_ringdown_time: float = 0.0,
        segment_dead_time: float = 0.0,
        segment_ringdown_time: float = 0.0,
    ) -> None:
        super().__init__(
            adc_dead_time=adc_dead_time,
            adc_raster_time=adc_raster_time,
            block_duration_raster=block_duration_raster,
            gamma=gamma,
            grad_raster_time=grad_raster_time,
            grad_unit=grad_unit,
            max_grad=max_grad,
            max_slew=max_slew,
            rf_dead_time=rf_dead_time,
            rf_raster_time=rf_raster_time,
            rf_ringdown_time=rf_ringdown_time,
            adc_samples_limit=adc_samples_limit,
            adc_samples_divisor=adc_samples_divisor,
            rise_time=rise_time,
            slew_unit=slew_unit,
            B0=B0,
        )

        self.b1_max_uT = None if b1_max_uT is None else float(b1_max_uT)
        self.psd_rf_wait_s = float(psd_rf_wait_s)
        self.psd_grd_wait_s = float(psd_grd_wait_s)
        self.chronaxie_us = None if chronaxie_us is None else float(chronaxie_us)
        self.rheobase = None if rheobase is None else float(rheobase)
        self.alpha = None if alpha is None else float(alpha)
        self.pns_hardware = pns_hardware
        self.adc_ringdown_time = float(adc_ringdown_time)
        self.segment_dead_time = float(segment_dead_time)
        self.segment_ringdown_time = float(segment_ringdown_time)

        self._forbidden_bands: list[tuple[float, float, float, str | None]] = []
        if forbidden_bands is not None:
            self.set_forbidden_bands(forbidden_bands)

        if self.b1_max_uT is not None and abs(self.b1_max_uT) >= 100.0:
            raise ValueError("b1_max_uT appears too large; expected microtesla scale")
        if self.alpha is not None and self.alpha <= 0.0:
            raise ValueError("alpha must be > 0")

    def set_as_default(self) -> None:
        """Use this object as the default for Pulserver RF factories."""
        type(self).default = self

    @classmethod
    def reset_default(cls) -> None:
        """Restore Pulserver's vendor-neutral defaults."""
        cls.default = cls()

    # ── mechanical resonance bands ───────────────────────────────────

    def set_forbidden_bands(self, forbidden_bands: _Sequence) -> None:
        """Set the mechanical resonance bands, in ``(Hz, mT/m)``.

        Parameters
        ----------
        forbidden_bands : sequence of tuple
            ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m)``, optionally
            with a trailing ``'gx'`` / ``'gy'`` / ``'gz'`` channel tag.

        Raises
        ------
        ValueError
            If a band's bounds or amplitude limit are invalid.
        """
        bands: list[tuple[float, float, float, str | None]] = []
        for band in forbidden_bands:
            if len(band) not in (3, 4):
                raise ValueError("Bands must be (fmin, fmax, amax) or (fmin, fmax, amax, channel)")

            fmin, fmax, amax = float(band[0]), float(band[1]), float(band[2])
            channel: str | None = None
            if len(band) == 4:
                channel = str(band[3]).lower().strip()
                if channel not in ("gx", "gy", "gz"):
                    raise ValueError("Band channel must be one of 'gx', 'gy', 'gz'")

            if fmin < 0.0 or fmax <= 0.0 or fmin >= fmax:
                raise ValueError("Band frequencies must satisfy 0 <= fmin < fmax")
            if amax < 0.0:
                raise ValueError("Band max amplitude must be >= 0")
            bands.append((fmin, fmax, amax, channel))
        self._forbidden_bands = bands

    @property
    def forbidden_bands(self) -> list[tuple]:
        """Mechanical resonance bands in public units ``(Hz, mT/m)``."""
        return [
            (fmin, fmax, amax) if channel is None else (fmin, fmax, amax, channel)
            for fmin, fmax, amax, channel in self._forbidden_bands
        ]

    def forbidden_bands_hz_per_m(self, *, include_channel: bool = False) -> list[tuple]:
        """Mechanical resonance bands in backend units ``(Hz, Hz/m)``.

        Parameters
        ----------
        include_channel : bool, default False
            Keep the channel tag on bands that carry one.
        """
        gamma = float(self.gamma)
        out: list[tuple] = []
        for fmin, fmax, amax, channel in self._forbidden_bands:
            amp = amax * 1e-3 * gamma
            out.append((fmin, fmax, amp, channel) if include_channel and channel else (fmin, fmax, amp))
        return out

    # ── PNS ──────────────────────────────────────────────────────────

    def default_stim_threshold(self) -> float | None:
        """PNS stimulation threshold ``rheobase / alpha`` in Hz/m/s, if known."""
        if self.rheobase is None or self.alpha is None or self.alpha <= 0.0:
            return None
        return float(self.rheobase) / float(self.alpha)


Opts.default = Opts()


def default_system(system: _pp.Opts | None) -> _pp.Opts:
    """Return ``system`` or Pulserver's current default limits."""
    return Opts.default if system is None else system
