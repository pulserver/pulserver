"""PNS and acoustic-band helpers, and the bridge to the C safety core.

The functions in the first half decide nothing: the authoritative gates are
the C safety core (``pulseg_check_safety``, run by the interpreter at
predownload) and the vendor's own predownload checks, and these exist so the
same quantities can be *looked at* while a sequence is being written.

* :func:`read_esp_bands` / :func:`read_asc_bands` / :func:`bands_to_resonances`
  — reading mechanical resonance bands out of a vendor lockout table, in either
  vendor's spelling, and handing them to upstream's spectrogram plotter.

Both table formats are documented here; no vendor table of either kind is
distributed with Pulserver, and none is needed to use the rest of the module.

The second half is what :meth:`~.Sequence.calculate_pns`,
:meth:`~.Sequence.calculate_gradient_spectrum` and :meth:`~.Sequence.plot`
stand on when they are asked about a TR rather than about the timeline. The
design point is that **none of upstream PyPulseq's analysis or plotting code
is reimplemented here, and neither is the TR**: the C safety core builds the
canonical TR, and :class:`TRSequence` — a real :class:`pypulseq.Sequence` —
is only the shape upstream needs to read it in.

Upstream reaches a sequence two ways, and :class:`TRSequence` answers both
against the one ``pulseg_get_tr_waveforms`` call:

* **Analysis** goes through ``system``/``grad_raster_time``/``get_gradients()``,
  and ``get_gradients`` is ``waveforms()`` fed through a PPoly constructor. So
  overriding ``waveforms()`` is the whole of it.
* **Plotting** walks ``block_events``, asking ``get_block`` for each one and
  accumulating ``block_durations``. So those three answer too, cutting the
  core's arrays at the block boundaries the core itself reported.

Which is what makes a picture and a predownload verdict incapable of
disagreeing: they are drawn from the same array.
"""

from __future__ import annotations

__all__ = [
    "bands_to_hz_per_m",
    "bands_to_resonances",
    "read_asc_bands",
    "read_esp_bands",
]

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

#: Axis order of an ESP lockout table.
_ESP_AXES = ("gx", "gy", "gz")

#: Largest band count a single ESP axis may declare, matching
#: ``PULSERVER_MAX_ESP_PER_AXIS``.
_MAX_ESP_PER_AXIS = 10


def _esp_rows(lines: list[str]) -> list[str]:
    """Strip comment (``#``) and blank lines from an ESP table."""
    return [
        line
        for line in (raw.strip() for raw in lines)
        if line and not line.startswith("#")
    ]


def read_esp_bands(path: str | Path) -> list[tuple[float, float, float, str]]:
    """Read mechanical resonance bands from a vendor ESP lockout table.

    The file lists three axes in X, Y, Z order. Each axis is a count line
    followed by that many ``esp_min_us esp_max_us max_amplitude_G_per_cm``
    rows. Echo spacings map to frequency through ``f = 1 / (2 * ESP)``,
    because an EPI gradient period spans two echo spacings.

    No ESP table ships with Pulserver — supply the one for the system you are
    targeting.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ESP lockout file (e.g. ``epiesp.dat``).

    Returns
    -------
    list of tuple
        ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m, channel)``,
        where channel is ``'gx'``, ``'gy'`` or ``'gz'``.

    Raises
    ------
    ValueError
        If the table is malformed. This mirrors the scanner-side behaviour,
        where a present-but-corrupt lockout table fails closed rather than
        degrading to "no bands".
    """
    rows = _esp_rows(Path(path).read_text().splitlines())
    bands: list[tuple[float, float, float, str]] = []
    cursor = 0

    for axis in _ESP_AXES:
        if cursor >= len(rows):
            raise ValueError(f"ESP table {path}: truncated before the {axis} axis")
        try:
            count = int(rows[cursor].split()[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"ESP table {path}: bad band count on line {cursor + 1}"
            ) from exc
        cursor += 1
        if not 0 <= count <= _MAX_ESP_PER_AXIS:
            raise ValueError(
                f"ESP table {path}: implausible band count {count} for {axis}"
            )

        for _ in range(count):
            if cursor >= len(rows):
                raise ValueError(f"ESP table {path}: truncated inside the {axis} axis")
            fields = rows[cursor].split()
            cursor += 1
            if len(fields) < 3:
                raise ValueError(
                    f"ESP table {path}: expected 'min max amplitude', got {fields!r}"
                )
            esp_min, esp_max, amp_g_per_cm = (
                float(fields[0]),
                float(fields[1]),
                float(fields[2]),
            )
            if esp_min <= 0 or esp_max <= 0 or esp_max < esp_min:
                raise ValueError(
                    f"ESP table {path}: invalid echo-spacing range {esp_min}-{esp_max}"
                )
            if amp_g_per_cm < 0:
                raise ValueError(
                    f"ESP table {path}: negative amplitude limit {amp_g_per_cm}"
                )
            bands.append((5.0e5 / esp_max, 5.0e5 / esp_min, amp_g_per_cm * 10.0, axis))

    return bands


def read_asc_bands(path: str | Path) -> list[tuple[float, float, float]]:
    """Read mechanical resonance bands from a Siemens ``.asc`` file.

    The same tables :meth:`pypulseq.Sequence.calculate_pns` reads for its SAFE
    hardware description also declare the gradient coil's acoustic resonances,
    as a centre frequency and a bandwidth per resonance. This reads them with
    upstream's own parser and restates them as ``(fmin, fmax)`` bands.

    A Siemens table declares no amplitude limit and no axis, so the limit comes
    back as ``0.0``: the safety core then falls back to its hardware-scaled
    floor, ``0.08 * max_grad``, for that band. Bands are unlabelled, which is
    what the core wants — it checks every axis against every band regardless.

    No ``.asc`` table ships with Pulserver — supply the one for the system you
    are targeting.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Siemens ``.asc`` file.

    Returns
    -------
    list of tuple
        ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m)``.
    """
    from pypulseq.utils.siemens.asc_to_hw import asc_to_acoustic_resonances
    from pypulseq.utils.siemens.readasc import readasc

    asc, _ = readasc(str(path))
    bands: list[tuple[float, float, float]] = []
    for resonance in asc_to_acoustic_resonances(asc):
        centre = float(resonance["frequency"])
        width = float(resonance["bandwidth"])
        if centre <= 0.0 or width <= 0.0:
            continue
        bands.append((max(centre - 0.5 * width, 0.0), centre + 0.5 * width, 0.0))
    return bands


def bands_to_resonances(
    bands: list[tuple[float, float, float] | tuple[float, float, float, str]],
) -> list[dict[str, float]]:
    """Convert forbidden bands to upstream's ``acoustic_resonances`` form.

    Parameters
    ----------
    bands : list of tuple
        ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m[, channel])``.

    Returns
    -------
    list of dict
        Entries with ``'frequency'`` and ``'bandwidth'`` keys, as
        :meth:`pypulseq.Sequence.calculate_gradient_spectrum` expects.
    """
    return [
        {
            "frequency": 0.5 * (float(band[0]) + float(band[1])),
            "bandwidth": float(band[1]) - float(band[0]),
        }
        for band in bands
    ]


def bands_to_hz_per_m(
    bands: list[tuple[float, float, float] | tuple[float, float, float, str]],
    *,
    gamma: float | None = None,
    keep_channel: bool = False,
) -> list[tuple]:
    """Restate forbidden bands from mT/m to Hz/m.

    :func:`read_esp_bands` and :func:`read_asc_bands` report amplitude limits
    in mT/m, which is how both vendors' tables state them. The C safety core
    and :meth:`~.Sequence.calculate_gradient_spectrum`'s ``bands`` argument
    want Hz/m. This is the conversion between the two.

    Parameters
    ----------
    bands : list of tuple
        ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m[, channel])``.
    gamma : float, optional
        Gyromagnetic ratio in Hz/T; defaults to
        :attr:`pulserver.pypulseq.Opts.default`'s.
    keep_channel : bool, default False
        Keep the trailing channel tag on the bands that carry one.

    Returns
    -------
    list of tuple
        ``(freq_min_hz, freq_max_hz, max_amplitude_hz_per_m[, channel])``.
    """
    if gamma is None:
        gamma = float(pp.Opts.default.gamma)
    out: list[tuple] = []
    for band in bands:
        converted = (
            float(band[0]),
            float(band[1]),
            float(band[2]) * 1e-3 * float(gamma),
        )
        out.append(
            (*converted, str(band[3])) if keep_channel and len(band) > 3 else converted
        )
    return out


# %% the C safety core's own object


#: The uniform raster the C safety core resamples gradients onto, as a
#: fraction of the gradient raster. Half-raster sampling is what
#: ``pulseg__get_gradient_waveforms_range`` uses so that a block boundary
#: falling mid-raster still lands on a sample; every quantity the core
#: reports -- PNS sample counts included -- is on this grid.
SAFETY_RASTER_FRACTION = 0.5

#: ``pulseg_types.h``'s amplitude modes, by the name this module uses for
#: them. ``"worst_case"`` is what ``pulseg_check_safety`` runs on: not any
#: single TR, but the per-sample maximum over every instance of one.
AMPLITUDE_MODES = {"worst_case": 0, "zero_variable": 1, "actual": 2}


@dataclass(frozen=True)
class MechResonances:
    """The C safety core's mechanical-resonance analysis of one TR.

    A periodic sequence does not have a continuous gradient spectrum: it has
    a line spectrum, with energy only at multiples of the TR fundamental
    ``1 / tr_duration``. This carries those lines, in the equivalent-drive
    units the acoustic verdict is stated in --
    ``A_eq(f) = (2 / T_TR) * |S_ax(f)|``, the amplitude of the constant
    sinusoidal gradient that would drive the resonance as hard.

    Attributes
    ----------
    tr_duration : float
        The TR the lines are harmonics of, in seconds.
    num_instances : int
        How many times that TR repeats, which sets how sharp each line is.
    line_freqs : numpy.ndarray
        ``(L,)`` the harmonic grid ``k / tr_duration``, in Hz.
    line_a_eq : numpy.ndarray
        ``(L, 3)`` A_eq per axis at each line, in Hz/m.
    line_widths : numpy.ndarray
        ``(L,)`` FWHM of each line, in Hz.
    candidate_freqs : numpy.ndarray
        ``(C,)`` the subset of lines falling inside a guarded forbidden band
        -- the only ones the verdict looks at, in Hz.
    candidate_a_eq : numpy.ndarray
        ``(C, 3)`` A_eq per axis at each candidate, in Hz/m.
    violations : numpy.ndarray
        ``(C,)`` bool, whether each candidate exceeds its band's tolerance.
    bands : list of tuple
        The forbidden bands the candidates were selected against.

    Notes
    -----
    ``line_a_eq`` is **not** the same quantity as the magnitude a spectrogram
    reports: A_eq is an equivalent constant drive amplitude in Hz/m, while
    :meth:`~.Sequence.calculate_gradient_spectrum` returns a short-time
    Fourier magnitude whose scale depends on the window length. Plotting them
    against a shared vertical axis would be meaningless, which is why
    :func:`overlay_resonance_lines` twins its own.
    """

    tr_duration: float
    num_instances: int
    line_freqs: np.ndarray
    line_a_eq: np.ndarray
    line_widths: np.ndarray
    candidate_freqs: np.ndarray
    candidate_a_eq: np.ndarray
    violations: np.ndarray
    bands: list

    @property
    def ok(self) -> bool:
        """Whether no candidate line violates its band."""
        return not bool(np.any(self.violations))

    @classmethod
    def from_spectra(
        cls, spectra: dict, tr_duration: float, bands: list
    ) -> MechResonances:
        """Build one from the dict ``_calc_mech_resonances`` hands back."""

        def stack(prefix: str, count: int) -> np.ndarray:
            if count == 0:
                return np.zeros((0, 3))
            return np.stack(
                [np.asarray(spectra[f"{prefix}_g{a}"], float) for a in "xyz"], axis=-1
            )

        num_lines = int(spectra["num_analytical_peaks"])
        num_candidates = int(spectra["num_candidates"])
        return cls(
            tr_duration=float(tr_duration),
            num_instances=int(spectra["num_instances"]),
            line_freqs=np.asarray(spectra["analytical_peak_freqs"], float),
            line_a_eq=stack("analytical_peak_amp", num_lines),
            line_widths=np.asarray(spectra["analytical_peak_widths_hz"], float),
            candidate_freqs=np.asarray(spectra["candidate_freqs"], float),
            candidate_a_eq=stack("candidate_amps", num_candidates),
            violations=np.asarray(spectra["candidate_violations"], bool),
            bands=list(bands),
        )


class TRSequence(pp.Sequence):
    """A :class:`pypulseq.Sequence` whose contents are one the C core extracted.

    Upstream's ``get_gradients`` is ``waveforms()`` run through a PPoly
    constructor, and every analysis built on it -- ``calc_pns``,
    ``calculate_gradient_spectrum`` -- reaches the waveform only that way. So
    overriding ``waveforms()`` is enough to point the whole of upstream's
    analysis code at a waveform that is not on the sequence's timeline at all,
    such as the per-sample maximum over every instance of a TR that
    :func:`pulseg_check_safety` judges.

    Upstream's *plotting* reaches a sequence a second way -- it walks
    :attr:`block_events`, asking :meth:`get_block` for each one and
    accumulating :attr:`block_durations` as it goes -- so drawing a TR needs
    those three to answer as well. They do, out of the same
    ``pulseg_get_tr_waveforms`` call: the C core reports the TR's block
    boundaries alongside its waveforms, and :meth:`get_block` slices the
    channels against them. **Nothing is reconstructed from the timeline**; a
    drawn TR is the array the safety checks were handed, cut at the block
    edges the core itself reported.

    Parameters
    ----------
    system : pypulseq.Opts
        The system the waveform was laid out on.
    channels : sequence of numpy.ndarray
        One ``(2, N)`` array per gradient axis: breakpoint times in seconds
        on the first row, amplitudes in Hz/m on the second. This is exactly
        the shape upstream's own ``waveforms()`` returns.
    duration : float
        The waveform's duration in seconds.
    blocks : list of dict, optional
        The core's block descriptors, each carrying ``start`` and ``duration``
        in seconds. Without them the TR is one block wide, which is all the
        analyses need and less than plotting does.
    rf : tuple of numpy.ndarray, optional
        ``(times, magnitudes, phases)`` on the RF raster, in seconds, Hz and
        radians, over the whole TR. Magnitude and phase are channel-major when
        ``num_rf_channels`` exceeds one.
    adc_events : list of dict, optional
        One entry per ADC in the TR, with ``onset`` and ``dwell`` in seconds.
    num_rf_channels : int, default 1
        Transmit channels the RF arrays hold.
    rf_channel : int, default 0
        Which of them :meth:`get_block` reports. Upstream's block model has
        one RF event, so a pTx TR is drawn one channel at a time; see
        :func:`overlay_rf_channels`.

    Notes
    -----
    The blocks this reports are **display objects, not Pulseq events**: their
    gradients are the core's resampled trace rather than the trapezoid or
    arbitrary shape the file stores, and their RF carries its phase baked into
    the signal rather than in ``phase_offset``. That is deliberate --- what is
    drawn should be what the core measured.
    """

    def __init__(
        self,
        system: pp.Opts,
        channels,
        duration: float,
        *,
        blocks: list | None = None,
        rf: tuple | None = None,
        adc_events: list | None = None,
        num_rf_channels: int = 1,
        rf_channel: int = 0,
    ) -> None:
        super().__init__(system=system)
        self._channels = tuple(np.asarray(channel, dtype=float) for channel in channels)
        self._duration = float(duration)
        self._rf = rf
        self._adc_events = list(adc_events or [])
        self.num_rf_channels = int(num_rf_channels)
        self.rf_channel = int(rf_channel)

        self._blocks = list(blocks or [])
        if not self._blocks:
            # Read by get_gradients only for an axis that carries nothing, and
            # only to give a background-gradient offset somewhere to live.
            self.block_durations = {1: self._duration}
            self.block_events = {}
            return

        # Upstream's plotter iterates block_events for its counters and reads
        # block_durations under the same keys, so both are 1-based and dense.
        self.block_durations = {
            number: block["duration"]
            for number, block in enumerate(self._blocks, start=1)
        }
        self.block_events = {
            number: np.zeros(7, dtype=int) for number in range(1, len(self._blocks) + 1)
        }
        self._rf_spans = self._split_rf()

    def _split_rf(self) -> dict:
        """Which slice of the RF arrays each block owns, and over how many channels.

        The core emits a block's RF samples contiguously and **channel-major
        within the block**, repeating the time base once per transmit channel,
        so the TR-wide time array is not monotonic under pTx and cannot be
        searched. It can still be *bucketed*: every sample of a block lies
        inside that block's window, so one pass over the block edges assigns
        them all.

        The channel count is then read off the times themselves -- each repeat
        of the time base restarts it -- rather than off the TR-wide
        ``num_rf_channels``, which is the largest any block uses and need not
        be what this one does.
        """
        if self._rf is None:
            return {}

        times = self._rf[0]
        edges = np.array([block["start"] for block in self._blocks] + [np.inf])
        owner = np.searchsorted(edges, times, side="right") - 1

        spans = {}
        for index in np.unique(owner):
            if index < 0:
                continue
            (positions,) = np.nonzero(owner == index)
            first, last = int(positions[0]), int(positions[-1]) + 1
            within = times[first:last]
            channels = 1 + int(np.count_nonzero(np.diff(within) <= 0))
            if (last - first) % channels:
                # Not the layout described above; treat it as single-channel
                # rather than slicing it into pieces that mean nothing.
                channels = 1
            spans[int(index) + 1] = (first, last, channels)
        return spans

    def rf_over(
        self, block_number: int
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Block ``block_number``'s RF as ``(times, magnitudes, phases)`` per channel.

        Empty when the block carries no RF. Times are TR-relative, magnitudes
        in Hz and phases in radians, exactly as the core reported them.
        """
        span = getattr(self, "_rf_spans", {}).get(block_number)
        if span is None:
            return []
        first, last, channels = span
        times, magnitudes, phases = self._rf
        count = (last - first) // channels
        return [
            (
                times[first : first + count],
                magnitudes[first + channel * count : first + (channel + 1) * count],
                phases[first + channel * count : first + (channel + 1) * count],
            )
            for channel in range(channels)
        ]

    def waveforms(self, append_RF: bool = False, time_range=None):  # noqa: ARG002
        """The extracted waveform, in upstream's ``(2, N)``-per-axis form.

        ``append_RF`` and ``time_range`` are upstream's and are accepted so
        that this stands in for a real sequence; neither applies to a single
        extracted TR, and both are ignored.
        """
        return self._channels

    def get_block(self, block_index: int) -> SimpleNamespace:
        """Block ``block_index`` of the TR, 1-based, as upstream's plotter reads it.

        The gradient, RF and ADC events are the core's own arrays cut at the
        block boundary the core reported, with times made relative to the
        block start so that upstream's running ``t0`` puts them back where
        they were.
        """
        if not self._blocks:
            raise IndexError("this TR carries no block descriptors")
        block = self._blocks[block_index - 1]
        start, duration = block["start"], block["duration"]

        made = SimpleNamespace(
            block_duration=duration,
            rf=self._rf_event(block_index, start),
            adc=self._adc_over(start, duration),
            label=None,
            soft_delay=None,
        )
        for axis, channel in zip("xyz", self._channels, strict=True):
            setattr(made, f"g{axis}", _grad_over(channel, start, duration))
        return made

    def _rf_event(self, block_number: int, start: float) -> SimpleNamespace | None:
        """Block ``block_number``'s RF on :attr:`rf_channel`, or None."""
        channels = self.rf_over(block_number)
        if not channels:
            return None
        times, magnitudes, phases = channels[min(self.rf_channel, len(channels) - 1)]

        return SimpleNamespace(
            type="rf",
            # Phase is already in the signal: the core reports what is played,
            # and upstream would otherwise add phase_offset on top of it.
            signal=magnitudes * np.exp(1j * phases),
            t=times - start,
            shape_dur=float(times[-1] - times[0]) if times.size else 0.0,
            delay=0.0,
            freq_offset=0.0,
            phase_offset=0.0,
            freq_ppm=0.0,
            phase_ppm=0.0,
            use="undefined",
        )

    def _adc_over(self, start: float, duration: float) -> SimpleNamespace | None:
        """The ADC starting inside ``start..start + duration``, or None."""
        for event in self._adc_events:
            if start <= event["onset"] < start + duration:
                return SimpleNamespace(
                    type="adc",
                    num_samples=event["num_samples"],
                    dwell=event["dwell"],
                    delay=event["onset"] - start,
                    freq_offset=event["freq_offset"],
                    phase_offset=event["phase_offset"],
                    freq_ppm=0.0,
                    phase_ppm=0.0,
                    phase_modulation=None,
                    dead_time=0.0,
                )
        return None

    @classmethod
    def from_c(
        cls, waveforms: dict, system: pp.Opts, *, rf_channel: int = 0
    ) -> TRSequence:
        """Build one from the dict ``_get_tr_waveforms`` hands back."""
        channels = []
        for axis in "xyz":
            channel = waveforms[f"g{axis}"]
            times = np.asarray(channel["time_us"], dtype=float) * 1e-6
            amplitudes = np.asarray(channel["amplitude"], dtype=float)
            channels.append(np.stack((times, amplitudes)))

        rf = None
        magnitude, phase = waveforms.get("rf_mag"), waveforms.get("rf_phase")
        if magnitude and len(magnitude["time_us"]):
            rf = (
                np.asarray(magnitude["time_us"], dtype=float) * 1e-6,
                np.asarray(magnitude["amplitude"], dtype=float),
                np.asarray(phase["amplitude"], dtype=float),
            )

        adc_events = [
            {
                "onset": float(event["onset_us"]) * 1e-6,
                # The core reports the window the samples span; upstream draws
                # them from the dwell, which is that window per sample.
                "dwell": float(event["duration_us"])
                * 1e-6
                / max(int(event["num_samples"]), 1),
                "num_samples": int(event["num_samples"]),
                "freq_offset": float(event["freq_offset_hz"]),
                "phase_offset": float(event["phase_offset_rad"]),
            }
            for event in waveforms.get("adc_events", [])
        ]

        blocks = [
            {
                "start": float(block["start_us"]) * 1e-6,
                "duration": float(block["duration_us"]) * 1e-6,
                "segment": int(block["segment_idx"]),
            }
            for block in waveforms.get("blocks", [])
        ]

        return cls(
            system,
            channels,
            float(waveforms["total_duration_us"]) * 1e-6,
            blocks=blocks,
            rf=rf,
            adc_events=adc_events,
            num_rf_channels=int(waveforms.get("num_rf_channels", 1)),
            rf_channel=rf_channel,
        )


def _grad_over(
    channel: np.ndarray, start: float, duration: float
) -> SimpleNamespace | None:
    """One gradient axis over ``start..start + duration``, in upstream's arbitrary form.

    A gradient axis is one monotonic time base over the whole TR --- unlike
    RF, which repeats its own once per transmit channel --- so the block's
    samples are found by search rather than by bucketing.

    The core's trace is continuous across a block boundary, so the sample
    sitting on the closing edge is kept as well: dropping it would draw every
    block falling back towards zero just before its neighbour picks it up.
    """
    times, amplitudes = channel
    first = int(np.searchsorted(times, start, side="left"))
    last = int(np.searchsorted(times, start + duration, side="left"))
    if last <= first:
        return None
    last = min(last + 1, times.size)

    window = amplitudes[first:last]
    if not np.any(window):
        return None

    return SimpleNamespace(
        type="grad",
        waveform=window,
        tt=times[first:last] - start,
        shape_dur=duration,
        delay=0.0,
        first=window[0],
        last=window[-1],
    )


def safe_hardware(hardware) -> SimpleNamespace:
    """Normalise a SAFE hardware description, however it was given.

    Parameters
    ----------
    hardware : str or pathlib.Path or types.SimpleNamespace
        A Siemens ``.asc`` file, or an already-parsed hardware description of
        the kind :func:`pypulseq.utils.safe_pns_prediction.safe_example_hw`
        returns. Reading is upstream's ``readasc``/``asc_to_hw``.

    Returns
    -------
    types.SimpleNamespace
        With ``x``, ``y`` and ``z``, each carrying ``a1``--``a3``,
        ``tau1``--``tau3`` (ms), ``stim_limit`` (T/m/s) and ``g_scale``.
    """
    if isinstance(hardware, (str, Path)):
        from pypulseq.utils.siemens.asc_to_hw import asc_to_hw
        from pypulseq.utils.siemens.readasc import readasc

        asc, _ = readasc(str(hardware))
        hardware = asc_to_hw(asc)

    missing = [axis for axis in "xyz" if not hasattr(hardware, axis)]
    if missing:
        raise ValueError(
            f"SAFE hardware description is missing axis {', '.join(missing)}; "
            "pass a .asc path or a namespace shaped like safe_example_hw()"
        )
    return hardware


def safe_coefficients(hardware) -> tuple[tuple[float, ...], ...]:
    """A SAFE description as the three coefficient tuples the binding takes.

    Returns
    -------
    tuple
        ``(gx, gy, gz)``, each ``(a1, a2, a3, tau1_ms, tau2_ms, tau3_ms,
        stim_limit, g_scale)``.
    """
    hardware = safe_hardware(hardware)
    fields = ("a1", "a2", "a3", "tau1", "tau2", "tau3", "stim_limit", "g_scale")
    return tuple(
        tuple(float(getattr(getattr(hardware, axis), field)) for field in fields)
        for axis in "xyz"
    )


def is_safe_hardware(hardware) -> bool:
    """Whether ``hardware`` describes a SAFE model rather than an Irnich one.

    A ``.asc`` path or a per-axis namespace means SAFE. A mapping or namespace
    carrying ``chronaxie`` means the Irnich rheobase/chronaxie model the GE
    gate uses. Nothing else is recognised.
    """
    if isinstance(hardware, (str, Path)):
        return True
    if isinstance(hardware, dict):
        return False
    return all(hasattr(hardware, axis) for axis in "xyz")


def irnich_coefficients(hardware) -> tuple[float, float, float]:
    """An Irnich model description as ``(chronaxie_us, rheobase, alpha)``.

    Parameters
    ----------
    hardware : dict or types.SimpleNamespace
        Carrying ``chronaxie`` (seconds) or ``chronaxie_us``, ``rheobase``
        in T/m/s, and optionally ``alpha`` (default 1).
    """
    read = (
        hardware.get
        if isinstance(hardware, dict)
        else lambda k, d=None: getattr(hardware, k, d)
    )

    chronaxie_us = read("chronaxie_us")
    if chronaxie_us is None:
        chronaxie = read("chronaxie")
        if chronaxie is None:
            raise ValueError("Irnich PNS model needs 'chronaxie' (s) or 'chronaxie_us'")
        chronaxie_us = float(chronaxie) * 1e6

    rheobase = read("rheobase")
    if rheobase is None:
        raise ValueError("Irnich PNS model needs 'rheobase' in T/m/s")

    return float(chronaxie_us), float(rheobase), float(read("alpha", 1.0) or 1.0)


#: Fractions of the stimulation threshold worth drawing on a PNS plot, as
#: ``percent -> (label, colour, linestyle)``. 100 % is the model's own limit.
#: 80 % is where a sequence stops having any margin worth the name: the
#: models are calibrated on populations rather than on the subject in the
#: magnet, and vendors derate accordingly, so a design that only clears 100 %
#: is one recalibration away from not clearing it.
PNS_THRESHOLDS = {
    100.0: ("100% threshold", "tab:red", "-"),
    80.0: ("80% margin", "tab:orange", "--"),
}


def overlay_pns_thresholds(axes=None) -> None:
    """Mark the stimulation threshold and the 80 % margin on a PNS plot.

    Adds to the current axes, which -- because both
    :func:`pypulseq.utils.safe_pns_prediction.safe_plot` and upstream's
    :func:`~pypulseq.Sequence.calc_pns.calc_pns` draw through pyplot and
    leave their figure current -- is the PNS panel just drawn, whichever
    nerve model produced it.

    Upstream's plot marks only the peak the sequence reached, which says how
    high it got but not how close that is to being a problem. These two say
    what the number has to beat.

    Parameters
    ----------
    axes : matplotlib.axes.Axes, optional
        Where to draw. Defaults to the current axes.

    Notes
    -----
    Labelled with text at the right edge rather than through the legend:
    ``safe_plot`` builds its legend from an explicit four-entry list, so
    calling :meth:`~matplotlib.axes.Axes.legend` again would rebuild it from
    labelled artists and drop the four traces it is there to name.
    """
    import matplotlib.pyplot as plt

    axes = plt.gca() if axes is None else axes
    right = axes.get_xlim()[1]

    # Read the traces before adding any of our own, so the threshold lines do
    # not count towards the peak they are being compared against.
    drawn = [np.asarray(line.get_ydata(), dtype=float) for line in axes.lines]
    peak = max(
        (float(np.nanmax(values)) for values in drawn if values.size), default=0.0
    )

    for level, (label, colour, style) in PNS_THRESHOLDS.items():
        axes.axhline(level, color=colour, linestyle=style, linewidth=1.0, zorder=0)
        axes.text(
            right,
            level,
            f" {label}",
            color=colour,
            fontsize="x-small",
            va="bottom",
            ha="right",
        )

    # safe_plot fixes the axis at 0-120 %, which crops away precisely the
    # sequence worth looking at: one that overshoots leaves the panel and its
    # shape cannot be read at all. Grow to fit, never shrink, so a passing
    # plot keeps upstream's framing and a failing one is legible.
    bottom, top = axes.get_ylim()
    axes.set_ylim(bottom, max(top, 1.1 * peak))


def overlay_resonance_lines(
    resonances: MechResonances, axes=None, *, max_frequency=None
) -> None:
    """Draw a :class:`MechResonances` line spectrum over a gradient spectrogram.

    Adds to the current figure, which -- because upstream draws through
    pyplot and leaves its figure current -- is the one
    :meth:`~.Sequence.calculate_gradient_spectrum` just produced.

    The lines go on a **twinned vertical axis** carrying their own Hz/m
    label. A_eq and a short-time Fourier magnitude are different quantities
    (see :class:`MechResonances`), and sharing one scale would invite reading
    a crossing point as meaningful.

    Parameters
    ----------
    resonances : MechResonances
        What to draw.
    axes : matplotlib.axes.Axes, optional
        Where to draw. Defaults to the current axes.
    max_frequency : float, optional
        Drop lines above this frequency. Defaults to the host axes' limit.
    """
    import matplotlib.pyplot as plt

    axes = plt.gca() if axes is None else axes
    limit = axes.get_xlim()[1] if max_frequency is None else float(max_frequency)

    twin = axes.twinx()
    twin.set_ylabel("$A_{eq}$ (Hz/m)")

    keep = resonances.line_freqs <= limit
    if np.any(keep):
        twin.vlines(
            resonances.line_freqs[keep],
            0.0,
            resonances.line_a_eq[keep].max(axis=-1),
            color="0.6",
            linewidth=0.8,
            label="$A_{eq}$ at $k/T_{TR}$",
        )

    flagged = resonances.violations & (resonances.candidate_freqs <= limit)
    if np.any(flagged):
        twin.vlines(
            resonances.candidate_freqs[flagged],
            0.0,
            resonances.candidate_a_eq[flagged].max(axis=-1),
            color="tab:red",
            linewidth=1.6,
            label="violating",
        )

    twin.set_ylim(bottom=0.0)
    twin.legend(loc="upper right", fontsize="small")


def overlay_rf_channels(
    waveform: TRSequence, magnitude_axes, phase_axes, time_factor
) -> None:
    """Draw a pTx TR's remaining transmit channels over the one already plotted.

    Upstream's block model carries a single RF event, so upstream's plotter
    draws one transmit channel and has no way to be told about the others.
    This adds them to the same two panels, from the same channel-major arrays
    the C core reported, so a pTx TR is looked at whole rather than one
    channel at a time.

    Does nothing for a single-transmit TR, which is the case that has to stay
    pixel-for-pixel upstream's.

    Parameters
    ----------
    waveform : TRSequence
        The extracted TR, whose ``rf_channel`` is the one already drawn.
    magnitude_axes, phase_axes : matplotlib.axes.Axes
        Upstream's RF magnitude and RF/ADC phase panels.
    time_factor : float
        The seconds-to-display-unit factor upstream drew its own traces with.
    """
    if waveform.num_rf_channels <= 1:
        return

    named = set()
    for number in waveform.block_events:
        for channel, (times, magnitudes, phases) in enumerate(waveform.rf_over(number)):
            if channel == waveform.rf_channel:
                continue
            # One legend entry per channel, not per block that transmits on it.
            label = f"Tx{channel}" if channel not in named else None
            named.add(channel)
            magnitude_axes.plot(time_factor * times, magnitudes, label=label)
            phase_axes.plot(time_factor * times, phases, linewidth=0.5)

    if named:
        magnitude_axes.legend(loc="upper right", fontsize="x-small")
