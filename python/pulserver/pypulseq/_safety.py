"""PNS and acoustic-band helpers, and the bridge to the C safety core.

The functions in the first half decide nothing: the authoritative gates are
the C safety core (``pulseg_check_safety``, run by the interpreter at
predownload) and the vendor's own predownload checks, and these exist so the
same quantities can be *looked at* while a sequence is being written.

* :func:`chronaxie_pns` — the Irnich rheobase/chronaxie nerve model, the form
  GE uses, as opposed to the SAFE model upstream PyPulseq implements. It is a
  line-for-line Python counterpart of ``pulserver_ge_pns.c`` so a plot here and
  a predownload verdict on the scanner cannot silently disagree.
* :func:`read_esp_bands` / :func:`read_asc_bands` / :func:`bands_to_resonances`
  — reading mechanical resonance bands out of a vendor lockout table, in either
  vendor's spelling, and handing them to upstream's spectrogram plotter.

Both table formats are documented here; no vendor table of either kind is
distributed with Pulserver, and none is needed to use the rest of the module.

The second half is what :meth:`~.Sequence.calculate_pns` and
:meth:`~.Sequence.calculate_gradient_spectrum` stand on when they are asked
about a TR rather than about the timeline. The design point is that **none of
upstream PyPulseq's analysis or plotting code is reimplemented here**: both of
upstream's entry points reach their sequence through only
``system``/``grad_raster_time``/``get_gradients()``, and ``get_gradients`` is
``waveforms()`` fed through a PPoly constructor. So :class:`TRSequence` — a
real :class:`pypulseq.Sequence` that overrides ``waveforms()`` and nothing
else — hands the C library's extracted TR waveform to upstream's own code,
which then computes and draws exactly as it would for any other sequence.
"""

from __future__ import annotations

__all__ = ["bands_to_resonances", "chronaxie_pns", "read_asc_bands", "read_esp_bands"]

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

#: How many chronaxie constants of kernel to keep before truncating the
#: ``1/tau**2`` tail. Matches ``PULSERVER_GE_PNS_KERNEL_DURATION_FACTOR``.
_KERNEL_DURATION_FACTOR = 20.0

#: Axis order of an ESP lockout table.
_ESP_AXES = ("gx", "gy", "gz")

#: Largest band count a single ESP axis may declare, matching
#: ``PULSERVER_MAX_ESP_PER_AXIS``.
_MAX_ESP_PER_AXIS = 10


def chronaxie_kernel(dt: float, chronaxie_us: float, rheobase: float, alpha: float = 1.0) -> np.ndarray:
    """Irnich rheobase/chronaxie PNS kernel.

    Parameters
    ----------
    dt : float
        Sampling interval in seconds (the gradient raster).
    chronaxie_us : float
        Chronaxie time constant in microseconds.
    rheobase : float
        Rheobase in Hz/m/s.
    alpha : float, default 1.0
        Coil attenuation factor. The stimulation threshold is
        ``rheobase / alpha``.

    Returns
    -------
    numpy.ndarray
        Kernel ``k[i] = (dt / s_min) * c / (c + i*dt)**2``, normalised so
        that convolving a slew waveform in Hz/m/s yields a fraction of the
        threshold.
    """
    if chronaxie_us <= 0.0:
        raise ValueError("chronaxie_us must be > 0")
    if rheobase <= 0.0:
        raise ValueError("rheobase must be > 0")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")

    c = chronaxie_us * 1e-6
    s_min = rheobase / alpha
    n = int(_KERNEL_DURATION_FACTOR * c / dt) + 1
    tau = np.arange(n, dtype=float) * dt
    return (dt / s_min) * c / (c + tau) ** 2


def chronaxie_pns(
    gradients: np.ndarray,
    dt: float,
    *,
    chronaxie_us: float,
    rheobase: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """PNS response of a gradient waveform under the Irnich model.

    Parameters
    ----------
    gradients : numpy.ndarray
        Gradient waveform, shape ``(N, 3)``, in Hz/m.
    dt : float
        Sampling interval in seconds.
    chronaxie_us, rheobase, alpha
        Model parameters, see :func:`chronaxie_kernel`.

    Returns
    -------
    numpy.ndarray
        PNS per axis, shape ``(N, 3)``, as a **percentage** of the
        stimulation threshold. Take the root-sum-square across axes for the
        combined level.
    """
    gradients = np.atleast_2d(np.asarray(gradients, dtype=float))
    if gradients.shape[-1] != 3:
        raise ValueError("gradients must have shape (N, 3)")

    slew = np.diff(gradients, axis=0, prepend=gradients[:1]) / dt
    kernel = chronaxie_kernel(dt, chronaxie_us, rheobase, alpha)

    out = np.empty_like(slew)
    for axis in range(3):
        out[:, axis] = np.convolve(slew[:, axis], kernel)[: slew.shape[0]]
    return 100.0 * out


def _esp_rows(lines: list[str]) -> list[str]:
    """Strip comment (``#``) and blank lines from an ESP table."""
    return [line for line in (raw.strip() for raw in lines) if line and not line.startswith("#")]


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
            raise ValueError(f"ESP table {path}: bad band count on line {cursor + 1}") from exc
        cursor += 1
        if not 0 <= count <= _MAX_ESP_PER_AXIS:
            raise ValueError(f"ESP table {path}: implausible band count {count} for {axis}")

        for _ in range(count):
            if cursor >= len(rows):
                raise ValueError(f"ESP table {path}: truncated inside the {axis} axis")
            fields = rows[cursor].split()
            cursor += 1
            if len(fields) < 3:
                raise ValueError(f"ESP table {path}: expected 'min max amplitude', got {fields!r}")
            esp_min, esp_max, amp_g_per_cm = float(fields[0]), float(fields[1]), float(fields[2])
            if esp_min <= 0 or esp_max <= 0 or esp_max < esp_min:
                raise ValueError(f"ESP table {path}: invalid echo-spacing range {esp_min}-{esp_max}")
            if amp_g_per_cm < 0:
                raise ValueError(f"ESP table {path}: negative amplitude limit {amp_g_per_cm}")
            bands.append((5.0e5 / esp_max, 5.0e5 / esp_min, amp_g_per_cm * 10.0, axis))

    return bands


def read_asc_bands(path: str | Path) -> list[tuple[float, float, float]]:
    """Read mechanical resonance bands from a Siemens ``.asc`` file.

    The same tables :meth:`pypulseq.Sequence.calculate_pns` reads for its SAFE
    hardware description also declare the gradient coil's acoustic resonances,
    as a centre frequency and a bandwidth per resonance. This reads them with
    upstream's own parser and restates them as ``(fmin, fmax)`` bands, the form
    :class:`~pulserver.pypulseq.Opts` and the C safety core take.

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
        {"frequency": 0.5 * (float(band[0]) + float(band[1])), "bandwidth": float(band[1]) - float(band[0])}
        for band in bands
    ]


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
    def from_spectra(cls, spectra: dict, tr_duration: float, bands: list) -> MechResonances:
        """Build one from the dict ``_calc_mech_resonances`` hands back."""

        def stack(prefix: str, count: int) -> np.ndarray:
            if count == 0:
                return np.zeros((0, 3))
            return np.stack([np.asarray(spectra[f"{prefix}_g{a}"], float) for a in "xyz"], axis=-1)

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
    """A :class:`pypulseq.Sequence` whose waveform is one the C core extracted.

    Upstream's ``get_gradients`` is ``waveforms()`` run through a PPoly
    constructor, and every analysis built on it -- ``calc_pns``,
    ``calculate_gradient_spectrum`` -- reaches the waveform only that way. So
    overriding ``waveforms()`` is enough to point the whole of upstream's
    analysis and plotting code at a waveform that is not on the sequence's
    timeline at all, such as the per-sample maximum over every instance of a
    TR that :func:`pulseg_check_safety` judges.

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
    """

    def __init__(self, system: pp.Opts, channels, duration: float) -> None:
        super().__init__(system=system)
        self._channels = tuple(np.asarray(channel, dtype=float) for channel in channels)
        self._duration = float(duration)
        # Read by get_gradients only for an axis that carries nothing, and
        # only to give a background-gradient offset somewhere to live.
        self.block_durations = {1: self._duration}

    def waveforms(self, append_RF: bool = False, time_range=None):  # noqa: ARG002
        """The extracted waveform, in upstream's ``(2, N)``-per-axis form.

        ``append_RF`` and ``time_range`` are upstream's and are accepted so
        that this stands in for a real sequence; neither applies to a single
        extracted TR, and both are ignored.
        """
        return self._channels

    @classmethod
    def from_c(cls, waveforms: dict, system: pp.Opts) -> TRSequence:
        """Build one from the dict ``_get_tr_waveforms`` hands back."""
        channels = []
        for axis in "xyz":
            channel = waveforms[f"g{axis}"]
            times = np.asarray(channel["time_us"], dtype=float) * 1e-6
            amplitudes = np.asarray(channel["amplitude"], dtype=float)
            channels.append(np.stack((times, amplitudes)))
        return cls(system, channels, float(waveforms["total_duration_us"]) * 1e-6)


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
        tuple(float(getattr(getattr(hardware, axis), field)) for field in fields) for axis in "xyz"
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
        in Hz/m/s, and optionally ``alpha`` (default 1).
    """
    read = hardware.get if isinstance(hardware, dict) else lambda k, d=None: getattr(hardware, k, d)

    chronaxie_us = read("chronaxie_us")
    if chronaxie_us is None:
        chronaxie = read("chronaxie")
        if chronaxie is None:
            raise ValueError("Irnich PNS model needs 'chronaxie' (s) or 'chronaxie_us'")
        chronaxie_us = float(chronaxie) * 1e6

    rheobase = read("rheobase")
    if rheobase is None:
        raise ValueError("Irnich PNS model needs 'rheobase' in Hz/m/s")

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
    peak = max((float(np.nanmax(values)) for values in drawn if values.size), default=0.0)

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


def overlay_resonance_lines(resonances: MechResonances, axes=None, *, max_frequency=None) -> None:
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
