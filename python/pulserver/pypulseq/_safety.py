"""Visualisation-grade PNS and acoustic-band helpers.

Nothing here decides pass or fail. The authoritative gates are the C safety
core (``pulseg_check_safety``, run by the interpreter at predownload) and the
vendor's own predownload checks; these functions exist so the same quantities
can be *looked at* while a sequence is being written.

Two pieces are provided:

* :func:`chronaxie_pns` — the Irnich rheobase/chronaxie nerve model, the form
  GE uses, as opposed to the SAFE model upstream PyPulseq implements. It is a
  line-for-line Python counterpart of ``pulserver_ge_pns.c`` so a plot here and
  a predownload verdict on the scanner cannot silently disagree.
* :func:`read_esp_bands` / :func:`read_asc_bands` / :func:`bands_to_resonances`
  — reading mechanical resonance bands out of a vendor lockout table, in either
  vendor's spelling, and handing them to upstream's spectrogram plotter.

Both table formats are documented here; no vendor table of either kind is
distributed with Pulserver, and none is needed to use the rest of the module.
"""

from __future__ import annotations

__all__ = ["bands_to_resonances", "chronaxie_pns", "read_asc_bands", "read_esp_bands"]

from pathlib import Path

import numpy as np

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
