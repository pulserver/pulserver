"""Validation: compare pulserver waveforms against a reference source."""

__all__ = ['validate']

from pathlib import Path
from typing import Union

import numpy as np

from ._sequence import SequenceCollection
from ._waveforms import get_tr_waveforms


# ── helpers ──────────────────────────────────────────────────────────

def _interp_to_ref(t_ref, a_ref, t_test, a_test):
    """Interpolate *test* waveform onto *ref* time base, return aligned pair."""
    if len(t_ref) == 0 or len(t_test) == 0:
        return np.empty(0), np.empty(0)
    a_interp = np.interp(t_ref, t_test, a_test, left=0.0, right=0.0)
    return a_ref, a_interp


def _rms_error(ref, test):
    """Percentage RMS error relative to ref (0 if ref is silent)."""
    norm = np.sqrt(np.mean(ref ** 2))
    if norm < 1e-30:
        return 0.0
    return 100.0 * np.sqrt(np.mean((ref - test) ** 2)) / norm


def _abs_tr_start_s(src_seq, num_prep_blocks: int, tr_idx: int,
                     tr_duration_us: float) -> float:
    """Compute the absolute start time (in seconds) of a TR.

    Sums block durations over the preparation region and adds
    ``tr_idx * tr_duration``.
    """
    prep_s = 0.0
    bd = src_seq.block_durations
    for k in range(1, num_prep_blocks + 1):
        prep_s += bd.get(k, 0.0) if isinstance(bd, dict) else bd[k - 1]
    return prep_s + tr_idx * (tr_duration_us * 1e-6)


# ── reference extraction ─────────────────────────────────────────────

def _pypulseq_reference(seq, sequence_idx, tr_idx, tr_info):
    """Extract reference waveforms from pypulseq for a single TR.

    Returns
    -------
    dict
        Mapping of channel name to ``(time_us, amplitude)`` tuples.
        Times are relative to TR start; amplitudes in mT/m (grad) or
        µT (RF magnitude).
    """
    src_seq = seq._seqs[sequence_idx]
    gamma = seq.system.gamma

    t0 = _abs_tr_start_s(
        src_seq, tr_info['num_prep_blocks'], tr_idx,
        tr_info['tr_duration_us'],
    )
    t1 = t0 + tr_info['tr_duration_us'] * 1e-6

    # waveforms_and_times returns
    #   (wave_data, tfp_exc, tfp_ref, t_adc, fp_adc)
    # wave_data is a list of (2, N) ndarrays per channel.
    result = src_seq.waveforms_and_times(append_RF=True,
                                          time_range=[t0, t1])
    channels = result[0]  # list of (2,N) ndarrays

    hz_to_mT_per_m = 1.0 / (gamma * 1e-3)
    hz_to_uT = 1e6 / gamma

    ref: dict[str, tuple] = {}
    for ch_idx, ch_name in enumerate(('gx', 'gy', 'gz')):
        if ch_idx < len(channels):
            arr = channels[ch_idx]
            if arr.shape[1] > 0:
                ref[ch_name] = (
                    (arr[0] - t0) * 1e6,        # relative µs
                    arr[1] * hz_to_mT_per_m,
                )
                continue
        ref[ch_name] = (np.empty(0), np.empty(0))

    # RF (index 3 when append_RF=True)
    if len(channels) > 3:
        arr = channels[3]
        if arr.shape[1] > 0:
            ref['rf_mag'] = (
                (arr[0] - t0) * 1e6,
                np.abs(arr[1]) * hz_to_uT,
            )
        else:
            ref['rf_mag'] = (np.empty(0), np.empty(0))
    else:
        ref['rf_mag'] = (np.empty(0), np.empty(0))

    return ref


def _xml_reference(xml_path):
    """Extract reference waveforms from an XML file.

    Returns
    -------
    dict
        Same format as :func:`_pypulseq_reference`.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    g_per_cm_to_mT_per_m = 10.0
    g_to_uT = 100.0

    ref: dict[str, tuple] = {}
    for ch_name in ('gx', 'gy', 'gz'):
        elem = root.find(f'.//{ch_name}')
        if elem is not None and elem.text:
            data = np.fromstring(elem.text, sep=' ')
            if data.size >= 2:
                n = data.size // 2
                ref[ch_name] = (data[:n], data[n:] * g_per_cm_to_mT_per_m)
                continue
        ref[ch_name] = (np.empty(0), np.empty(0))

    rf_elem = root.find('.//rf')
    if rf_elem is not None and rf_elem.text:
        data = np.fromstring(rf_elem.text, sep=' ')
        if data.size >= 2:
            n = data.size // 2
            ref['rf_mag'] = (data[:n], np.abs(data[n:]) * g_to_uT)
        else:
            ref['rf_mag'] = (np.empty(0), np.empty(0))
    else:
        ref['rf_mag'] = (np.empty(0), np.empty(0))

    return ref


# ── public entry point ───────────────────────────────────────────────

def validate(
    seq: SequenceCollection,
    *,
    sequence_idx: int = 0,
    xml_path: Union[str, Path, None] = None,
    do_plot: bool = False,
    tr_range: tuple[int, int] = (0, 1),
    hide_prep: bool = True,
    hide_cooldown: bool = True,
    show_rf_centers: bool = False,
    show_echoes: bool = False,
    show_segments: bool = True,
    show_blocks: bool = False,
    max_grad_mT_per_m: Union[float, bool, None] = True,
    grad_atol: float | None = None,
    rf_rms_percent: float = 10.0,
) -> dict:
    """Compare pulserver C-backend waveforms against a reference.

    The reference is either the source pypulseq ``Sequence`` (default,
    ``xml_path=None``) or an exported XML waveform file.  Gradient
    channels are compared by maximum absolute error; RF magnitude is
    compared by RMS percentage error.

    When *do_plot* is ``True`` the base TR waveform plot is created from
    the C backend (using :func:`_plot.plot` Branch 1) and the reference
    is overlaid (Branch 2 for pypulseq, Branch 3 for XML).

    Parameters
    ----------
    seq : SequenceCollection
        Sequence collection to validate.
    sequence_idx : int
        Subsequence index (0-based, default 0).
    xml_path : str, Path or None
        Path to an XML waveform file.  ``None`` (default) validates
        against the stored pypulseq ``Sequence``.
    do_plot : bool
        Show a comparison overlay plot (default ``False``).
    tr_range : (int, int)
        Half-open TR index range ``[start, stop)`` to validate
        (default ``(0, 1)`` — first TR only).
    hide_prep : bool
        Hide preparation blocks in the plot (default ``True``).
    hide_cooldown : bool
        Hide cooldown blocks in the plot (default ``True``).
    show_rf_centers : bool
        Mark RF iso-centres on the plot (default ``False``).
    show_echoes : bool
        Mark echo (ADC centre) on the plot (default ``False``).
    show_segments : bool
        Colour-code gradients by segment (default ``True``).
    show_blocks : bool
        Block-boundary lines on the plot (default ``False``).
    max_grad_mT_per_m : float, bool or None
        Gradient-limit reference line.  ``True`` (default) derives the
        value from system limits; a float is used directly; ``False``
        or ``None`` disables the line.
    grad_atol : float or None
        Absolute gradient error tolerance in mT/m.  ``None`` (default)
        uses ``3 × max_slew × grad_raster_time``.
    rf_rms_percent : float
        RF magnitude percent-RMS error threshold (default 10).

    Returns
    -------
    dict
        ``'ok'`` : bool — ``True`` if all channels pass.
        ``'errors'`` : dict[str, float] — per-channel max/RMS error.
        ``'messages'`` : list[str] — human-readable violation messages.
    """
    from ._extension._pulseqlib_wrapper import _find_tr

    sys = seq.system

    # Default gradient tolerance: 3 slew steps (mT/m)
    if grad_atol is None:
        grad_atol = 3.0 * sys.max_slew * sys.grad_raster_time * 1e3

    # Resolve max-grad line for plotting
    if max_grad_mT_per_m is True:
        _max_grad_plot = sys.max_grad * 1e3          # T/m → mT/m
    elif isinstance(max_grad_mT_per_m, (int, float)) and max_grad_mT_per_m is not False:
        _max_grad_plot = float(max_grad_mT_per_m)
    else:
        _max_grad_plot = None

    # TR metadata
    tr_info = _find_tr(seq._cseq, subsequence_idx=sequence_idx)

    # ── Validate each TR in range ────────────────────────────
    errors_per_tr: list[dict[str, float]] = []
    messages: list[str] = []
    multi_tr = (tr_range[1] - tr_range[0]) > 1

    for tr_idx in range(tr_range[0], tr_range[1]):
        # C-backend waveforms
        wf = get_tr_waveforms(
            seq,
            subsequence_idx=sequence_idx,
            amplitude_mode='actual',
            tr_index=tr_idx,
        )

        # Reference waveforms
        if xml_path is None:
            ref = _pypulseq_reference(seq, sequence_idx, tr_idx, tr_info)
        else:
            ref = _xml_reference(xml_path)

        # Compare gradients
        tr_err: dict[str, float] = {}
        for ch in ('gx', 'gy', 'gz'):
            ref_t, ref_a = ref[ch]
            test_ch = getattr(wf, ch)
            r, t = _interp_to_ref(ref_t, ref_a,
                                   test_ch.time_us, test_ch.amplitude)
            err = float(np.max(np.abs(r - t))) if len(r) > 0 else 0.0
            tr_err[ch] = err
            if err > grad_atol:
                pfx = f'TR {tr_idx}: ' if multi_tr else ''
                messages.append(
                    f'{pfx}{ch} mismatch: max diff {err:.4f} mT/m '
                    f'(tol {grad_atol:.4f} mT/m)')

        # Compare RF
        ref_t, ref_a = ref['rf_mag']
        r, t = _interp_to_ref(ref_t, ref_a,
                               wf.rf_mag.time_us, wf.rf_mag.amplitude)
        rf_err = _rms_error(r, t)
        tr_err['rf_mag'] = rf_err
        if rf_err > rf_rms_percent:
            pfx = f'TR {tr_idx}: ' if multi_tr else ''
            messages.append(
                f'{pfx}RF mismatch: {rf_err:.1f}% RMS '
                f'(tol {rf_rms_percent:.1f}%)')

        errors_per_tr.append(tr_err)

    # Aggregate: worst-case across TRs
    errors: dict[str, float] = {}
    if errors_per_tr:
        for key in errors_per_tr[0]:
            errors[key] = max(e[key] for e in errors_per_tr)

    ok = len(messages) == 0

    # ── Optional overlay plot ────────────────────────────────
    if do_plot:
        _validation_plot(
            seq,
            sequence_idx=sequence_idx,
            xml_path=xml_path,
            tr_idx=tr_range[0],
            hide_prep=hide_prep,
            hide_cooldown=hide_cooldown,
            show_rf_centers=show_rf_centers,
            show_echoes=show_echoes,
            show_segments=show_segments,
            show_blocks=show_blocks,
            max_grad_mT_per_m=_max_grad_plot,
            ok=ok,
            messages=messages,
        )

    return {'ok': ok, 'errors': errors, 'messages': messages}


# ── validation plot (private) ────────────────────────────────────────

def _validation_plot(
    seq, *, sequence_idx, xml_path, tr_idx,
    hide_prep, hide_cooldown,
    show_rf_centers, show_echoes,
    show_segments, show_blocks,
    max_grad_mT_per_m,
    ok, messages,
):
    """Create the comparison overlay using :func:`_plot.plot`."""
    from ._plot import plot as _plot_impl

    # Branch 1: C-backend base figure
    handle = _plot_impl(
        seq,
        subsequence_idx=sequence_idx,
        tr_idx=tr_idx,
        hide_prep=hide_prep,
        hide_cooldown=hide_cooldown,
        collapse_delays=False,
        show_segments=show_segments,
        show_blocks=show_blocks,
        show_slew=False,
        show_rf_centers=show_rf_centers,
        show_echoes=show_echoes,
        max_grad_mT_per_m=max_grad_mT_per_m,
        max_slew_T_per_m_per_s=None,
    )

    # Branch 2 (pypulseq) or Branch 3 (XML) overlay
    label = 'XML' if xml_path is not None else 'pypulseq'
    overlay_source = (str(xml_path) if xml_path is not None
                      else seq._seqs[sequence_idx])
    _plot_impl(overlay_source, fig=handle, label=label)

    # Annotate pass / fail
    status = 'PASS' if ok else 'FAIL'
    colour = 'green' if ok else 'red'
    handle.fig.suptitle(f'validate()  \u2014  {status}',
                        fontweight='bold', color=colour)
    if messages:
        handle.fig.text(0.5, 0.01, '\n'.join(messages),
                        ha='center', fontsize=8, color='red')
    handle.fig.tight_layout(rect=[0, 0.05 if messages else 0, 1, 0.95])
