""" """

__all__ = [
    "PulserverSequence", 
    "get_unique_blocks", 
    "find_tr", 
    "find_segments_in_tr", 
    "get_tr_gradient_waveforms", 
    "get_tr_acoustic_spectra",
    "get_pns",
]

import copy
import warnings

from types import SimpleNamespace

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
    
import pypulseq as pp

from ._extension._pulseqlib_wrapper import (
    _find_tr_in_sequence,
    _find_segments_in_tr,
    _get_unique_blocks,
    _get_tr_gradient_waveforms,
    _get_tr_acoustic_spectra,
    _get_pns,  # Add this import
    _PulserverSeqFile,
)
from ._iostream import write_to_stream


def _make_diagnostic(diag_dict: dict) -> SimpleNamespace:
    """Create a diagnostic SimpleNamespace from a dict."""
    diag = SimpleNamespace(
        code=diag_dict["code"],
        message=diag_dict["message"],
        hint=diag_dict["hint"],
        block_index=diag_dict["block_index"],
        channel=diag_dict["channel"],
        num_unique_blocks=diag_dict["num_unique_blocks"],
        imaging_region_length=diag_dict["imaging_region_length"],
        candidate_pattern_length=diag_dict["candidate_pattern_length"],
        mismatch_position=diag_dict["mismatch_position"],
    )
    # Add convenience property
    diag.success = diag.code > 0
    return diag


def _format_diagnostic(diag: SimpleNamespace) -> str:
    """Format diagnostic info as a human-readable string."""
    if diag.success:
        return "Success"
    
    lines = [f"Error: {diag.message}"]
    
    if diag.block_index >= 0:
        lines.append(f"  Block index: {diag.block_index}")
    if diag.mismatch_position >= 0:
        lines.append(f"  Mismatch at position: {diag.mismatch_position}")
    if diag.num_unique_blocks > 0:
        lines.append(f"  Unique blocks found: {diag.num_unique_blocks}")
    if diag.imaging_region_length > 0:
        lines.append(f"  Imaging region length: {diag.imaging_region_length}")
    if diag.candidate_pattern_length > 0:
        lines.append(f"  Best candidate pattern length: {diag.candidate_pattern_length}")
    
    if diag.hint:
        lines.append(f"\nHint: {diag.hint}")
    
    return "\n".join(lines)


class SequenceAnalysisError(Exception):
    """Exception raised when sequence analysis fails."""
    
    def __init__(self, diagnostic: SimpleNamespace):
        self.diagnostic = diagnostic
        super().__init__(_format_diagnostic(diagnostic))


class PulserverSequence(pp.Sequence):
    """ """

    def __init__(self, seq: pp.Sequence):
        object.__setattr__(self, '_seq', copy.deepcopy(seq))
        sys = seq.system
        cseq = _PulserverSeqFile(
            write_to_stream(seq),
            float(sys.gamma),
            float(sys.B0),
            float(sys.max_grad),
            float(sys.max_slew),
            float(sys.rf_raster_time),
            float(sys.grad_raster_time),
            float(sys.adc_raster_time),
            float(sys.block_duration_raster),
        )
        object.__setattr__(self, '_cseq', cseq)

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self._seq, name)

    def __setattr__(self, name, value):
        if name in ('_seq', '_cseq'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._seq, name, value)

    def __str__(self):
        return str(self._seq)


def get_unique_blocks(seq: PulserverSequence) -> SimpleNamespace:
    """
    Get unique blocks and event deduplication info from a sequence.

    Parameters
    ----------
    seq : PulserverSequence
        Input sequence.

    Returns
    -------
    SimpleNamespace
        Result containing:
        - num_blocks: total number of blocks in the sequence
        - num_unique_blocks: number of unique block definitions
        - block_table: list of per-block info (maps block index -> block metadata)
        - block_definitions: list of unique block definitions
        - rf_definitions: list of unique RF definitions
        - rf_table: list mapping RF index -> unique RF params
        - grad_definitions: list of unique gradient definitions, each containing:
            - id, type, delay (all types)
            - rise_time, flat_time, fall_time (trapezoids only)
            - num_shots, num_samples, time_shape_id (arbitrary only)
            - max_amplitude: max |amplitude| across instances per shot (Hz/m)
            - slew_rate: max |d(waveform)/dt| per shot (1/s, normalized)
            - energy: integral of waveform² dt per shot (s, normalized)
            - shot_shape_ids, first_value, last_value (arbitrary only)
        - grad_table: list mapping gradient index -> unique gradient params
        - adc_definitions: list of unique ADC definitions
        - adc_table: list mapping ADC index -> unique ADC params
        - tr_descriptor: TR structure info (prep/cooldown blocks)
        
    Notes
    -----
    A unique block is determined by the tuple of unique (rf, gx, gy, gz),
    i.e., the tuple of blocks whose rf and gradient events has the same normalized
    waveforms on all channels. For gradients, waveforms are uniquely determined by
    class (trapezoids vs arbitrary grads) and timing.

    """
    result_dict = _get_unique_blocks(seq._cseq)
    
    if not result_dict["success"]:
        raise RuntimeError(f"Failed to get unique blocks: {result_dict.get('error', 'unknown error')}")
    
    result = SimpleNamespace(
        num_blocks=result_dict["num_blocks"],
        num_unique_blocks=result_dict["num_unique_blocks"],
        block_table=result_dict["block_table"],
        block_definitions=result_dict["block_definitions"],
        num_unique_rfs=result_dict["num_unique_rfs"],
        rf_definitions=result_dict["rf_definitions"],
        rf_table=result_dict["rf_table"],
        num_unique_grads=result_dict["num_unique_grads"],
        grad_definitions=result_dict["grad_definitions"],
        grad_table=result_dict["grad_table"],
        num_unique_adcs=result_dict["num_unique_adcs"],
        adc_definitions=result_dict["adc_definitions"],
        adc_table=result_dict["adc_table"],
        tr_descriptor=SimpleNamespace(**result_dict["tr_descriptor"]),
    )
    
    return result


def find_tr(seq: PulserverSequence, num_reps: int = 1, raise_on_error: bool = True) -> SimpleNamespace:
    """
    Find TR structure in a sequence.
    
    Parameters
    ----------
    seq : PulserverSequence
        The sequence to analyze.
    num_reps : int
        Number of repetitions (for output structure).
    raise_on_error : bool
        If True, raise SequenceAnalysisError on failure.
        If False, return result with diagnostic info.
    
    Returns
    -------
    SimpleNamespace
        Result containing:
        - main_tr: The main TR as a pp.Sequence
        - first_rep_first_tr: First TR with prep blocks (or None)
        - last_rep_last_tr: Last TR with cooldown blocks (or None)
    
    Raises
    ------
    SequenceAnalysisError
        If raise_on_error=True and TR detection fails.
    """
    # Call the C wrapper which internally does getUniqueBlocks + findTRInSequence
    tr_result = _find_tr_in_sequence(seq._cseq)
    
    diagnostic = _make_diagnostic(tr_result["diagnostic"])
    
    # Check for failure
    if not tr_result["success"]:
        if raise_on_error:
            raise SequenceAnalysisError(diagnostic)
        else:
            result = SimpleNamespace()
            result.main_tr = None
            result.first_rep_first_tr = None
            result.last_rep_last_tr = None
            result.diagnostic = diagnostic
            return result
    
    tr_size = tr_result["tr_size"]
    num_trs = tr_result["num_trs"]
    num_prep = tr_result["num_prep_blocks"]
    num_cooldown = tr_result["num_cooldown_blocks"]
    degenerate_prep = tr_result["degenerate_prep"]
    degenerate_cooldown = tr_result["degenerate_cooldown"]
    
    # Prepare result
    result = SimpleNamespace()
        
    # Special case: only one TR
    if num_reps == 1 and num_trs == 1:
        result.main_tr = seq._seq
        result.first_rep_first_tr = None
        result.last_rep_last_tr = None
        return result
        
    # Prepare main TR
    result.main_tr = pp.Sequence(system=seq.system)
    
    # Prepare first and last repetitions header and footer blocks
    if degenerate_prep or num_prep == 0:
        result.first_rep_first_tr = None
    else:
        result.first_rep_first_tr = pp.Sequence(system=seq.system)
    if degenerate_cooldown or num_cooldown == 0:
        result.last_rep_last_tr = None
    else:
        result.last_rep_last_tr = pp.Sequence(system=seq.system)
        
    # Add first repetition header blocks
    if result.first_rep_first_tr is not None:
        for n in range(num_prep):
            block = seq._seq.get_block(n + 1)
            result.first_rep_first_tr.add_block(block)
    
    # Add main TR blocks            
    for n in range(num_prep, num_prep + tr_size):
        block = seq._seq.get_block(n + 1)
        result.main_tr.add_block(block)
        if result.first_rep_first_tr is not None:
            result.first_rep_first_tr.add_block(block)
        if result.last_rep_last_tr is not None:
            result.last_rep_last_tr.add_block(block)
    
    # Add last repetition footer blocks
    if result.last_rep_last_tr is not None:
        for n in range(len(seq.block_events) - num_cooldown, len(seq.block_events)):
            block = seq._seq.get_block(n + 1)
            result.last_rep_last_tr.add_block(block)
                 
    return result


def find_segments_in_tr(seq: PulserverSequence, raise_on_error: bool = True) -> tuple[list[pp.Sequence], SimpleNamespace]:
    """
    Find segment definitions within the TR structure of a sequence.
    
    Parameters
    ----------
    seq : PulserverSequence
        The sequence to analyze.
    raise_on_error : bool
        If True, raise SequenceAnalysisError on failure.
        If False, return result with diagnostic info.
        
    Returns
    -------
    list[pp.Sequence]: 
        List of pp.Sequences representing the unique sequence Segments.
    SimpleNamespace
        Result containing:
        - prep_segment_table: Maps prep section segments to unique segment IDs
        - main_segment_table: Maps main TR segments to unique segment IDs
        - cooldown_segment_table: Maps cooldown section segments to unique segment IDs
    """
    # Call the C wrapper which internally does getUniqueBlocks + findTR + findSegments
    raw_result = _find_segments_in_tr(seq._cseq)
    
    # Check for failure
    if not raw_result.get("success", True):
        if raise_on_error:
            if "diagnostic" in raw_result:
                diagnostic = _make_diagnostic(raw_result["diagnostic"])
                raise SequenceAnalysisError(diagnostic)
            else:
                raise RuntimeError(raw_result.get("error", "Unknown error in find_segments_in_tr"))
        result = SimpleNamespace()
        result.prep_segment_table = []
        result.main_segment_table = []
        result.cooldown_segment_table = []
        return [], result
    
    # Build pp.Sequence for each unique segment
    unique_segments = []
    for seg_dict in raw_result["unique_segments"]:
        start = seg_dict["start_block"]
        count = seg_dict["num_blocks"]
        
        # Build a pp.Sequence for this segment
        segment_seq = pp.Sequence(system=seq.system)
        for n in range(start, start + count):
            block = seq._seq.get_block(n + 1)
            segment_seq.add_block(block)
        unique_segments.append(segment_seq)
    
    result = SimpleNamespace()
    result.prep_segment_table = raw_result["prep_segment_table"]
    result.main_segment_table = raw_result["main_segment_table"]
    result.cooldown_segment_table = raw_result["cooldown_segment_table"]
    
    return unique_segments, result


def get_tr_gradient_waveforms(seq: PulserverSequence) -> SimpleNamespace:
    """
    Extract concatenated gradient waveforms for a single TR.
    
    Parameters
    ----------
    seq : PulserverSequence
        The sequence to analyze.
        
    Returns
    -------
    SimpleNamespace
        Result containing:
        - time: np.ndarray of time points (microseconds)
        - waveform_gx: np.ndarray of Gx amplitude (Hz/m)
        - waveform_gy: np.ndarray of Gy amplitude (Hz/m)
        - waveform_gz: np.ndarray of Gz amplitude (Hz/m)
        
    Notes
    -----
    Timing conventions:
    - Trapezoids: corner points (0, rise, rise+flat, rise+flat+fall)
    - Extended trapezoids: samples at raster edges (from time shape)
    - Arbitrary gradients: samples at raster centers (0.5*raster, 1.5*raster, ...)
    
    The waveforms are concatenated across all blocks in the TR, with time
    offsets adjusted so each block's waveform starts where the previous ended.
    """    
    result_dict = _get_tr_gradient_waveforms(seq._cseq)
    
    if not result_dict["success"]:
        raise RuntimeError(f"Failed to get TR gradient waveforms: {result_dict.get('error', 'unknown error')}")
    
    result = SimpleNamespace(
        time=np.asarray(result_dict["time_gx"], dtype=np.float32),
        waveform_gx=np.asarray(result_dict["waveform_gx"], dtype=np.float32),
        waveform_gy=np.asarray(result_dict["waveform_gy"], dtype=np.float32),
        waveform_gz=np.asarray(result_dict["waveform_gz"], dtype=np.float32),
    )
    
    return result


def get_tr_acoustic_spectra(
    seq: PulserverSequence, 
    window_duration: float = 25.0e-3,
    spectral_resolution: float = 5.0,
    max_frequency: float = 3000.0,
    combined: bool = False,
    forbidden_bands: list[dict] | None = None,
    do_plot: bool = False,
) -> SimpleNamespace:
    """
    Compute acoustic spectra for gradient waveforms in a TR.
    
    Performs sliding-window FFT analysis on each gradient axis to identify
    potential acoustic resonance frequencies, plus full TR and N-TR sequence spectra.
    
    Parameters
    ----------
    seq : PulserverSequence
        The sequence to analyze.
    window_duration : float
        Target window size in seconds for sliding window analysis. Default is 0.05 s.
    spectral_resolution : float
        Target frequency resolution in Hz. Default is 5.0 Hz.
        FFT size is automatically chosen via zero-padding to achieve approximately
        this resolution.
    max_frequency : float
        Maximum frequency to include in output (Hz). Default is 3000.0 Hz.
        If None, the full spectrum up to Nyquist is returned.
    combined : bool
        If True, return pointwise maximum across all windows (1D arrays).
        If False, stack all windows (2D arrays). Default is False.
    forbidden_bands : list[dict] | None
        Optional list of forbidden frequency bands for acoustic resonance check.
        Each dict should contain:
        - 'freq_min_hz': float, minimum frequency of the band (Hz)
        - 'freq_max_hz': float, maximum frequency of the band (Hz)
        - 'max_amplitude': float, maximum allowed gradient amplitude (Hz/m)
        If None or empty list, no acoustic check is performed.
    do_plot : bool
        If True, plot the computed spectra. Default is False.
        
    Returns
    -------
    SimpleNamespace
        Result containing:
        
        Sliding window spectra:
        - frequencies: np.ndarray of shape (num_freq_bins,), frequency values in Hz
        - spectra_gx: np.ndarray, Gx spectra
            - If combined=False: shape (num_windows, num_freq_bins)
            - If combined=True: shape (num_freq_bins,)
        - spectra_gy: np.ndarray, Gy spectra (same shape as spectra_gx)
        - spectra_gz: np.ndarray, Gz spectra (same shape as spectra_gx)
        - max_envelope_gx: np.ndarray of shape (num_freq_bins,), max gradient envelope (mT/m)
        - max_envelope_gy: np.ndarray of shape (num_freq_bins,), max gradient envelope (mT/m)
        - max_envelope_gz: np.ndarray of shape (num_freq_bins,), max gradient envelope (mT/m)
        - peaks_gx: np.ndarray of shape (num_windows, num_freq_bins) or None
            - Binary array indicating detected peaks along X (None if combined=True)
        - peaks_gy: np.ndarray of shape (num_windows, num_freq_bins) or None
            - Binary array indicating detected peaks along Y (None if combined=True)
        - peaks_gz: np.ndarray of shape (num_windows, num_freq_bins) or None
            - Binary array indicating detected peaks along Z (None if combined=True)
                
        N-TR sequence spectra (discrete lines at harmonics, only if numTRs > 1):
        - num_trs: Number of TR repetitions
        - tr_duration_us: TR duration in microseconds
        - fundamental_freq: Fundamental frequency (1/TR) in Hz
        - frequencies_seq: np.ndarray of harmonic frequencies (k * f0)
        - spectrum_gx_seq: np.ndarray, Gx sequence spectrum at harmonics
        - spectrum_gy_seq: np.ndarray, Gy sequence spectrum at harmonics
        - spectrum_gz_seq: np.ndarray, Gz sequence spectrum at harmonics
        - max_envelope_gx_seq: np.ndarray of shape (num_harmonics,), max gradient envelope (mT/m)
        - max_envelope_gy_seq: np.ndarray of shape (num_harmonics,), max gradient envelope (mT/m)
        - max_envelope_gz_seq: np.ndarray of shape (num_harmonics,), max gradient envelope (mT/m)
        - peaks_gx_seq: np.ndarray of shape (num_harmonics,) or None
            - Binary array indicating detected peaks along X
        - peaks_gy_seq: np.ndarray of shape (num_harmonics,) or None
            - Binary array indicating detected peaks along Y
        - peaks_gz_seq: np.ndarray of shape (num_harmonics,) or None
            - Binary array indicating detected peaks along Z
    
    Notes
    -----
    The analysis performs three types of spectral analysis:
    
    1. **Sliding window spectra**: Shows temporal evolution of frequencies
       - Extract window samples (50% overlap between windows)
       - Zero-pad to FFT size determined by target spectral resolution
       - Subtract mean (DC removal)
       - Apply cosine taper window
       - Compute magnitude spectrum via real FFT
           
    2. **N-TR sequence spectrum**: Spectral lines at harmonics of 1/TR (discrete)
       - Only computed if numTRs > 1
       - Samples complex FFT at harmonic frequencies k*f0 where f0 = 1/TR
       - Uses linear interpolation of complex FFT before taking magnitude
       - Represents the steady-state spectrum of the repeated TR pattern
    
    When combined=True, the sliding window output is the pointwise maximum magnitude
    across all windows, useful for identifying the worst-case acoustic excitation.
    
    Examples
    --------
    >>> # Get all spectra with 5 Hz resolution up to 3000 Hz
    >>> result = get_tr_acoustic_spectra(seq)
    >>> 
    >>> # Plot sliding window max vs full TR vs sequence spectrum
    >>> plt.plot(result.frequencies, result.spectra_gx.max(axis=0), label='Sliding window max')
    >>> plt.plot(result.frequencies_full, result.spectrum_gx_full, label='Full TR')
    >>> if hasattr(result, 'frequencies_seq'):
    >>>     plt.stem(result.frequencies_seq, result.spectrum_gx_seq, label='Sequence (harmonics)')
    >>> plt.legend()
    """
    if max_frequency is None:
        max_frequency = -1.0  # Use -1 to indicate full spectrum in C wrapper
    
    if forbidden_bands is None:
        forbidden_bands = []
        
    # Get gamma and gradient raster from sequence system
    gamma = seq.system.gamma  # in Hz/T
    grad_raster_time = seq.system.grad_raster_time  # in seconds
    
    # Compute window size
    target_window_size = int(2.0 * window_duration / grad_raster_time)
    
    # Run the C++ extension
    result_dict = _get_tr_acoustic_spectra(
        seq._cseq, 
        target_window_size, 
        spectral_resolution, 
        max_frequency, 
        combined,
        forbidden_bands,
        True,
    )
    
    if not result_dict["success"]:
        raise RuntimeError(f"Acoustic spectra computation failed: {result_dict.get('error', 'Unknown error')}")
    
    num_windows = result_dict["num_windows"]
    num_freq_bins = result_dict["num_freq_bins"]
    is_combined = bool(result_dict["combined"])
    
    # Reshape sliding window spectra based on combined flag
    if is_combined:
        spectra_gx = np.asarray(result_dict["spectra_gx"], dtype=np.float32)
        spectra_gy = np.asarray(result_dict["spectra_gy"], dtype=np.float32)
        spectra_gz = np.asarray(result_dict["spectra_gz"], dtype=np.float32)
    else:
        spectra_gx = np.asarray(result_dict["spectra_gx"], dtype=np.float32).reshape(num_windows, num_freq_bins)
        spectra_gy = np.asarray(result_dict["spectra_gy"], dtype=np.float32).reshape(num_windows, num_freq_bins)
        spectra_gz = np.asarray(result_dict["spectra_gz"], dtype=np.float32).reshape(num_windows, num_freq_bins)
    
    result = SimpleNamespace(
        # Sliding window spectra
        frequencies=np.asarray(result_dict["frequencies"], dtype=np.float32),
        spectra_gx=spectra_gx,
        spectra_gy=spectra_gy,
        spectra_gz=spectra_gz,
        max_envelope_gx=np.asarray(result_dict["max_envelope_gx"], dtype=np.float32) / gamma * 1000.0, # mT/m
        max_envelope_gy=np.asarray(result_dict["max_envelope_gy"], dtype=np.float32) / gamma * 1000.0, # mT/m
        max_envelope_gz=np.asarray(result_dict["max_envelope_gz"], dtype=np.float32) / gamma * 1000.0, # mT/m
    )
    
    # Add peak arrays for sliding window (only if not combined)
    if not is_combined and "peaks_gx" in result_dict:
        peaks_gx = np.asarray(result_dict["peaks_gx"], dtype=np.int32).reshape(num_windows, num_freq_bins)
        peaks_gy = np.asarray(result_dict["peaks_gy"], dtype=np.int32).reshape(num_windows, num_freq_bins)
        peaks_gz = np.asarray(result_dict["peaks_gz"], dtype=np.int32).reshape(num_windows, num_freq_bins)
        result.peaks_gx = peaks_gx
        result.peaks_gy = peaks_gy
        result.peaks_gz = peaks_gz
    else:
        result.peaks_gx = None
        result.peaks_gy = None
        result.peaks_gz = None
    
    # Add sequence spectra if present (only when numTRs > 1)
    if "frequencies_seq" in result_dict:
        result.num_trs = result_dict["num_trs"]
        result.tr_duration_us = result_dict["tr_duration_us"]
        result.fundamental_freq = result_dict["fundamental_freq"]
        result.frequencies_seq = np.asarray(result_dict["frequencies_seq"], dtype=np.float32)
        result.spectrum_gx_seq = np.asarray(result_dict["spectra_gx_seq"], dtype=np.float32)
        result.spectrum_gy_seq = np.asarray(result_dict["spectra_gy_seq"], dtype=np.float32)
        result.spectrum_gz_seq = np.asarray(result_dict["spectra_gz_seq"], dtype=np.float32)
        result.max_envelope_gx_seq=result_dict["max_envelope_gx_full"] / gamma * 1000.0, # mT/m
        result.max_envelope_gy_seq=result_dict["max_envelope_gy_full"] / gamma * 1000.0, # mT/m
        result.max_envelope_gz_seq=result_dict["max_envelope_gz_full"] / gamma * 1000.0, # mT/m
        result.peaks_gx_seq = np.asarray(result_dict["peaks_gx_seq"], dtype=np.int32)
        result.peaks_gy_seq = np.asarray(result_dict["peaks_gy_seq"], dtype=np.int32)
        result.peaks_gz_seq = np.asarray(result_dict["peaks_gz_seq"], dtype=np.int32)
        
    if do_plot:
        _plot_acoustic_spectra(result, seq=seq, forbidden_bands=forbidden_bands)
        
    # Acoustic check
    if forbidden_bands:
        _check_acoustic_forbidden_bands(result, seq, forbidden_bands)

    return result


def get_pns(
    seq: PulserverSequence,
    chronaxie_us: float | None = None,
    rheobase: float | None = None,
    alpha: float | None = None,
    do_plot: bool = False,
) -> SimpleNamespace:
    """
    Compute Peripheral Nerve Stimulation (PNS) levels for gradient waveforms.
    
    Uses convolution with a vendor-specific nerve response kernel to estimate
    PNS as a percentage of the stimulation threshold. Circular padding is 
    automatically applied based on the kernel length.
    
    Parameters
    ----------
    seq : PulserverSequence
        The sequence to analyze.
    chronaxie_us : float | None
        Chronaxie time constant in microseconds. Required for GE model.
        Typical value: ~360 µs (IEC 60601-2-33:2022).
    rheobase : float | None
        Rheobase - minimum slew rate for stimulation in T/m/s.
        Required for GE model. Typical value: ~20 T/m/s.
    alpha : float | None
        Effective coil length in meters. Required for GE model.
        The stimulation threshold Smin = rheobase / alpha.
        Typical value: ~0.333 m (IEC 60601-2-33:2022).
    do_plot : bool
        If True, plot the computed spectra. Default is False.
        
    Returns
    -------
    SimpleNamespace
        Result containing:
        - max_pns: Maximum PNS value (%) - values > 100% indicate stimulation
        - max_pns_index: Sample index of maximum
        - max_pns_time_us: Time of maximum in microseconds
        - num_samples: Number of output samples
        
        If store_waveforms=True, also includes:
        - pns_total: np.ndarray of combined PNS waveform sqrt(X² + Y² + Z²)
        - pns_x, pns_y, pns_z: np.ndarray of per-axis PNS waveforms
        
    Raises
    ------
    NotImplementedError
        If called with non-GE vendor (currently only GE/GEHC is supported).
    RuntimeError
        If PNS computation fails.
    ValueError
        If required GE parameters are not provided.
        
    Notes
    -----
    **Currently Supported: GE/GEHC Model**
    
    Uses IEC 60601-2-33:2022 Eq. AA.21. The nerve response kernel is:
    
    h(tau) = (dt / Smin) * c / (c + tau)^2
    
    where:
    - c = chronaxie (µs)
    - Smin = rheobase / alpha (T/m/s)
    
    PNS is computed as the convolution of the gradient slew rate with this kernel,
    with circular padding automatically applied based on kernel length.
    
    **Not Yet Implemented:**
    - Siemens model (tau parameter)
    - Philips vendor
    - United Imaging vendor
    - Bruker vendor
    
    Examples
    --------
    >>> # GE model
    >>> result = get_pns(seq, chronaxie_us=360.0, rheobase=20.0, alpha=0.333)
    >>> print(f"Max PNS: {result.max_pns:.1f}%")
    >>> if result.max_pns > 100.0:
    ...     print("Warning: PNS threshold exceeded!")
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(result.pns_total)
    >>> plt.axhline(100.0, color='r', linestyle='--', label='Threshold')
    >>> plt.show()
    """        
    # GE model
    if chronaxie_us is None or rheobase is None or alpha is None:
        raise ValueError(
            "GE PNS model requires 'chronaxie_us', 'rheobase', and 'alpha' parameters. "
            "Typical values: chronaxie_us=360.0, rheobase=20.0, alpha=0.333"
        )
    
    try:
        result_dict = _get_pns(
            seq._cseq,
            100.0, # pns_threshold (ignored when store_waveforms=True)
            chronaxie_us,
            rheobase,
            alpha,
            True, # store_waveforms
        )
    except RuntimeError as e:
        # Check if it's a vendor not implemented error
        if "PNS computation is currently only implemented for GE/GEHC" in str(e):
            raise NotImplementedError(str(e))
        raise
    
    if not result_dict["success"]:
        raise RuntimeError(f"PNS computation failed: {result_dict.get('error', 'Unknown error')}")
    
    result = SimpleNamespace(
        max_pns=result_dict["max_pns"],
        max_pns_index=result_dict["max_pns_index"],
        max_pns_time_us=result_dict["max_pns_time_us"],
        num_samples=result_dict["num_samples"],
    )
    
    # Add waveforms if stored
    if "pns_total" in result_dict:
        result.pns_total = np.asarray(result_dict["pns_total"], dtype=np.float32)
        if "pns_x" in result_dict:
            result.pns_x = np.asarray(result_dict["pns_x"], dtype=np.float32)
        if "pns_y" in result_dict:
            result.pns_y = np.asarray(result_dict["pns_y"], dtype=np.float32)
        if "pns_z" in result_dict:
            result.pns_z = np.asarray(result_dict["pns_z"], dtype=np.float32)
            
    if do_plot:
        _plot_pns(result, seq)
        
    # Check PNS thresholds
    _check_pns_thresholds(result)
    
    return result


def _plot_acoustic_spectra(
    spectra: SimpleNamespace,
    seq: PulserverSequence | None = None,
    forbidden_bands: list[dict] | None = None,
) -> tuple:
    """
    Plot acoustic spectra with waveforms and sliding windows.
    
    Creates a comprehensive visualization of acoustic spectra including:
    - Full sequence spectrum with waveforms
    - Sliding window spectra matrix (if combined=False)
    - Detected peaks and forbidden frequency bands
    
    Parameters
    ----------
    spectra : SimpleNamespace
        Output from get_tr_acoustic_spectra() with combined=False.
    seq : PulserverSequence | None
        The sequence object (needed to get waveforms and system parameters).
        If None, waveform panel is skipped.
    forbidden_bands : list[dict] | None
        List of forbidden frequency bands with keys:
        - 'freq_min_hz': float, minimum frequency (Hz)
        - 'freq_max_hz': float, maximum frequency (Hz)
        - 'max_amplitude': float (optional, for reference)
        
    Returns
    -------
    tuple
        (fig, axes) - Matplotlib figure and axes for further customization.
        
    Raises
    ------
    ValueError
        If spectra is from combined=True mode (needs per-window data).
    RuntimeError
        If waveforms requested but seq not provided.
        
    Notes
    -----
    - Top panel: Full spectrum with overlaid waveforms
    - Bottom panel: Sliding window spectrograms (3 subplots for X, Y, Z)
    - Peaks marked with red stars in spectra
    - Forbidden bands shown as black dashed lines (full spectrum) or white dashed lines (windows)
    - Dual x-axis: frequency (Hz) and corresponding echo spacing (µs) = 1/(2*f)
    """    
    # Validate input
    if not hasattr(spectra, 'spectra_gx') or not hasattr(spectra, 'peaks_gx'):
        raise ValueError("spectra must be from get_tr_acoustic_spectra() with combined=False")
    
    if spectra.peaks_gx is None:
        raise ValueError("Spectra must have peaks detected (combined=False)")
    
    if spectra.spectra_gx.ndim != 2:
        num_windows = 1
    else:
        num_windows = spectra.spectra_gx.shape[0]
    
    # Get waveforms if seq provided
    waveforms = None
    if seq is not None:
        try:
            waveforms = get_tr_gradient_waveforms(seq)
            gamma = seq.system.gamma
        except Exception as e:
            print(f"Warning: Could not get waveforms: {e}")
    
    # Create figure
    fig = plt.figure()
    
    # Determine frequency range
    freq_min = 0.0
    freq_max = spectra.frequencies[-1]
    
    # ============ Panel 1: Full Sequence + Waveforms ============
    ax_seq_spec = plt.subplot(2, 1, 1)
    
    # Plot spectra
    colors = {'x': 'C0', 'y': 'C1', 'z': 'C2'}
    labels = {'x': 'Gx', 'y': 'Gy', 'z': 'Gz'}
    
    for axis, (axis_name, color) in enumerate(colors.items()):
        spec_attr = f'spectra_g{axis_name}'
        peaks_attr = f'peaks_g{axis_name}'
        
        spectra_axis = getattr(spectra, spec_attr)
        peaks_axis = getattr(spectra, peaks_attr, None)
        
        # Max across all windows for this axis
        spec_max = spectra_axis.max(axis=0)
        
        # Plot full spectrum
        ax_seq_spec.plot(spectra.frequencies, spec_max, color=color, linewidth=2, label=labels[axis_name])
        
        # Plot detected peaks
        if peaks_axis is not None:
            peaks_any = peaks_axis.max(axis=0) > 0  # Any window detected a peak
            peak_freqs = spectra.frequencies[peaks_any]
            peak_mags = spec_max[peaks_any]
            ax_seq_spec.plot(peak_freqs, peak_mags, 
                           marker='o', color=color, linestyle='none', 
                           markersize=6.5, markerfacecolor=color, 
                           markeredgecolor=color, markeredgewidth=0)
    
    # Add forbidden bands
    if forbidden_bands:
        for band in forbidden_bands:
            freq_min_band = band['freq_min_hz']
            freq_max_band = band['freq_max_hz']
            ax_seq_spec.axvline(freq_min_band, color='black', linestyle='--', 
                              linewidth=1.5, alpha=0.7)
            ax_seq_spec.axvline(freq_max_band, color='black', linestyle='--', 
                              linewidth=1.5, alpha=0.7)
    
    # Force x-axis limits
    ax_seq_spec.set_xlim(freq_min, freq_max)
    ax_seq_spec.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_seq_spec.set_ylabel('Magnitude [a.u.]', fontsize=12)
    ax_seq_spec.set_title('Full Sequence Acoustic Spectrum', fontsize=14, fontweight='bold')
    ax_seq_spec.legend(loc='upper right', fontsize=11)
    ax_seq_spec.grid(True, alpha=0.3)
    
    # Add secondary x-axis (echo spacing in µs)
    _add_echo_spacing_axis(ax_seq_spec, freq_min, freq_max)
    
    # ============ Panel 2: Waveforms (always in first figure) ============
    if waveforms is not None:
        ax_wf = plt.subplot(2, 1, 2)
        
        # Convert waveforms to mT/m
        wf_gx_mtpm = waveforms.waveform_gx / gamma * 1000.0
        wf_gy_mtpm = waveforms.waveform_gy / gamma * 1000.0
        wf_gz_mtpm = waveforms.waveform_gz / gamma * 1000.0
        
        # Convert time to ms
        time_ms = waveforms.time / 1000.0
        
        ax_wf.plot(time_ms, wf_gx_mtpm, color=colors['x'], 
                  linewidth=2, label='Gx')
        ax_wf.plot(time_ms, wf_gy_mtpm, color=colors['y'], 
                  linewidth=2, label='Gy')
        ax_wf.plot(time_ms, wf_gz_mtpm, color=colors['z'], 
                  linewidth=2, label='Gz')
        
        ax_wf.set_xlabel('Time (ms)', fontsize=12)
        ax_wf.set_ylabel('Gradient Amplitude (mT/m)', fontsize=12)
        ax_wf.set_title('TR Gradient Waveforms', fontsize=14, fontweight='bold')
        ax_wf.legend(loc='upper right', fontsize=11)
        ax_wf.grid(True, alpha=0.3)
    
    plt.figure(fig.number)  # Ensure we're still on fig
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    # ============ Panel 3: Sliding Window Spectrograms (separate figure if num_windows > 1) ============
    fig2 = None
    if num_windows > 1:
        fig2 = plt.figure()

        # Create three subplots for X, Y, Z
        axes_windows = []
        im = None
        
        for idx, (axis_name, color) in enumerate(colors.items()):
            ax = plt.subplot(1, 3, 1 + idx)
            axes_windows.append(ax)
            
            spec_attr = f'spectra_g{axis_name}'
            peaks_attr = f'peaks_g{axis_name}'
            
            spectra_matrix = getattr(spectra, spec_attr)
            peaks_matrix = getattr(spectra, peaks_attr, None)
            
            # Create spectrogram
            im = ax.pcolormesh(spectra.frequencies, np.arange(num_windows), 
                              spectra_matrix, cmap='viridis', shading='auto',
                              norm=Normalize(vmin=spectra_matrix.min(), 
                                           vmax=spectra_matrix.max()))
            
            # Overlay peaks with red asterisks
            if peaks_matrix is not None:
                peak_coords = np.where(peaks_matrix)
                if len(peak_coords[0]) > 0:
                    window_indices = peak_coords[0]
                    freq_indices = peak_coords[1]
                    peak_freqs = spectra.frequencies[freq_indices]
                    ax.plot(peak_freqs, window_indices, 
                           marker='*', color='red', linestyle='none',
                           markersize=16, markerfacecolor='red', 
                           markeredgecolor='red', markeredgewidth=0)
            
            # Add forbidden bands as white dashed lines
            if forbidden_bands:
                for band in forbidden_bands:
                    freq_min_band = band['freq_min_hz']
                    freq_max_band = band['freq_max_hz']
                    ax.axvline(freq_min_band, color='white', linestyle='--', 
                             linewidth=1.5, alpha=0.8)
                    ax.axvline(freq_max_band, color='white', linestyle='--', 
                             linewidth=1.5, alpha=0.8)
            
            # Force x-axis limits
            ax.set_xlim(freq_min, freq_max)
            ax.set_xlabel('Frequency (Hz)', fontsize=11)
            ax.set_ylabel('Window Index', fontsize=11)
            ax.set_title(f'Sliding Window - {labels[axis_name]}', 
                        fontsize=12, fontweight='bold')
            ax.set_ylim(-0.5, num_windows - 0.5)
            
            # Adaptive y-ticks: aim for ~10 ticks maximum for readability
            max_ticks = 10
            tick_stride = max(1, int(np.ceil(num_windows / max_ticks)))
            yticks = np.arange(0, num_windows, tick_stride)
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(int(i)) for i in yticks], fontsize=9)
            
            # Add white separator lines between windows (always at every window edge)
            for win_idx in np.arange(0.5, num_windows, 1):
                ax.axhline(win_idx, color='white', linestyle='-', linewidth=0.5, alpha=0.6)
            
            # Add secondary x-axis for echo spacing
            _add_echo_spacing_axis(ax, freq_min, freq_max)
        
        # Add common colorbar below the three subplots
        if im is not None:
            cbar_ax = fig2.add_axes([0.15, 0.05, 0.7, 0.02])
            cbar = fig2.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Magnitude [a.u.]', fontsize=11)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig2.tight_layout(rect=[0, 0.08, 1, 1])
    
    return fig, fig.axes


def _add_echo_spacing_axis(ax, freq_min, freq_max):
    """
    Add a secondary x-axis showing echo spacing (µs) = 1/(2*frequency).
    Uses proper inverse transformation so tick values are correct.
    """
    ax_top = ax.twiny()
    
    # Get the primary axis limits in data coordinates
    ax_top.set_xlim(ax.get_xlim())
        
    def freq_to_es(freq_hz):
        """Convert frequency (Hz) to echo spacing (µs)."""
        if freq_hz <= 0:
            return np.inf
        return 1e6 / (2 * freq_hz)
        
    # Get current ticks from primary axis
    primary_ticks = ax.get_xticks()
    # Filter out negative ticks
    primary_ticks = primary_ticks[primary_ticks >= 0]
        
    # Set ticks and labels
    ax_top.set_xticks(primary_ticks)
    ax_top.set_xticklabels([f'{freq_to_es(f):.1f}' if f > 0 else '∞' for f in primary_ticks])
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xlabel('Echo Spacing (µs)', fontsize=12)
    
    return ax_top


def _plot_pns(
    pns: SimpleNamespace,
    seq: PulserverSequence | None = None,
) -> tuple:
    """
    Plot PNS waveforms with TR gradient waveforms circularly padded to match.
    
    Creates a comprehensive visualization of PNS analysis including:
    - Top panel: PNS % vs time (ms) with 80% (dotted) and 100% (dashed) limits
    - Bottom panel: Gradient waveforms circularly padded to match PNS length
    
    Parameters
    ----------
    pns : SimpleNamespace
        Output from get_pns() with store_waveforms=True.
        Must contain pns_x, pns_y, pns_z, and num_samples.
    seq : PulserverSequence | None
        The sequence object (needed to get gradient waveforms).
        If None, bottom waveform panel is skipped.
        
    Returns
    -------
    tuple
        (fig, axes) - Matplotlib figure and axes for further customization.
        
    Raises
    ------
    ValueError
        If pns is missing required attributes (pns_x, pns_y, pns_z).
    RuntimeError
        If waveforms requested but seq not provided.
        
    Notes
    -----
    - Top panel: PNS % vs time (ms) with 80% (dotted) and 100% (dashed) limits
    - Bottom panel: Gradient waveforms circularly padded to match PNS length
    - Waveforms are in mT/m
    """
    # Validate input
    if not hasattr(pns, 'pns_x') or not hasattr(pns, 'pns_y') or not hasattr(pns, 'pns_z'):
        raise ValueError("pns must have pns_x, pns_y, pns_z arrays from get_pns() with store_waveforms=True")
    
    if not hasattr(pns, 'num_samples'):
        raise ValueError("pns must have num_samples attribute")
    
    num_pns_samples = pns.num_samples
    grad_raster_time = seq.system.grad_raster_time  # in seconds
    
    # Create figure
    fig = plt.figure()
    
    # ============ Panel 1: PNS Waveforms ============
    ax_pns = plt.subplot(2, 1, 1)
    
    # Create time array in ms based on PNS samples and gradient raster
    time_pns_ms = np.arange(num_pns_samples) * 0.5 * grad_raster_time * 1000.0  # Convert to ms
    
    # Plot PNS waveforms
    colors = {'x': 'C0', 'y': 'C1', 'z': 'C2'}
    labels = {'x': 'PNS_X', 'y': 'PNS_Y', 'z': 'PNS_Z'}
    
    ax_pns.plot(time_pns_ms, pns.pns_x, color=colors['x'], linewidth=2, label=labels['x'])
    ax_pns.plot(time_pns_ms, pns.pns_y, color=colors['y'], linewidth=2, label=labels['y'])
    ax_pns.plot(time_pns_ms, pns.pns_z, color=colors['z'], linewidth=2, label=labels['z'])
    
    # Add threshold lines
    ax_pns.axhline(80.0, color='gray', linestyle=':', linewidth=2, label='80% limit')
    ax_pns.axhline(100.0, color='red', linestyle='--', linewidth=2, label='100% threshold')
    
    ax_pns.set_xlabel('Time (ms)', fontsize=12)
    ax_pns.set_ylabel('PNS (%)', fontsize=12)
    ax_pns.set_title('Peripheral Nerve Stimulation (PNS)', fontsize=14, fontweight='bold')
    ax_pns.legend(loc='upper right', fontsize=11)
    ax_pns.grid(True, alpha=0.3)
    ax_pns.set_ylim(bottom=0)
    
    # ============ Panel 2: Gradient Waveforms (Circularly Padded) ============
    if seq is not None:
        ax_wf = plt.subplot(2, 1, 2)
        
        gamma = seq.system.gamma
        
        # Get original waveforms
        waveforms = get_tr_gradient_waveforms(seq)
        
        # Convert waveforms to mT/m
        wf_gx_mtpm = waveforms.waveform_gx / gamma * 1000.0
        wf_gy_mtpm = waveforms.waveform_gy / gamma * 1000.0
        wf_gz_mtpm = waveforms.waveform_gz / gamma * 1000.0
        
        # Circularly pad waveforms to match PNS size
        wf_gx_padded = _circular_pad(wf_gx_mtpm, num_pns_samples)
        wf_gy_padded = _circular_pad(wf_gy_mtpm, num_pns_samples)
        wf_gz_padded = _circular_pad(wf_gz_mtpm, num_pns_samples)
        
        # Use the SAME time array as PNS plot (based on num_pns_samples)
        time_wf_ms = time_pns_ms  # <-- FIX: Use consistent time axis
        
        ax_wf.plot(time_wf_ms, wf_gx_padded, color=colors['x'], linewidth=2, label='Gx')
        ax_wf.plot(time_wf_ms, wf_gy_padded, color=colors['y'], linewidth=2, label='Gy')
        ax_wf.plot(time_wf_ms, wf_gz_padded, color=colors['z'], linewidth=2, label='Gz')
        
        ax_wf.set_xlabel('Time (ms)', fontsize=12)
        ax_wf.set_ylabel('Gradient Amplitude (mT/m)', fontsize=12)
        ax_wf.set_title('TR Gradient Waveforms (Circularly Padded to PNS Length)', fontsize=14, fontweight='bold')
        ax_wf.legend(loc='upper right', fontsize=11)
        ax_wf.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, fig.axes


def _circular_pad(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """
    Circularly pad a waveform to a target length.
    
    If target_length > len(waveform), repeats the waveform cyclically.
    If target_length < len(waveform), truncates the waveform.
    If target_length == len(waveform), returns a copy.
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform.
    target_length : int
        Desired output length.
        
    Returns
    -------
    np.ndarray
        Padded waveform of length target_length.
    """
    current_length = len(waveform)
    
    if current_length == target_length:
        return waveform.copy()
    elif current_length > target_length:
        # Truncate
        return waveform[:target_length].copy()
    else:
        # Circular repeat: tile the waveform and then truncate
        num_repeats = int(np.ceil(target_length / current_length))
        padded = np.tile(waveform, num_repeats)
        return padded[:target_length].copy()


def _check_pns_thresholds(pns: SimpleNamespace) -> None:
    """
    Check PNS waveforms for threshold violations and print warnings.
    
    Parameters
    ----------
    pns : SimpleNamespace
        Output from get_pns() with pns_x, pns_y, pns_z arrays.
    """
    if not hasattr(pns, 'pns_x') or not hasattr(pns, 'pns_y') or not hasattr(pns, 'pns_z'):
        return  # No per-axis PNS data available
    
    thresholds = [80.0, 100.0]
    axes = {'pns_x': 'X', 'pns_y': 'Y', 'pns_z': 'Z'}
    
    for axis_attr, axis_name in axes.items():
        pns_data = getattr(pns, axis_attr, None)
        if pns_data is None:
            continue
        
        for threshold in thresholds:
            max_val = np.max(pns_data)
            if max_val > threshold:
                warnings.warn(
                    f"PNS {axis_name} exceeds {threshold}% threshold: "
                    f"max = {max_val:.1f}%",
                    UserWarning
                )


def _check_acoustic_forbidden_bands(
    spectra: SimpleNamespace,
    seq: PulserverSequence,
    forbidden_bands: list[dict]
) -> None:
    """
    Check acoustic spectra against forbidden frequency bands and print warnings.
    
    Parameters
    ----------
    spectra : SimpleNamespace
        Output from get_tr_acoustic_spectra().
    seq : PulserverSequence
        The sequence (for gamma conversion).
    forbidden_bands : list[dict]
        List of forbidden bands, each with 'freq_min_hz', 'freq_max_hz', 'max_amplitude'.
    """
    if not forbidden_bands:
        return
    
    gamma = seq.system.gamma
    axes = {'gx': 'X', 'gy': 'Y', 'gz': 'Z'}
    
    for band in forbidden_bands:
        freq_min = band['freq_min_hz']
        freq_max = band['freq_max_hz']
        max_allowed_hzm = band.get('max_amplitude', float('inf'))
        max_allowed_mtpm = max_allowed_hzm / gamma * 1000.0  # Convert to mT/m
        
        for axis_short, axis_name in axes.items():
            # Check sliding window peaks (if available)
            peaks_attr = f'peaks_g{axis_short}'
            spec_attr = f'spectra_g{axis_short}'
            envelope_attr = f'max_envelope_g{axis_short}'
            
            if not hasattr(spectra, peaks_attr) or not hasattr(spectra, spec_attr):
                continue
            
            peaks = getattr(spectra, peaks_attr)
            specs = getattr(spectra, spec_attr)
            envelope = getattr(spectra, envelope_attr)
            
            # Check if there are peaks in the forbidden band
            freq_indices = (spectra.frequencies >= freq_min) & (spectra.frequencies <= freq_max)
            
            if peaks is None:
                # For combined=True, check if spectrum has energy in band
                if specs.ndim == 1:
                    has_peaks_in_band = np.any(specs[freq_indices] > 0)
                else:
                    has_peaks_in_band = np.any(peaks[:, freq_indices] > 0)
            else:
                has_peaks_in_band = np.any(peaks[:, freq_indices] > 0) if peaks.ndim == 2 else np.any(peaks[freq_indices] > 0)
            
            if has_peaks_in_band:
                # Find max envelope in the band
                max_envelope_in_band = np.max(envelope[freq_indices])
                if max_envelope_in_band > max_allowed_mtpm:
                    warnings.warn(
                        f"Acoustic forbidden band violation: {axis_name} "
                        f"({freq_min:.1f}–{freq_max:.1f} Hz) "
                        f"has peaks with envelope {max_envelope_in_band:.2f} mT/m "
                        f"(allowed: {max_allowed_mtpm:.2f} mT/m)",
                        UserWarning
                    )
            
            # Also check full sequence spectrum if available (for sequence harmonic spectrum)
            if hasattr(spectra, f'spectrum_g{axis_short}_seq') and hasattr(spectra, f'peaks_g{axis_short}_seq'):
                spec_seq = getattr(spectra, f'spectrum_g{axis_short}_seq')
                peaks_seq = getattr(spectra, f'peaks_g{axis_short}_seq')
                envelope_seq = getattr(spectra, f'max_envelope_g{axis_short}_seq', None)
                
                if spec_seq is not None and peaks_seq is not None:
                    freq_indices_seq = (spectra.frequencies_seq >= freq_min) & (spectra.frequencies_seq <= freq_max)
                    has_peaks_in_band_seq = np.any(peaks_seq[freq_indices_seq] > 0)
                    
                    if has_peaks_in_band_seq and envelope_seq is not None:
                        max_envelope_in_band_seq = np.max(envelope_seq[freq_indices_seq])
                        if max_envelope_in_band_seq > max_allowed_mtpm:
                            warnings.warn(
                                f"Acoustic forbidden band violation (sequence spectrum): {axis_name} "
                                f"({freq_min:.1f}–{freq_max:.1f} Hz) "
                                f"has peaks with envelope {max_envelope_in_band_seq:.2f} mT/m "
                                f"(allowed: {max_allowed_mtpm:.2f} mT/m)",
                                UserWarning
                            )