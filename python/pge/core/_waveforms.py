def calc_mech_resonance_spectra(
    seq: 'SequenceCollection',
    subsequence_idx: int = 0,
    canonical_tr_idx: int = 0,
    target_resolution_hz: float = 0.0,
    max_freq_hz: float = 0.0,
    forbidden_bands=None,
) -> dict:
    """Compute mechanical resonances spectral data (gradient frequency content) for a TR.

    Returns the structural analytical candidate set together with a
    full-TR FFT magnitude spectrum (display-only).  The pass/fail
    verdict comes from the structural candidates, not from the FFT.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyze.
    subsequence_idx : int, default 0
        Subsequence index (0-based).
    canonical_tr_idx : int, default 0
        Canonical TR index (0-based) to analyze.
    target_resolution_hz : float, default 0.0
        Target frequency resolution (Hz). If 0, uses default (5 Hz).
    max_freq_hz : float, default 0.0
        Maximum frequency for analysis (Hz). If 0, uses default (3000 Hz).
    forbidden_bands : list, optional
        List of ``(freq_min, freq_max, max_amplitude)`` tuples defining
        mechanical resonance forbidden bands. If ``None``, no bands are checked.

    Returns
    -------
    dict
        Mechanical resonances spectrum analysis results from C backend.
    """
    if forbidden_bands is None:
        forbidden_bands = []
    return seq._cseq._calc_mech_resonances(
        subsequence_idx,
        canonical_tr_idx,
        target_resolution_hz,
        max_freq_hz,
        forbidden_bands,
    )


def calc_pns(
    seq: 'SequenceCollection',
    subsequence_idx: int = 0,
    canonical_tr_idx: int = 0,
    chronaxie_us: float = 360.0,
    rheobase: float = 20.0,
    alpha: float = 0.333,
) -> dict:
    """Compute peripheral nerve stimulation (PNS) risk waveforms for a TR.

    Uses the SAFE nerve model (Siebold et al. 2015) to convolve gradient
    slew rates with exponential decay, producing a PNS percentage waveform.
    Identifies peak stimulation levels per axis and combined.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyze.
    subsequence_idx : int, default 0
        Subsequence index (0-based).
    canonical_tr_idx : int, default 0
        Canonical TR index (0-based) to analyze.
    chronaxie_us : float, default 360.0
        Chronaxie parameter (µs) — decay time constant of nerve response.
    rheobase : float, default 20.0
        Rheobase parameter (Hz/m/s) — minimum stimulation threshold.
    alpha : float, default 0.333
        Alpha parameter (dimensionless) — scaling exponent for nerve model.

    Returns
    -------
    dict
        PNS analysis results from C backend, including per-axis and
        combined percentage waveforms and peak levels.
    """
    return seq._cseq._calc_pns(
        subsequence_idx,
        canonical_tr_idx,
        chronaxie_us,
        rheobase,
        alpha,
    )


"""Native-timing TR waveform extraction for SequenceCollection."""

__all__ = ['TrWaveforms', 'get_tr_waveforms']

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ._extension._pulseqlib_wrapper import _find_tr, _get_tr_waveforms

# Import SequenceCollection before any type annotations use it
from ._sequence import SequenceCollection

# Physical-unit conversion constants
_GAMMA_DEFAULT = 42.576e6  # Hz/T


@dataclass
class ChannelWaveform:
    """Single-channel waveform with native (non-uniform) timing.

    Represents a waveform (gradient, RF, ADC) with per-point time stamps
    and amplitudes. Times are native microsecond clock positions from the
    C library, not interpolated to a uniform raster. This preserves ADC
    sampling patterns and reveals aliasing artifacts.

    Attributes
    ----------
    time_us : np.ndarray
        Time points in microseconds, shape ``(N,)``. Typically a 1D float32 array.
    amplitude : np.ndarray
        Amplitude values, shape ``(N,)``. Units depend on channel:

        - Gradients (Gx, Gy, Gz): mT/m
        - RF magnitude: µT
        - RF phase: radians
        - ADC: normalized (0-1 range)
    """

    time_us: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    amplitude: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


@dataclass
class AdcEvent:
    """ADC (analog-to-digital converter) event descriptor.

    Represents a single readout window with timing, frequency offset,
    phase offset, and sampling parameters.

    Attributes
    ----------
    onset_us : float
        Event start time relative to TR start (microseconds).
    duration_us : float
        Event duration (microseconds).
    num_samples : int
        Number of samples acquired during this event.
    freq_offset_hz : float
        Receiver frequency offset (Hz) for this event.
    phase_offset_rad : float
        Receiver phase offset (radians) for this event.
    """

    onset_us: float = 0.0
    duration_us: float = 0.0
    num_samples: int = 0
    freq_offset_hz: float = 0.0
    phase_offset_rad: float = 0.0


@dataclass
class BlockDescriptor:
    """Per-block metadata within a TR (scan table parameters).

    Describes a pypulseq block's timing, segment assignment, and RF/ADC
    offsets in the context of the extracted TR waveforms.

    Attributes
    ----------
    start_us : float
        Block start time relative to TR start (microseconds).
    duration_us : float
        Block duration (microseconds).
    segment_idx : int
        Segment index (-1 for prep/cooldown, 0+ for imaging segments).
    rf_isocenter_us : float
        RF isocenter time relative to TR start (µs), or -1.0 if no RF.
    adc_kzero_us : float
        ADC k=0 time relative to TR start (µs), or -1.0 if no ADC.
    rf_freq_offset_hz : float
        RF transmit frequency offset for this block (Hz).
    rf_phase_offset_rad : float
        RF transmit phase offset for this block (radians).
    adc_freq_offset_hz : float
        ADC receive frequency offset for this block (Hz).
    adc_phase_offset_rad : float
        ADC receive phase offset for this block (radians).
    rotation_matrix : list
        Slice selection rotation matrix (if applicable), typically 3x3.
    """

    start_us: float = 0.0
    duration_us: float = 0.0
    segment_idx: int = -1
    rf_isocenter_us: float = -1.0
    adc_kzero_us: float = -1.0
    rf_freq_offset_hz: float = 0.0
    rf_phase_offset_rad: float = 0.0
    adc_freq_offset_hz: float = 0.0
    adc_phase_offset_rad: float = 0.0
    rotation_matrix: list = field(default_factory=list)


@dataclass
class TrWaveforms:
    """Complete native-timing TR waveforms for plotting and analysis.

    Contains gradient, RF, and ADC waveforms extracted from the C backend
    with native (non-uniform) microsecond timing. Amplitudes are converted
    to physical units: mT/m for gradients, µT for RF magnitude.
    RF phase is in radians.

    Attributes
    ----------
    gx, gy, gz : ChannelWaveform
        Gradient waveforms (mT/m) for X, Y, Z axes.
    rf_mag : ChannelWaveform
        RF magnitude envelope (µT).
    rf_phase : ChannelWaveform
        RF phase (rad).
    rf_mag_channels : list[ChannelWaveform]
        Per-channel RF magnitudes (for pTx systems). Empty for single-Tx.
    rf_phase_channels : list[ChannelWaveform]
        Per-channel RF phases (for pTx systems). Empty for single-Tx.
    adc_events : list[AdcEvent]
        Readout window descriptors (onset, duration, sampling rate, offsets).
    blocks : list[BlockDescriptor]
        Per-block scan table metadata (timing, segment assignment, offsets).
    total_duration_us : float
        Total duration of extracted block range (µs).
    gamma_hz_per_t : float
        Gyromagnetic ratio used for unit conversion (Hz/T). Default 42.576e6.
    tr_duration_us : float
        Duration of a single main imaging TR (µs), for pypulseq overlay context.
    prep_duration_us : float
        Total duration of prep blocks (µs), for pypulseq overlay context.
    """

    gx: ChannelWaveform = field(default_factory=ChannelWaveform)
    gy: ChannelWaveform = field(default_factory=ChannelWaveform)
    gz: ChannelWaveform = field(default_factory=ChannelWaveform)
    rf_mag: ChannelWaveform = field(default_factory=ChannelWaveform)
    rf_phase: ChannelWaveform = field(default_factory=ChannelWaveform)
    rf_mag_channels: list = field(
        default_factory=list
    )  # per-channel ChannelWaveform; empty for single-Tx
    rf_phase_channels: list = field(
        default_factory=list
    )  # per-channel ChannelWaveform; empty for single-Tx
    adc_events: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    total_duration_us: float = 0.0
    gamma_hz_per_t: float = _GAMMA_DEFAULT
    tr_duration_us: float = 0.0
    prep_duration_us: float = 0.0


# Amplitude mode constants (must match C defines)


# Amplitude mode constants (must match C defines)
AMP_MAX_POS = 0
AMP_ZERO_VAR = 1
AMP_ACTUAL = 2


def get_tr_waveforms(
    seq: SequenceCollection,
    subsequence_idx: int = 0,
    amplitude_mode: Literal['max_pos', 'zero_var', 'actual'] = 'max_pos',
    tr_index: int = 0,
    collapse_delays: bool = False,
    num_averages: int = 0,
) -> TrWaveforms:
    """Extract native-timing TR waveforms from the segmented representation.

    Returns gradient, RF, and ADC waveforms with per-channel native
    timing (NOT interpolated to a uniform raster).  Amplitudes are
    converted to physical units: mT/m for gradients, µT for RF.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyse.
    subsequence_idx : int
        Subsequence index (0-based, default 0).
    amplitude_mode : {'max_pos', 'zero_var', 'actual'}
        How to resolve multi-shot gradient amplitudes:

        - ``'max_pos'`` — position-max across all TRs (safety view).
        - ``'zero_var'`` — zero variable grads, keep constant (k-space view).
        - ``'actual'``  — signed amplitude for a specific TR instance.
    tr_index : int
        TR instance index (0-based).  Only used for ``'actual'`` mode.
    (Preparation and cooldown blocks are always included if present.)
    collapse_delays : bool
        Shrink pure-delay blocks (no RF, no grad, no ADC) to 0.1 ms
        at the C level, producing a compact waveform timeline.
    num_averages : int
        Override average count for ACTUAL mode.  0 uses the descriptor
        default stored at load time.

    Returns
    -------
    TrWaveforms
        Dataclass with per-channel waveforms and metadata.
    """
    # Accept 'canonical' as alias for 'max_pos' (structural canonical TR)

    if amplitude_mode == 'canonical':
        amplitude_mode = 'max_pos'
    mode_map = {'max_pos': AMP_MAX_POS, 'zero_var': AMP_ZERO_VAR, 'actual': AMP_ACTUAL}
    c_mode = mode_map[amplitude_mode]

    gamma = seq.system.gamma  # Hz/T

    # Call C extension
    raw = _get_tr_waveforms(
        seq._cseq,
        subsequence_idx,
        c_mode,
        tr_index,
        collapse_delays,
        num_averages,
    )

    # Unpack raw waveform arrays
    hz_per_m_to_mT_per_m = 1e3 / gamma
    gx = ChannelWaveform(
        np.array(raw['gx']['time_us']),
        np.array(raw['gx']['amplitude']) * hz_per_m_to_mT_per_m,
    )
    gy = ChannelWaveform(
        np.array(raw['gy']['time_us']),
        np.array(raw['gy']['amplitude']) * hz_per_m_to_mT_per_m,
    )
    gz = ChannelWaveform(
        np.array(raw['gz']['time_us']),
        np.array(raw['gz']['amplitude']) * hz_per_m_to_mT_per_m,
    )
    # Convert RF magnitude from Hz to µT for validation (µT = 1e6 / gamma * Hz).
    # The C backend stores signed magnitude (negative for negative lobes) with
    # phase separate.  Convert to unsigned magnitude + adjusted phase so the
    # phase of negative lobes is π, matching the pypulseq convention.
    rf_mag_hz = np.array(raw['rf_mag']['amplitude'])
    rf_phase_raw = np.array(raw['rf_phase']['amplitude'])
    neg = rf_mag_hz < 0
    rf_mag_hz = np.abs(rf_mag_hz)
    rf_phase_raw = rf_phase_raw.copy()
    rf_phase_raw[neg] += np.pi
    rf_mag_uT = rf_mag_hz * (1e6 / gamma)
    rf_mag = ChannelWaveform(np.array(raw['rf_mag']['time_us']), rf_mag_uT)
    rf_phase = ChannelWaveform(np.array(raw['rf_phase']['time_us']), rf_phase_raw)
    # For pTx (num_rf_channels > 1), split the flat channel-major array into
    # per-channel ChannelWaveform objects.
    nch = raw.get('num_rf_channels', 1)
    if nch > 1 and len(rf_mag_uT) % nch == 0:
        npts = len(rf_mag_uT) // nch
        t_rf = np.array(raw['rf_mag']['time_us'])
        rf_mag_channels = [
            ChannelWaveform(
                t_rf[c * npts : (c + 1) * npts], rf_mag_uT[c * npts : (c + 1) * npts]
            )
            for c in range(nch)
        ]
        rf_phase_channels = [
            ChannelWaveform(
                t_rf[c * npts : (c + 1) * npts], rf_phase_raw[c * npts : (c + 1) * npts]
            )
            for c in range(nch)
        ]
    else:
        rf_mag_channels = []
        rf_phase_channels = []
    adc_events = [AdcEvent(**adc) for adc in raw['adc_events']]
    # Extract per-block scan table parameters if present
    blocks = []
    for blk in raw['blocks']:
        block_kwargs = {
            'start_us': blk.get('start_us', 0.0),
            'duration_us': blk.get('duration_us', 0.0),
            'segment_idx': blk.get('segment_idx', -1),
            'rf_isocenter_us': blk.get('rf_isocenter_us', -1.0),
            'adc_kzero_us': blk.get('adc_kzero_us', -1.0),
            'rf_freq_offset_hz': blk.get('rf_freq_offset_hz', 0.0),
            'rf_phase_offset_rad': blk.get('rf_phase_offset_rad', 0.0),
            'adc_freq_offset_hz': blk.get('adc_freq_offset_hz', 0.0),
            'adc_phase_offset_rad': blk.get('adc_phase_offset_rad', 0.0),
            'rotation_matrix': blk.get('rotation_matrix', []),
        }
        blocks.append(BlockDescriptor(**block_kwargs))

    # Get TR timing info for pypulseq overlay
    tr_info = _find_tr(seq._cseq, subsequence_idx=subsequence_idx)
    tr_dur = tr_info['tr_duration_us']
    prep_dur = sum(
        b['duration_us']
        for b in raw['blocks']
        if b['segment_idx'] < 0  # prep/cooldown heuristic — refine if needed
    )

    return TrWaveforms(
        gx=gx,
        gy=gy,
        gz=gz,
        rf_mag=rf_mag,
        rf_phase=rf_phase,
        rf_mag_channels=rf_mag_channels,
        rf_phase_channels=rf_phase_channels,
        adc_events=adc_events,
        blocks=blocks,
        total_duration_us=raw['total_duration_us'],
        gamma_hz_per_t=gamma,
        tr_duration_us=tr_dur,
        prep_duration_us=prep_dur,
    )
