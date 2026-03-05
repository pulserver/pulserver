"""Native-timing TR waveform extraction for SequenceCollection."""

__all__ = ['get_tr_waveforms', 'TrWaveforms']

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ._extension._pulseqlib_wrapper import _find_tr, _get_tr_waveforms
from ._sequence import SequenceCollection

# Physical-unit conversion constants
_GAMMA_DEFAULT = 42.576e6  # Hz/T


@dataclass
class ChannelWaveform:
    """Single-channel waveform with native (non-uniform) timing.

    Attributes
    ----------
    time_us : np.ndarray
        Time points in microseconds, shape ``(N,)``.
    amplitude : np.ndarray
        Amplitude values, shape ``(N,)``.  Units depend on channel:
        mT/m for gradients, µT for RF magnitude, rad for RF phase.
    """
    time_us: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    amplitude: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


@dataclass
class AdcEvent:
    """ADC event descriptor."""
    onset_us: float = 0.0
    duration_us: float = 0.0
    num_samples: int = 0
    freq_offset_hz: float = 0.0
    phase_offset_rad: float = 0.0


@dataclass
class BlockDescriptor:
    """Per-block metadata within a TR."""
    start_us: float = 0.0
    duration_us: float = 0.0
    segment_idx: int = -1


@dataclass
class TrWaveforms:
    """Complete native-timing TR waveforms for plotting.

    Gradient amplitudes are in mT/m,  RF magnitude in µT,
    RF phase in radians.  Times are in microseconds.

    Attributes
    ----------
    gx, gy, gz : ChannelWaveform
        Gradient waveforms (mT/m).
    rf_mag : ChannelWaveform
        RF magnitude envelope (µT).
    rf_phase : ChannelWaveform
        RF phase (rad).
    adc_events : list[AdcEvent]
        ADC event descriptors.
    blocks : list[BlockDescriptor]
        Per-block metadata (timing, segment assignment).
    total_duration_us : float
        Total duration of the extracted block range (µs).
    gamma_hz_per_t : float
        Gyromagnetic ratio used for unit conversion.
    tr_duration_us : float
        Duration of a single main TR (µs), for pypulseq overlay.
    prep_duration_us : float
        Total duration of prep blocks (µs), for pypulseq overlay.
    """
    gx: ChannelWaveform = field(default_factory=ChannelWaveform)
    gy: ChannelWaveform = field(default_factory=ChannelWaveform)
    gz: ChannelWaveform = field(default_factory=ChannelWaveform)
    rf_mag: ChannelWaveform = field(default_factory=ChannelWaveform)
    rf_phase: ChannelWaveform = field(default_factory=ChannelWaveform)
    adc_events: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    total_duration_us: float = 0.0
    gamma_hz_per_t: float = _GAMMA_DEFAULT
    tr_duration_us: float = 0.0
    prep_duration_us: float = 0.0


# Amplitude mode constants (must match C defines)
AMP_MAX_POS = 0
AMP_ZERO_VAR = 1
AMP_ACTUAL = 2


def get_tr_waveforms(
    seq: SequenceCollection,
    subsequence_idx: int = 0,
    amplitude_mode: Literal['max_pos', 'zero_var', 'actual'] = 'max_pos',
    tr_index: int = 0,
    include_prep: bool = False,
    include_cooldown: bool = False,
    collapse_delays: bool = False,
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
    include_prep : bool
        Prepend preparation blocks (only valid for first TR).
    include_cooldown : bool
        Append cooldown blocks (only valid for last TR).
    collapse_delays : bool
        Shrink pure-delay blocks (no RF, no grad, no ADC) to 0.1 ms
        at the C level, producing a compact waveform timeline.

    Returns
    -------
    TrWaveforms
        Dataclass with per-channel waveforms and metadata.
    """
    mode_map = {'max_pos': AMP_MAX_POS, 'zero_var': AMP_ZERO_VAR, 'actual': AMP_ACTUAL}
    c_mode = mode_map[amplitude_mode]

    gamma = seq.system.gamma  # Hz/T

    # Call C extension
    raw = _get_tr_waveforms(
        seq._cseq,
        subsequence_idx=subsequence_idx,
        amplitude_mode=c_mode,
        tr_index=tr_index,
        include_prep=include_prep,
        include_cooldown=include_cooldown,
        collapse_delays=collapse_delays,
    )

    # Unit conversions:
    #   grad: Hz/m → mT/m :  x / (gamma * 1e3)  [since gamma is Hz/T, and 1 T/m = gamma Hz/m]
    #   rf:   Hz   → µT   :  x / gamma * 1e6
    hz_per_m_to_mT_per_m = 1.0 / (gamma * 1e-3)  # Hz/m → mT/m
    hz_to_uT = 1e6 / gamma                         # Hz → µT

    def _make_channel(d, scale=1.0):
        t = np.asarray(d['time_us'], dtype=np.float32)
        a = np.asarray(d['amplitude'], dtype=np.float32) * scale
        return ChannelWaveform(time_us=t, amplitude=a)

    gx = _make_channel(raw['gx'], hz_per_m_to_mT_per_m)
    gy = _make_channel(raw['gy'], hz_per_m_to_mT_per_m)
    gz = _make_channel(raw['gz'], hz_per_m_to_mT_per_m)
    rf_mag = _make_channel(raw['rf_mag'], hz_to_uT)
    rf_phase = _make_channel(raw['rf_phase'], 1.0)  # already radians

    adc_events = [
        AdcEvent(
            onset_us=a['onset_us'],
            duration_us=a['duration_us'],
            num_samples=a['num_samples'],
            freq_offset_hz=a['freq_offset_hz'],
            phase_offset_rad=a['phase_offset_rad'],
        )
        for a in raw['adc_events']
    ]

    blocks = [
        BlockDescriptor(
            start_us=b['start_us'],
            duration_us=b['duration_us'],
            segment_idx=b['segment_idx'],
        )
        for b in raw['blocks']
    ]

    # Get TR timing info for pypulseq overlay
    tr_info = _find_tr(seq._cseq, subsequence_idx=subsequence_idx)
    tr_dur = tr_info['tr_duration_us']
    prep_dur = sum(
        b['duration_us'] for b in raw['blocks']
        if b['segment_idx'] < 0  # prep/cooldown heuristic — refine if needed
    ) if include_prep else 0.0

    return TrWaveforms(
        gx=gx, gy=gy, gz=gz,
        rf_mag=rf_mag, rf_phase=rf_phase,
        adc_events=adc_events,
        blocks=blocks,
        total_duration_us=raw['total_duration_us'],
        gamma_hz_per_t=gamma,
        tr_duration_us=tr_dur,
        prep_duration_us=prep_dur,
    )
