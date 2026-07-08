"""
Pulserver analysis sub-package.

This sub-package contains vendor-neutral routines for TR-based sequence
representation, waveform validation, and visualization.

Public API:

    SequenceCollection    Wraps a pypulseq Sequence with C-backed analysis.
    serialize             Save collection to linked .seq file chain.
    deserialize           Restore collection from linked .seq file chain.
    Opts                  Vendor-neutral pypulseq.Opts extension.

    Sequence-description dataclasses (from info() / trajectory_info()):
    EncodingSpace, LabelLimits, SeqRow,
    SequenceDescription, SequenceDescriptionInfo, SequenceParameters,
    TrajTableEntry, TrajectoryInfo
"""

__all__ = [
    # ── sequence description ──────────────────────────────────
    'EncodingSpace',
    'LabelLimits',
    'Opts',
    'SeqRow',
    'SequenceCollection',
    'SequenceDescription',
    'SequenceDescriptionInfo',
    'SequenceParameters',
    'TrajTableEntry',
    'TrajectoryInfo',
    'deserialize',
    'serialize',
]

from ._cache import deserialize, serialize
from ._cache_sections import (
    EncodingSpace,
    LabelLimits,
    SeqRow,
    SequenceDescription,
    SequenceDescriptionInfo,
    SequenceParameters,
    TrajectoryInfo,
    TrajTableEntry,
)
from ._opts import Opts
from ._sequence import SequenceCollection
