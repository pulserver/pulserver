"""
Pulserver — offline validation and debugging for pulseq MRI sequences.

Public API:

    SequenceCollection    Wraps a pypulseq Sequence with C-backed analysis.
    serialize             Save collection to linked .seq file chain.
    deserialize           Restore collection from linked .seq file chain.

    Sequence-description dataclasses (from info() / trajectory_info()):
    EncodingSpace, LabelLimits, SeqRow,
    SequenceDescription, SequenceDescriptionInfo, SequenceParameters,
    TrajTableEntry, TrajectoryInfo
"""

__all__ = [
    # ── sequence description ──────────────────────────────────
    'EncodingSpace',
    'GEOpts',
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

from .core import (
    GEOpts,
    Opts,
    SequenceCollection,
    deserialize,
    serialize,
)
from .core._cache_sections import (
    EncodingSpace,
    LabelLimits,
    SeqRow,
    SequenceDescription,
    SequenceDescriptionInfo,
    SequenceParameters,
    TrajectoryInfo,
    TrajTableEntry,
)
