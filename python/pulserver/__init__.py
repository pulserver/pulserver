"""
Pulserver — offline validation and debugging for pulseq MRI sequences.

Public API:

    SequenceCollection    Wraps a pypulseq Sequence with C-backed analysis.
    serialize             Save collection to linked .seq file chain.
    deserialize           Restore collection from linked .seq file chain.
"""

__all__ = [
    "SequenceCollection",
    "serialize",
    "deserialize",
    "FastSequence",
    "write",
]

from .core import (  # noqa: F401
    SequenceCollection,
    deserialize,
    serialize,
)
from .io import write  # noqa: F401
from .sequence import FastSequence  # noqa: F401
