"""
Pulserver — offline validation and debugging for pulseq MRI sequences.

Public API:

    SequenceCollection    Wraps a pypulseq Sequence with C-backed analysis.
    plot                  Plot TR waveforms with overlay support.
    pns                   Estimate peripheral nerve stimulation.
    grad_spectrum         Compute acoustic spectrograms and harmonics.
    check                 Run consistency and safety checks.
    validate              Compare C-backend vs pypulseq waveforms.
    report                Structured collection report.
    serialize             Save collection to binary cache.
    deserialize           Restore collection from binary cache.
"""

__all__ = [
    'SequenceCollection',
    'plot',
    'pns',
    'grad_spectrum',
    'check',
    'validate',
    'report',
    'serialize',
    'deserialize',
]

from .core import (  # noqa: F401
    SequenceCollection,
    plot,
    pns,
    grad_spectrum,
    check,
    validate,
    report,
    serialize,
    deserialize,
)
