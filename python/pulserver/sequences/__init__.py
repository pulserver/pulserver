"""Ready-to-run sequence factories for interactive Pulserver use.

Each factory takes the sequence's controls as explicit keyword arguments and
returns an in-memory :class:`pulserver.pypulseq.Sequence`, so a sequence
designed at a REPL and one designed through the bridge share an implementation.

None are shipped yet: the readout and excitation modules they are written
against are being rebuilt in :mod:`pulserver.design`.
"""

__all__ = []
