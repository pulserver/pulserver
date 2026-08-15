"""Spec-first PulSeg oracle: content identities and partition validation.

The oracle answers two questions about a Pulseq file (or NextSequence
chain) independently of the C interpreter:

1. **Content**: the normalized, amplitude-free identity of every block
   (PulSeg spec 3.0-3.1 normalization: shapes by value, trapezoids by
   timing, extended gradients by delay + time shape, ADCs by count/dwell/
   delay) and the per-instance execution parameters the spec's
   SegmentInstance carries.
2. **Validation**: whether a given segment partition -- the interpreter's,
   derived from TR content alone -- satisfies the spec's structural
   constraints: the segments tile the stream exactly, every instance of a
   segment has the same block count and the same normalized structure, and
   a play-once region is never folded into a repeating unit.

The oracle never dictates a partition: segmentation granularity is the
interpreter's freedom under semantic conformance, and TRID plays no
segmentation role (it is an optional SAR/coil-heating refinement label
for interleaved-subsequence hyperTRs, nothing more).
"""

from .identity import block_signature, sequence_signatures
from .validate import PartitionError, tile_partition, validate_partition

__all__ = [
    "PartitionError",
    "block_signature",
    "sequence_signatures",
    "tile_partition",
    "validate_partition",
]
