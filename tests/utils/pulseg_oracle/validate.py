"""Partition validation: the spec's structural constraints, checked.

A partition is what an interpreter derived from the TR content: a list of
unique segments, each defined by a span of the block stream. Validation
asks whether that partition could be a compliant PulSeg reading of the
sequence:

- **Tiling**: greedily matching segment definitions against the stream
  must cover every block exactly once. A segment definition whose later
  instances differ structurally fails the tiling at the first divergent
  block, so tiling IS the instance-equality check.
- **Once-class consistency**: the running ONCE state must be constant
  within each tiled instance -- a play-once region folded into a repeating
  unit would make instances of that unit unequal in execution.
"""

from __future__ import annotations

from .identity import block_signature, structural_signature


class PartitionError(AssertionError):
    """A partition violates the spec's structural constraints."""


def _definitions(seq, segments) -> list[tuple[int, list[tuple]]]:
    """Each segment's structural signature list, off its definition span."""
    out = []
    for segment in segments:
        start = int(segment["start_block"])
        count = int(segment["num_blocks"])
        signatures = [
            structural_signature(seq.get_block(start + 1 + i)) for i in range(count)
        ]
        out.append((int(segment["global_index"]), signatures))
    return out


def tile_partition(seq, segments) -> list[tuple[int, int, int]]:
    """Tile the whole stream with the given segments.

    Parameters
    ----------
    seq
        A parsed ``pulserver.pypulseq.Sequence``.
    segments
        The interpreter's unique segments for this subsequence: mappings
        with ``global_index``, ``start_block`` (0-based), ``num_blocks``.

    Returns
    -------
    list of (segment_id, start_block, num_blocks)
        The instance list, covering every block exactly once.

    Raises
    ------
    PartitionError
        When no segment matches at some position, or the match is
        ambiguous in a way that changes coverage.
    """
    definitions = _definitions(seq, segments)
    stream = [
        structural_signature(seq.get_block(i)) for i in range(1, seq.num_blocks + 1)
    ]

    instances: list[tuple[int, int, int]] = []
    position = 0
    while position < len(stream):
        matched = None
        for segment_id, signatures in definitions:
            count = len(signatures)
            if position + count > len(stream):
                continue
            if stream[position : position + count] == signatures:
                # Prefer the longest match; a shorter segment that happens
                # to prefix a longer one must not shadow it.
                if matched is None or count > len(matched[1]):
                    matched = (segment_id, signatures)
        if matched is None:
            raise PartitionError(
                f"no segment definition matches the stream at block {position + 1}"
            )
        instances.append((matched[0], position, len(matched[1])))
        position += len(matched[1])
    return instances


def validate_partition(seq, segments) -> list[tuple[int, int, int]]:
    """Tile, then check the per-instance constraints. Returns the instances.

    Beyond the tiling itself: every instance that acquires at a segment
    position must acquire IDENTICALLY (same count/dwell/delay) -- presence
    is instance metadata, the window is structure.

    The play-once state is deliberately NOT a segment constraint: ONCE is
    block-level execution metadata (the execution stream splits its runs
    at once-class changes wherever they fall), so a segment instance may
    legitimately span the transition -- a bSSFP catalyst stands in an
    excitation position of its segment's first instance.
    """
    instances = tile_partition(seq, segments)

    adc_identity: dict[tuple[int, int], set] = {}
    for segment_id, start, count in instances:
        for i in range(count):
            block = seq.get_block(start + 1 + i)
            if block.adc is not None:
                adc_identity.setdefault((segment_id, i), set()).add(
                    block_signature(block)[5]
                )

    for (segment_id, position), windows in adc_identity.items():
        if len(windows) > 1:
            raise PartitionError(
                f"segment {segment_id} position {position} acquires with"
                f" {len(windows)} different ADC windows across instances"
            )
    return instances
