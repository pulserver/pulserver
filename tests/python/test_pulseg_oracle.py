"""The PulSeg oracle against the interpreter's content-derived partitions.

Segmentation authority is the TR content: the interpreter detects the
repeating structure and partitions it, and the oracle independently checks
that the partition satisfies the spec's structural constraints -- the
segments tile the stream exactly (which is the instance-equality rule in
executable form) and no play-once region hides inside a repeating unit.
TRID plays no role here: it is an optional SAR/coil-heating refinement
label for interleaved-subsequence hyperTRs, not a segmentation input.
"""

from __future__ import annotations

import pytest

import pulserver.pypulseq as pp
from fixture_corpus import FIXTURES_DIR
from pulseg_oracle import PartitionError, validate_partition
from pulserver._ext import _pulseg_wrapper as wrapper

GAMMA = 42577478.0


def collection_for(paths: list[str]) -> object:
    buffers = [open(path, "rb").read() for path in paths]
    return wrapper._PulseqCollection(
        buffers, GAMMA, 3.0, GAMMA * 1.0, GAMMA * 10000.0,
        1e-6, 10e-6, 0.1e-6, 10e-6, True, 1,
    )


def read(path) -> pp.Sequence:
    seq = pp.Sequence()
    seq.read(path)
    return seq


@pytest.mark.parametrize(
    "stem",
    ["gre_2d", "gre_2d_3sl", "se_2d", "fse_2d", "bssfp_2d", "gre_radial_2d",
     "gre_spiral_2d", "se_propeller_2d", "mprage_stack_of_spirals_3d", "zte_3d"],
)
def test_the_interpreter_partition_is_a_valid_pulseg_reading(stem):
    path = str(FIXTURES_DIR / f"{stem}.seq")
    collection = collection_for([path])
    segments = wrapper._get_segments(collection, 0)
    seq = read(path)

    instances = validate_partition(seq, segments)

    covered = sum(count for _, _, count in instances)
    assert covered == seq.num_blocks
    used = {segment_id for segment_id, _, _ in instances}
    assert used == {int(s["global_index"]) for s in segments}


def test_the_epi_collection_partition_is_blessed():
    """The adjudicated case: 1 GRE segment + 3 EPI segments is a valid
    reading -- the fatsat prep is its own unit because it is structurally
    distinct from the shot body, and carving the frame-counter block
    separately is granularity the interpreter is free to choose."""
    paths = [
        "tests/utils/expected/gre_epi_collection_2d_1sl_1avg.seq",
        "tests/utils/expected/gre_epi_collection_2d_1sl_1avg_epi.seq",
    ]
    collection = collection_for(paths)

    counts = []
    for subseq, path in enumerate(paths):
        segments = wrapper._get_segments(collection, subseq)
        instances = validate_partition(read(path), segments)
        assert sum(c for _, _, c in instances) == read(path).num_blocks
        counts.append(len(segments))

    assert counts == [1, 3]


def test_the_degenerate_whole_scan_partition_is_legal():
    """One segment spanning the whole scan, played once, breaks no spec
    rule -- it is the trivial fallback reading every scan admits. The
    oracle validates compliance, not granularity: play-once regions are
    block-level execution metadata, never a partition constraint."""
    path = "tests/utils/expected/gre_epi_collection_2d_1sl_1avg_epi.seq"
    seq = read(path)

    whole = [{"global_index": 0, "start_block": 0, "num_blocks": seq.num_blocks}]
    instances = validate_partition(seq, whole)
    assert instances == [(0, 0, seq.num_blocks)]


def test_a_partition_that_cannot_tile_is_refused():
    path = str(FIXTURES_DIR / "gre_2d.seq")
    seq = read(path)

    wrong = [{"global_index": 0, "start_block": 0, "num_blocks": 3}]
    with pytest.raises(PartitionError, match="no segment definition matches"):
        validate_partition(seq, wrong)
