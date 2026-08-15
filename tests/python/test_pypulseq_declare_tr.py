"""``Sequence.declare_tr``: the design side states the structural TR.

The ``TRSize`` definition is the design side's statement of blocks per
structural TR, written where TR detection already ran. Consumers may use it
or ignore it: a reconstruction derives its sequence description only when
the definition is present; an interpreter may take it as a hint and verify.
"""

from __future__ import annotations

import pulserver.pypulseq as pp
from pulserver.seqzoo import gre_2d


def build():
    return gre_2d.main(n_x=32, n_y=32, te=5e-3, tr=30e-3)


def test_declare_writes_the_detected_blocks_per_tr():
    seq = build()
    declared = seq.declare_tr()
    assert declared == seq.tr_size
    assert int(seq.get_definition("TRSize")) == declared


def test_writing_declares_automatically(tmp_path):
    """No explicit call needed: every written file carries the declaration."""
    seq = build()
    expected = seq.tr_size
    path = tmp_path / "declared.seq"
    seq.write(path)

    back = pp.Sequence()
    back.read(path)
    assert int(back.get_definition("TRSize")[0]) == expected


def test_the_binary_format_declares_too(tmp_path):
    seq = build()
    expected = seq.tr_size
    path = tmp_path / "declared.bin"
    seq.write_binary(path)

    back = pp.Sequence()
    back.read(path)
    assert int(back.get_definition("TRSize")[0]) == expected


def test_a_trivial_sequence_declares_its_trivial_structure():
    """Detection always answers -- a one-block scan is one one-block TR.

    ``None`` is reserved for detection genuinely failing, not for scans
    whose structure is merely uninteresting.
    """
    seq = pp.Sequence()
    seq.add_block(pp.make_delay(1e-3))
    assert seq.declare_tr() == seq.tr_size
