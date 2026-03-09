"""Tests for SequenceCollection construction and basic accessors."""

import numpy as np
import pypulseq as pp
import pytest

from pulserver import SequenceCollection


class TestConstruction:
    """SequenceCollection construction from various inputs."""

    def test_from_sequence_object(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        assert sc.num_sequences == 1

    def test_from_sequence_list(self, simple_gre_seq):
        sc = SequenceCollection([simple_gre_seq])
        assert sc.num_sequences == 1

    def test_from_file(self, simple_gre_seq, tmp_path):
        seq_file = tmp_path / 'test.seq'
        simple_gre_seq.write(str(seq_file))
        sc = SequenceCollection(str(seq_file))
        assert sc.num_sequences == 1

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            SequenceCollection(42)


class TestAccessors:
    """Basic accessor methods."""

    def test_num_blocks(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        n = sc.num_blocks(0)
        assert n > 0

    def test_get_block(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        block = sc.get_block(0, 0)
        assert block is not None

    def test_get_sequence_copy(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        seq_copy = sc.get_sequence(0)
        assert isinstance(seq_copy, pp.Sequence)

    def test_get_sequence_out_of_range(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        with pytest.raises(IndexError):
            sc.get_sequence(99)

    def test_num_segments(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        ns = sc.num_segments(0)
        assert ns >= 1

    def test_tr_size(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        ts = sc.tr_size(0)
        assert ts >= 1

    def test_segment_size(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        ns = sc.num_segments(0)
        for i in range(ns):
            sz = sc.segment_size(0, i)
            assert sz >= 1


class TestReport:
    """Report method."""

    def test_report_returns_list(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        reps = sc.report()
        assert isinstance(reps, list)
        assert len(reps) >= 1

    def test_report_fields(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        reps = sc.report()
        r = reps[0]
        assert hasattr(r, 'num_blocks')
        assert hasattr(r, 'segments')
        assert hasattr(r, 'tr_size')
        assert hasattr(r, 'tr_duration_s')
        assert r.tr_size >= 1
        assert r.tr_duration_s > 0

    def test_report_print(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        text = sc.report(do_print=True)
        assert isinstance(text, str)
        assert 'Subsequence' in text


class TestCheck:
    """Safety and consistency checks."""

    def test_check_passes(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        # Should not raise for a well-formed GRE
        sc.check()

    def test_check_with_pns(self, simple_gre_seq):
        sc = SequenceCollection(simple_gre_seq)
        # PNS check with generous threshold
        sc.check(
            stim_threshold=1e9,
            decay_constant_us=360.0,
            pns_threshold_percent=1000.0,
        )
