"""Tests for pge cache-section readers: info(), trajectory_info(), describe()."""

from pathlib import Path

import pytest

from pge import (
    SequenceCollection,
    SequenceDescriptionInfo,
    SequenceParameters,
    SequenceDescription,
    TrajectoryInfo,
)
from pge.core._cache_sections import build_sequence_description_info

# ── Fixtures ──────────────────────────────────────────────────────────

EXPECTED_DIR = Path(__file__).resolve().parents[1] / "utils" / "expected"

CANONICAL_SEQ = "gre_2d_1sl_1avg.seq"
NONCART_SEQ = "mprage_noncart_3d_1sl_1avg_userotext0.seq"
MULTISUBSEQ_SEQ = "gre_epi_collection_2d_1sl_1avg.seq"


@pytest.fixture(scope="module")
def sc_gre() -> SequenceCollection:
    return SequenceCollection(EXPECTED_DIR / CANONICAL_SEQ)


@pytest.fixture(scope="module")
def sc_noncart() -> SequenceCollection:
    return SequenceCollection(EXPECTED_DIR / NONCART_SEQ)


@pytest.fixture(scope="module")
def sc_collection() -> SequenceCollection:
    return SequenceCollection(EXPECTED_DIR / MULTISUBSEQ_SEQ)


# ── info() — always available ─────────────────────────────────────────


class TestInfo:
    def test_returns_sequence_description_info(self, sc_gre):
        si = sc_gre.info()
        assert isinstance(si, SequenceDescriptionInfo)

    def test_seq_params_populated(self, sc_gre):
        sp = sc_gre.info().seq_params
        assert isinstance(sp, SequenceParameters)
        assert sp.num_subseqs >= 1
        assert sp.min_te_us >= 0
        assert sp.min_tr_us > 0

    def test_subseqs_count_matches_num_sequences(self, sc_gre):
        si = sc_gre.info()
        assert len(si.subseqs) == si.seq_params.num_subseqs

    def test_single_subseq_has_events(self, sc_gre):
        sd = sc_gre.info().subseqs[0]
        assert isinstance(sd, SequenceDescription)
        assert sd.num_rf_events >= 1
        assert sd.num_adc_events >= 1
        assert len(sd.events) >= sd.num_rf_events + sd.num_adc_events

    def test_rf_shape_tuples_present(self, sc_gre):
        sd = sc_gre.info().subseqs[0]
        assert len(sd.rf_shape_tuples) >= 1
        t = sd.rf_shape_tuples[0]
        assert t.N_samples > 0
        assert len(t.mag) == t.N_tx * t.N_samples

    def test_flip_angles_nonempty(self, sc_gre):
        sd = sc_gre.info().subseqs[0]
        fas = sd.flip_angles_deg
        assert len(fas) >= 1
        assert all(0 < fa <= 180 for fa in fas)

    def test_collection_has_two_subseqs(self, sc_collection):
        si = sc_collection.info()
        assert si.seq_params.num_subseqs == 2
        assert len(si.subseqs) == 2

    def test_noncart_info_available(self, sc_noncart):
        si = sc_noncart.info()
        assert si.seq_params.num_subseqs >= 1

    def test_info_is_cached(self, sc_gre):
        si1 = sc_gre.info()
        si2 = sc_gre.info()
        assert si1 is si2  # same object — cached


# ── trajectory_info() — None for Cartesian (no .bin) ─────────────────


class TestTrajectoryInfo:
    def test_cartesian_returns_info(self, sc_gre):
        # gre_2d now has a .bin cache file alongside the fixture (Phase A wiring)
        ti = sc_gre.trajectory_info()
        assert ti is not None
        # Cartesian: no per-shot kspace samples are stored
        assert ti.kshots == []
        assert len(ti.encoding_spaces) >= 1

    def test_noncart_returns_info(self, sc_noncart):
        # noncart fixture now has a .bin cache file alongside it
        ti = sc_noncart.trajectory_info()
        assert ti is not None
        assert len(ti.kshots) >= 1
        assert len(ti.encoding_spaces) >= 1

    def test_trajectory_info_from_in_memory_seq(self, simple_gre_seq):
        """When constructed from a pp.Sequence object (not a path), trajectory_info is None."""
        sc = SequenceCollection(simple_gre_seq)
        assert sc.trajectory_info() is None

    def test_trajectory_info_is_cached(self, sc_gre):
        ti1 = sc_gre.trajectory_info()
        ti2 = sc_gre.trajectory_info()
        # Cached: same object on repeat call
        assert ti1 is ti2


# ── describe() ────────────────────────────────────────────────────────


class TestDescribe:
    def test_returns_nonempty_string(self, sc_gre):
        text = sc_gre.describe(do_print=False)
        assert isinstance(text, str) and len(text) > 0

    def test_contains_sequence_description_header(self, sc_gre):
        text = sc_gre.describe(do_print=False)
        assert "Sequence Description" in text

    def test_contains_min_te(self, sc_gre):
        text = sc_gre.describe(do_print=False)
        assert "Min TE" in text

    def test_noncart_describe(self, sc_noncart):
        text = sc_noncart.describe(do_print=False)
        assert "Sequence Description" in text

    def test_collection_describe(self, sc_collection):
        text = sc_collection.describe(do_print=False)
        assert "Subsequence 0" in text
        assert "Subsequence 1" in text

    @pytest.mark.parametrize(
        "seq_name",
        [
            "gre_2d_1sl_1avg.seq",
            "epi_2d_1sl_1avg.seq",
            "fse_2d_1sl_1avg.seq",
            "bssfp_2d_1sl_1avg.seq",
            "mprage_2d_1sl_1avg.seq",
            "mprage_nav_2d_1sl_1avg.seq",
            "mprage_noncart_3d_1sl_1avg_userotext0.seq",
        ],
    )
    def test_describe_runs_on_all_representative(self, seq_name):
        sc = SequenceCollection(EXPECTED_DIR / seq_name)
        text = sc.describe(do_print=False)
        assert len(text) > 0


# ── report() — extended with section-6 summary ───────────────────────


class TestReportExtended:
    def test_report_returns_list_of_namespaces(self, sc_gre):
        import types

        result = sc_gre.report(do_print=False)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], types.SimpleNamespace)

    def test_report_print_returns_string(self, sc_gre, capsys):
        result = sc_gre.report(do_print=True)
        assert isinstance(result, str)
        assert len(result) > 0
        # seqdesc summary line should be present
        assert "Sequence description" in result or "Description" in result


# ── build_sequence_description_info standalone ───────────────────────


class TestBuildSeqDescInfo:
    def test_from_raw_dicts(self):
        params = {
            "min_te_us": 5000.0,
            "min_tr_us": 7000.0,
            "max_tr_us": 7000.0,
            "max_flip_angle_deg": 30.0,
            "total_scan_time_us": 1e8,
            "num_subseqs": 1,
        }
        desc = {
            "subseq_idx": 0,
            "tr_duration_us": 7000.0,
            "rf_shape_tuples": [],
            "shim_defs": [],
            "events": [],
            "composite_rf_groups": [],
        }
        si = build_sequence_description_info(params, [desc])
        assert si.seq_params.num_subseqs == 1
        assert len(si.subseqs) == 1
        assert si.subseqs[0].tr_duration_us == pytest.approx(7000.0)
