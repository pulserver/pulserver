"""The label vocabulary, and its correspondence with ISMRMRD."""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.seqzoo import gre_2d
from pulserver._labels import (
    COUNTER_LABELS,
    FLAG_LABELS,
    MRD_COUNTERS,
    MRD_FLAGS,
    SCANNER_FLAGS,
    canonical_label,
)

ismrmrd = pytest.importorskip("ismrmrd")


# --------------------------------------------------------------------------
# the map, against ISMRMRD itself
# --------------------------------------------------------------------------


def test_every_ismrmrd_acquisition_flag_has_a_name():
    """Exactly ISMRMRD's set, so the table cannot fall behind it unnoticed."""
    assert set(MRD_FLAGS.values()) == {
        name for name in dir(ismrmrd) if name.startswith("ACQ_")
    }
    assert len(MRD_FLAGS) == len(set(MRD_FLAGS.values())), "two labels share a flag"
    assert set(MRD_FLAGS) <= set(FLAG_LABELS)


def test_every_encoding_counter_has_a_name():
    """The nine scalar counters, plus the eight of the user array.

    ``kspace_encode_step_0`` is absent from both sides: the header's
    ``encodingLimits`` carries one as the readout extent, but a readout is the
    samples inside an acquisition rather than something counted across them, so
    there is no such counter to label.
    """
    fields = {name for name, _ in ismrmrd.EncodingCounters._fields_}
    named = {mrd for mrd in MRD_COUNTERS.values() if not mrd.startswith("user")}

    assert named == fields - {"user"}
    assert "kspace_encode_step_0" not in fields
    assert sum(mrd.startswith("user") for mrd in MRD_COUNTERS.values()) == 8
    assert set(MRD_COUNTERS) <= set(COUNTER_LABELS)


def test_the_scanner_flow_labels_stop_at_the_scanner():
    """These steer the interpreter, so none of them is an ISMRMRD flag."""
    assert not set(SCANNER_FLAGS) & set(MRD_FLAGS)
    assert set(SCANNER_FLAGS) <= set(FLAG_LABELS)


def test_a_counter_and_a_flag_never_share_a_name():
    assert not set(COUNTER_LABELS) & set(FLAG_LABELS)


# --------------------------------------------------------------------------
# either spelling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "canonical"),
    [
        ("kspace_encode_step_1", "LIN"),
        ("KSPACE_ENCODE_STEP_2", "PAR"),
        ("slice", "SLC"),
        ("contrast", "ECO"),
        ("user_3", "USER3"),
        ("k1", "LIN"),
        ("K2", "PAR"),
        ("ACQ_LAST_IN_SLICE", "LASTSLC"),
        ("LAST_IN_SEGMENT", "LASTSEG"),
        ("ACQ_IS_NOISE_MEASUREMENT", "NOISE"),
        ("acq_is_phasecorr_data", "PHASECORR"),
        ("LIN", "LIN"),
        ("lin", "LIN"),
    ],
)
def test_an_ismrmrd_name_means_the_label_it_maps_to(written, canonical):
    assert canonical_label(written) == canonical
    assert pp.make_label(written, "SET", 1).label == canonical


def test_a_name_the_vocabulary_does_not_know_passes_through_untouched():
    """What lets a sequence carry bookkeeping of its own invention."""
    assert canonical_label("respiratory_bin") == "respiratory_bin"
    assert pp.make_label("respiratory_bin", "SET", 4).label == "respiratory_bin"


def test_k0_is_refused_rather_than_taken_as_a_custom_label():
    """The one name where passing through would be the wrong kindness.

    Someone writing ``k0, k1, k2`` for symmetry would otherwise get two
    counters and one label that nothing consumes, with nothing said about it.
    """
    for spelling in ("k0", "K0"):
        with pytest.raises(ValueError, match="is not a label"):
            canonical_label(spelling)
        with pytest.raises(ValueError, match="samples inside one acquisition"):
            pp.make_label(spelling, "SET", 0)


def test_the_vocabulary_is_what_get_supported_labels_reports():
    supported = set(pp.get_supported_labels())
    assert set(MRD_FLAGS) <= supported
    assert set(MRD_COUNTERS) <= supported
    assert {"USER0", "USER7", "FIRSTSLC", "LASTSCAN", "PHASECORR", "SMS"} <= supported


def test_a_new_name_survives_a_file_round_trip(tmp_path):
    """It reaches a reconstruction through the file, so the file must keep it."""
    seq = pp.Sequence()
    seq.add_block(pp.make_delay(1e-3), pp.make_label("acq_last_in_measurement", "SET", 1))
    seq.add_block(pp.make_delay(1e-3), pp.make_label("user_2", "SET", 7))

    path = tmp_path / "vocabulary.seq"
    seq.write(path)
    text = path.read_text()
    assert "LASTSCAN" in text and "USER2" in text

    back = pp.Sequence()
    back.read(path)
    assert back.evaluate_labels() == {"LASTSCAN": 1, "USER2": 7}


# --------------------------------------------------------------------------
# the recon side reads them back under either spelling
# --------------------------------------------------------------------------


def test_a_plugin_may_name_a_flag_the_way_the_sequence_did():
    from pulserver.recon import has_acquisition_flag

    acquisition = ismrmrd.Acquisition()
    acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)

    assert has_acquisition_flag(acquisition, "LASTSEG")
    assert has_acquisition_flag(acquisition, "ACQ_LAST_IN_SEGMENT")
    assert not has_acquisition_flag(acquisition, "LASTSLC")
    with pytest.raises(ValueError, match="Unknown ISMRMRD acquisition flag"):
        has_acquisition_flag(acquisition, "NOT_A_FLAG")


def test_every_named_flag_resolves_against_ismrmrd():
    """The whole table, not a sample of it: a typo in one row is a dead name."""
    from pulserver.recon import has_acquisition_flag

    acquisition = ismrmrd.Acquisition()
    for name, constant in MRD_FLAGS.items():
        acquisition.clearAllFlags()
        acquisition.setFlag(getattr(ismrmrd, constant))
        assert has_acquisition_flag(acquisition, name), name


# --------------------------------------------------------------------------
# boundary flags, against a sequence that computes its own
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prescription",
    [
        {"n_slices": 4, "acceleration": 2, "n_acs": 8},
        {"n_slices": 1, "acceleration": 1, "n_acs": 8},
        {"n_slices": 3, "acceleration": 3, "n_acs": 12},
    ],
)
def test_a_boundary_lands_on_the_last_acquisition_of_its_unit(prescription):
    """The rule, stated as the property rather than as the derivation.

    A frame boundary closes a slice, an encoding boundary closes a segment
    *within* a slice. ``gre_2d`` is a scan where both are known from the
    outside: it acquires a calibration block and then the rest, so ``LASTSEG``
    has to fire twice per slice and ``LASTSLC`` once.
    """
    seq = gre_2d.main(n_x=32, n_y=32, n_dummy=0, **prescription)
    labels = seq.evaluate_labels(evolution="adc")
    n_slices = prescription["n_slices"]
    n_adc = len(labels["LIN"])
    slices = labels["SLC"] if "SLC" in labels else np.zeros(n_adc, dtype=int)

    for flag, keys in (
        ("LASTSLC", (slices,)),
        ("LASTSEG", (slices, labels["SEG"])),
    ):
        if flag not in labels:
            # A single-slice scan has no slice counter, so no slice boundary.
            assert flag == "LASTSLC" and n_slices == 1
            continue
        last_of = {}
        for index, key in enumerate(zip(*keys, strict=True)):
            last_of[key] = index
        assert np.array_equal(
            np.flatnonzero(labels[flag]), sorted(last_of.values())
        ), flag

    assert labels["LASTSEG"].sum() == 2 * n_slices
    if n_slices > 1:
        assert labels["LASTSLC"].sum() == n_slices


def test_auto_label_leaves_a_counter_the_sequence_wrote_alone():
    """Detection fills the gaps around what is there; it does not correct it."""
    seq = gre_2d.main(n_x=32, n_y=32, n_slices=2, acceleration=2, n_acs=8, n_dummy=0)
    detected = seq.evaluate_labels(evolution="adc")["LIN"]

    # Something the trajectory plainly disagrees with, written as a counter.
    invented = np.arange(len(detected)) % 3
    seq.auto_label(use_labels={"LIN": invented}, boundary_flags=False)
    assert np.array_equal(seq.evaluate_labels(evolution="adc")["LIN"], invented)

    seq.auto_label()
    after = seq.evaluate_labels(evolution="adc")
    assert np.array_equal(after["LIN"], invented)
    assert "LASTSCAN" in after


def test_running_it_twice_changes_nothing_the_first_run_decided():
    seq = gre_2d.main(n_x=32, n_y=32, n_slices=3, acceleration=2, n_acs=8, n_dummy=0)
    once = seq.evaluate_labels(evolution="adc")

    seq.auto_label()
    twice = seq.evaluate_labels(evolution="adc")

    assert set(once) == set(twice)
    for name, values in once.items():
        assert np.array_equal(twice[name], values), name


def test_the_scan_ends_once():
    seq = gre_2d.main(n_x=32, n_y=32, n_slices=4, acceleration=2, n_acs=8, n_dummy=0)
    seq.auto_label()
    last_scan = seq.evaluate_labels(evolution="adc")["LASTSCAN"]
    assert np.array_equal(np.flatnonzero(last_scan), [len(last_scan) - 1])


def test_boundary_flags_can_be_restricted_or_refused():
    seq = gre_2d.main(n_x=32, n_y=32, n_slices=2, acceleration=1, n_acs=8, n_dummy=0)

    none, _ = seq.auto_label(skip_apply=True, boundary_flags=False)
    assert not set(none) & set(MRD_FLAGS)

    # Either spelling, and only what was asked for.
    some, _ = seq.auto_label(skip_apply=True, boundary_flags=["ACQ_LAST_IN_SLICE"])
    assert set(some) & set(MRD_FLAGS) == {"LASTSLC"}

    with pytest.raises(ValueError, match="boundary_flags does not name"):
        seq.auto_label(skip_apply=True, boundary_flags=["SLC"])
