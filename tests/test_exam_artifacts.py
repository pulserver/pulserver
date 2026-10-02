"""The maps the series of an exam share, as a hook reads and writes them."""

import pytest
from conftest import exam_header

import pulserver.recon as recon
from pulserver.recon._runtime.exam import ExamCacheManager


def test_a_hook_reads_a_map_as_an_attribute_of_its_context():
    context = recon.ReconContext.offline()
    context.exam[recon.B0_MAP] = [3.0, 4.0]
    assert context.b0_map == [3.0, 4.0]


def test_a_map_a_hook_assigns_is_stored_for_the_exam():
    """The write a calibration hook makes is the read a later series makes."""
    context = recon.ReconContext.offline()
    context.coil_sensitivities = ["c0", "c1"]
    assert context.exam[recon.COIL_SENSITIVITIES] == ["c0", "c1"]


def test_a_map_no_series_has_measured_reads_as_nothing():
    """A hook that can do without one asks; it does not have to look first."""
    context = recon.ReconContext.offline()
    assert context.b0_map is None
    assert context.b1_map is None
    assert context.coil_sensitivities is None


@pytest.mark.parametrize("name", ["b0_mapp", "b1map", "coil_sensitivity", "tx_field"])
def test_a_name_outside_the_vocabulary_is_refused(name):
    """A misspelt map stored is a map the next series reads nothing from."""
    context = recon.ReconContext.offline()
    with pytest.raises(AttributeError, match=name):
        setattr(context, name, [1.0])
    with pytest.raises(AttributeError, match=name):
        getattr(context, name)


def test_the_state_of_a_stream_starts_empty_and_stays_out_of_the_exam():
    """Per-slice maps, prewhitening and compression belong to the series that made them."""
    context = recon.ReconContext.offline()
    assert context.coil_maps == {}
    assert context.noise is None
    assert context.coil_compression is None

    context.coil_maps[0] = "maps"
    context.noise = "whitening"
    context.coil_compression = "basis"

    assert (context.coil_maps, context.noise, context.coil_compression) == (
        {0: "maps"},
        "whitening",
        "basis",
    )
    assert len(context.exam) == 0


def test_a_plugins_own_artifact_goes_in_the_exam_under_a_key_it_chooses():
    context = recon.ReconContext.offline()
    context.exam["navigator_baseline"] = [0.0]
    assert context.exam["navigator_baseline"] == [0.0]


def test_the_scan_a_context_describes_does_not_change_under_a_hook():
    context = recon.ReconContext(
        header=None, exam=recon.ExamCache("exam-1"), device="cuda:0"
    )
    for name in ("header", "exam", "config", "device"):
        with pytest.raises(AttributeError, match="the scan's"):
            setattr(context, name, None)


def test_a_map_one_series_measures_reaches_the_next_through_the_exam(tmp_path):
    """End to end, across the proxies of a host: the point of the three names."""
    calibration = ExamCacheManager(directory=tmp_path)
    imaging = ExamCacheManager(directory=tmp_path)
    with calibration.lease(exam_header("exam-7")) as cache:
        recon.ReconContext(header=None, exam=cache).b1_map = [1.0, 0.8]
    with imaging.lease(exam_header("exam-7")) as cache:
        assert recon.ReconContext(header=None, exam=cache).b1_map == [1.0, 0.8]
    calibration.close()
    imaging.close()


def test_every_name_is_one_of_the_agreed_keys():
    """An attribute and the key behind it are the same word, or they drift apart."""
    assert set(recon.EXAM_ARTIFACTS) == {
        recon.B0_MAP,
        recon.B1_MAP,
        recon.COIL_SENSITIVITIES,
    }
    context = recon.ReconContext.offline()
    for key in recon.EXAM_ARTIFACTS:
        setattr(context, key, key)
        assert context.exam[key] == key
