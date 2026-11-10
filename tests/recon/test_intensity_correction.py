"""Receive-field intensity correction: what it flattens, what it keeps and when it stands aside."""

import sys
from types import SimpleNamespace

import ismrmrd.xsd
import numpy as np
import pytest

from pulserver.recon import ReconResult, _correct
from pulserver.recon._correct import INTENSITY_CORRECTION, IntensityCorrection, _stated

pytest.importorskip("SimpleITK")
pytest.importorskip("bartorch.tools")

HEADER_XML = """<?xml version="1.0"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <experimentalConditions><H1resonanceFrequency_Hz>63500000</H1resonanceFrequency_Hz></experimentalConditions>
  <encoding>
    <encodedSpace><matrixSize><x>16</x><y>16</y><z>1</z></matrixSize><fieldOfView_mm><x>240</x><y>240</y><z>5</z></fieldOfView_mm></encodedSpace>
    <reconSpace><matrixSize><x>16</x><y>16</y><z>1</z></matrixSize><fieldOfView_mm><x>240</x><y>240</y><z>5</z></fieldOfView_mm></reconSpace>
    <encodingLimits/>
    <trajectory>cartesian</trajectory>
  </encoding>{parameters}
</ismrmrdHeader>
"""


def header(value=None):
    """A header stating ``IntensityCorrection`` as ``value``, or not at all."""
    parameters = (
        ""
        if value is None
        else (
            "\n  <userParameters><userParameterString>"
            f"<name>{INTENSITY_CORRECTION}</name><value>{value}</value>"
            "</userParameterString></userParameters>"
        )
    )
    return ismrmrd.xsd.CreateFromDocument(HEADER_XML.format(parameters=parameters))


def unit(**counters):
    return SimpleNamespace(counters=counters)


def shaded_cylinder(n=32):
    """A uniform cylinder ``(n, n, n)``, its shading centre-bright, and its mask."""
    z, y, x = np.meshgrid(*(np.linspace(-1, 1, n),) * 3, indexing="ij")
    inside = (x**2 + y**2 < 0.7) & (np.abs(z) < 0.8)
    shading = np.exp(-0.8 * (x**2 + y**2 + z**2))
    return inside * shading, inside


def spread(image, inside):
    """Coefficient of variation of ``image`` over ``inside``."""
    return float(np.std(image[inside]) / np.mean(image[inside]))


def test_a_shaded_uniform_object_is_flattened_and_keeps_its_mean():
    image, inside = shaded_cylinder()

    corrected = IntensityCorrection()(
        ReconResult(image.astype(np.float32)), unit()
    ).data

    assert spread(corrected, inside) < 0.25 * spread(image, inside)
    assert np.mean(corrected[inside]) == pytest.approx(np.mean(image[inside]), rel=0.1)


def test_the_magnitude_is_divided_and_the_real_and_imaginary_images_are_sent_as_reconstructed():
    magnitude, inside = shaded_cylinder()
    x = np.linspace(-1, 1, magnitude.shape[-1])
    image = (magnitude * np.exp(1j * 2.0 * x)).astype(np.complex64)
    results = [
        ReconResult(np.abs(image)),
        ReconResult(image.real, series_index=1, image_type="real"),
        ReconResult(image.imag, series_index=2, image_type="imaginary"),
    ]

    out = IntensityCorrection()(results, unit())

    assert spread(out[0].data, inside) < 0.25 * spread(np.abs(image), inside)
    assert out[0].attributes["ImageProcessingHistory"][-1] == "N4"
    assert out[1] is results[1] and out[2] is results[2]


def test_later_contrasts_and_repetitions_of_a_slice_share_its_first_field():
    image, inside = shaded_cylinder()
    correction = IntensityCorrection()
    first = correction(ReconResult(image.astype(np.float32)), unit(slice=0)).data
    later = correction(
        ReconResult(0.5 * image.astype(np.float32)), unit(slice=0, repetition=3)
    ).data

    np.testing.assert_allclose(later[inside], 0.5 * first[inside], rtol=1e-5)
    assert len(correction.fields) == 1
    correction(ReconResult(image.astype(np.float32)), unit(slice=1))
    assert len(correction.fields) == 2


def test_a_plane_is_corrected_in_two_dimensions():
    image, inside = shaded_cylinder()
    plane = image[16:17].astype(np.float32)

    corrected = IntensityCorrection()(ReconResult(plane), unit()).data

    assert corrected.shape == plane.shape
    assert spread(corrected, inside[16:17]) < 0.5 * spread(plane, inside[16:17])


@pytest.mark.parametrize(
    "result",
    [
        ReconResult(np.ones((2, 3, 4, 5), np.float32)),
        ReconResult(np.ones((8, 8), np.float32), image_type="phase"),
        ReconResult(np.zeros((32, 32), np.float32)),
        None,
    ],
    ids=["four-dimensional", "phase-only", "empty", "no-image"],
)
def test_what_cannot_be_corrected_is_returned_as_it_stands(result):
    assert IntensityCorrection()(result, unit()) is result


def test_without_simpleitk_the_images_are_sent_uncorrected_and_the_log_says_why(
    monkeypatch, caplog
):
    monkeypatch.setitem(sys.modules, "SimpleITK", None)
    result = ReconResult(shaded_cylinder()[0].astype(np.float32))

    with caplog.at_level("WARNING", logger="pulserver.recon"):
        assert IntensityCorrection()(result, unit()) is result
    assert "not intensity corrected" in caplog.text


def test_a_failing_estimate_sends_the_images_uncorrected_and_the_log_says_why(
    monkeypatch, caplog
):
    def failing(magnitude):
        raise RuntimeError("N4 diverged")

    monkeypatch.setattr(_correct, "receive_field", failing)
    result = ReconResult(shaded_cylinder()[0].astype(np.float32))

    with caplog.at_level("ERROR", logger="pulserver.recon"):
        assert IntensityCorrection()(result, unit()) is result
    assert "N4 diverged" in caplog.text


@pytest.mark.parametrize(
    ("value", "default", "stated"),
    [
        (None, True, True),
        (None, False, False),
        ("0", True, False),
        ("0.0", True, False),
        ("1", False, True),
        ("on", True, True),
    ],
)
def test_the_recon_parameter_overrides_the_plugins_default(value, default, stated):
    assert _stated(header(value), INTENSITY_CORRECTION, default) is stated
