"""Gradient nonlinearity correction: when it runs, and what it leaves alone."""

import ismrmrd
import ismrmrd.xsd
import numpy as np
import pytest

from pulserver.recon._correct import gradient_unwarped, states_gradient_coefficients

#: A coil whose field is linear: every scale but the first order is zero, so
#: the correction is the identity and a corrected image equals its input.
COIL = "\n".join(
    ["GRADWARPTYPE 1"]
    + [
        f"SCALE{axis}{order} {1.0 if order == 1 else 0.0:.9e}"
        for axis in "XYZ"
        for order in range(1, 11)
    ]
    + ["DELTA 0.0"]
)

#: A 16x16 axial slice; the coil table is inserted as a user parameter.
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


def header(table=None, name="gradient_coefficients"):
    """The header above, stating ``table`` under ``name`` when one is given."""
    parameters = (
        ""
        if table is None
        else (
            "\n  <userParameters><userParameterString>"
            f"<name>{name}</name><value>{table}</value>"
            "</userParameterString></userParameters>"
        )
    )
    return ismrmrd.xsd.CreateFromDocument(HEADER_XML.format(parameters=parameters))


def acquisition():
    """A reference acquisition of an axial slice at isocentre."""
    acq = ismrmrd.Acquisition()
    acq.resize(16, 1)
    acq.read_dir = (1.0, 0.0, 0.0)
    acq.phase_dir = (0.0, 1.0, 0.0)
    acq.slice_dir = (0.0, 0.0, 1.0)
    acq.position = (0.0, 0.0, 0.0)
    return acq


class _Context:
    def __init__(self, head):
        self.header = head


class _Buffer:
    def __init__(self, reference):
        self.reference = reference


class _Data:
    def __init__(self, reference):
        self.data = _Buffer(reference)


def test_a_header_stating_no_table_is_recognised_as_stating_none():
    assert not states_gradient_coefficients(header())
    assert states_gradient_coefficients(header(COIL))


def test_a_table_under_another_accepted_name_is_found():
    assert states_gradient_coefficients(header(COIL, name="coeff_dat"))


def test_an_image_is_returned_untouched_where_the_header_states_no_table():
    image = np.arange(256, dtype=np.complex64).reshape(16, 16)
    out = gradient_unwarped(image, _Context(header()), _Data(acquisition()))
    assert out is image


def test_an_image_is_returned_untouched_where_the_unit_has_no_reference():
    image = np.arange(256, dtype=np.complex64).reshape(16, 16)
    out = gradient_unwarped(image, _Context(header(COIL)), _Data(None))
    assert out is image


class _Recording:
    """A correction standing in for bartorch's, recording what it was given."""

    shape = (4, 16, 16)

    def __init__(self):
        self.seen = None

    def __call__(self, tensor):
        self.seen = tensor
        return tensor + 1


def _patched(monkeypatch, correction):
    """Make ``Gradunwarp.from_mrd`` return ``correction``."""
    import bartorch.tools

    monkeypatch.setattr(
        bartorch.tools.Gradunwarp,
        "from_mrd",
        classmethod(lambda cls, header, acquisition: correction),
    )


def test_a_volume_is_passed_through_as_it_stands(monkeypatch):
    pytest.importorskip("bartorch.tools")
    recording = _Recording()
    _patched(monkeypatch, recording)
    image = np.zeros((4, 16, 16), dtype=np.float32)

    gradient_unwarped(image, _Context(header(COIL)), _Data(acquisition()))

    assert tuple(recording.seen.shape) == (4, 16, 16)


#: A coil whose gradients fall off with the cube of the distance from
#: isocentre, as a short head coil's do.
CUBIC = "\n".join(
    ["GRADWARPTYPE 1"]
    + [
        f"SCALE{axis}{order} {-1.0e-4 if order == 3 else 0.0:.9e}"
        for axis in "XYZ"
        for order in range(1, 11)
    ]
    + ["DELTA 0.0"]
)


@pytest.mark.parametrize("partition_axis", [False, True])
def test_a_slice_far_off_isocentre_keeps_every_voxel(partition_axis):
    """The displacement across a 5 mm slice exceeds half its thickness at 140 mm.

    Corrected as a volume of one partition, the voxels displaced out of it
    would be lost; in its plane, each is resampled from the slice.
    """
    pytest.importorskip("bartorch.tools")
    reference = acquisition()
    reference.position = (0.0, 0.0, 140.0)
    image = np.ones((1, 16, 16) if partition_axis else (16, 16), dtype=np.float32)

    out = gradient_unwarped(image, _Context(header(CUBIC)), _Data(reference))

    assert out.shape == image.shape
    inner = out[..., 4:12, 4:12]
    assert np.all(inner > 0.5), "no voxel of the slice may be lost"
    assert not np.allclose(inner, 1.0), "the slice must still be corrected"


def test_a_correction_that_fails_returns_the_image_and_says_so(monkeypatch, caplog):
    pytest.importorskip("bartorch.tools")

    class _Failing(_Recording):
        def __call__(self, tensor):
            raise RuntimeError("no coefficients for this coil")

    _patched(monkeypatch, _Failing())
    image = np.zeros((4, 16, 16), dtype=np.complex64)

    with caplog.at_level("WARNING", logger="pulserver.recon"):
        out = gradient_unwarped(image, _Context(header(COIL)), _Data(acquisition()))

    assert out is image
    assert "not corrected for gradient nonlinearity" in caplog.text


def centred_fft(image, axes):
    shifted = np.fft.ifftshift(image, axes=axes)
    return np.fft.fftshift(np.fft.fftn(shifted, axes=axes, norm="ortho"), axes=axes)


@pytest.mark.parametrize("shape", [(24, 32), (6, 20, 16), (1, 40, 40)])
def test_the_filtered_image_has_the_k_space_of_the_image_times_the_fermi_window(shape):
    bartorch = pytest.importorskip("bartorch")
    from pulserver.recon import fermi_filtered

    rng = np.random.default_rng(0)
    image = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )
    axes = tuple(axis for axis, n in enumerate(shape) if n > 1)

    filtered = fermi_filtered(image)

    window = bartorch.fermi_window(tuple(shape[axis] for axis in axes)).numpy()
    window = window.reshape([n if n > 1 else 1 for n in shape])
    np.testing.assert_allclose(
        centred_fft(filtered, axes),
        centred_fft(image, axes) * window,
        atol=1e-4 * np.abs(image).max(),
    )
    assert filtered.shape == image.shape
    assert filtered.dtype == np.complex64


def test_a_real_image_stays_real_and_the_edge_of_k_space_is_halved():
    pytest.importorskip("bartorch")
    from pulserver.recon import fermi_filtered

    rng = np.random.default_rng(1)
    image = rng.random((64, 64)).astype(np.complex64)

    filtered = fermi_filtered(image)

    assert np.abs(filtered.imag).max() < 1e-5 * np.abs(filtered.real).max()
    ratio = np.abs(centred_fft(filtered, (0, 1))) / np.abs(centred_fft(image, (0, 1)))
    assert ratio[32, 0] == pytest.approx(0.5, abs=1e-3)
    assert ratio[32, 32] > 0.95


def test_the_partition_profile_is_the_volume_window_through_its_centre():
    bartorch = pytest.importorskip("bartorch")
    from pulserver.recon._correct import fermi_profile

    window = bartorch.fermi_window((8, 64, 48)).numpy()

    np.testing.assert_allclose(fermi_profile((8, 64, 48), 0), window[:, 32, 24])
    assert fermi_profile((8, 64, 48), 0)[0] == pytest.approx(0.5, abs=1e-3)


def test_a_streams_corrections_build_the_unwarp_of_each_geometry_once(monkeypatch):
    pytest.importorskip("bartorch.tools")
    from pulserver.recon import Corrections

    built = []

    def from_mrd(cls, head, reference):
        built.append(tuple(reference.position))
        return _Recording()

    import bartorch.tools

    monkeypatch.setattr(bartorch.tools.Gradunwarp, "from_mrd", classmethod(from_mrd))
    corrections = Corrections(header(COIL))
    image = np.zeros((4, 16, 16), dtype=np.float32)
    first, moved = acquisition(), acquisition()
    moved.position = (0.0, 0.0, 10.0)

    for reference in (first, first, moved, first):
        corrections.unwarped(image, _Context(header(COIL)), _Data(reference))

    assert built == [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]


SHIPPED = ("pics", "pics_train", "nufft", "nufft_train", "epi", "pmc", "qsm")


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shipped_reconstruction_makes_the_three_corrections_by_default(name):
    import importlib

    from conftest import calibration_header

    from pulserver.recon import Corrections, ExamCache, ReconContext

    plugin = importlib.import_module(f"pulserver.recon.handlers.{name}").PLUGIN
    head = calibration_header(4, 16)
    head.userParameters = header(COIL).userParameters
    stream = plugin.spawn()
    stream.startup(ReconContext(header=head, exam=ExamCache("exam")))

    assert isinstance(stream.corrections, Corrections)
    assert stream.corrections.kspace_filter
    assert stream.corrections.gradient_unwarp
    assert stream.corrections.intensity is not None
    assert stream.corrections.intensity.levels == 4
