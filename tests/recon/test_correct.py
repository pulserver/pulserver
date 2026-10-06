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

#: The same coil with a second-order term, which displaces voxels off axis.
_UNUSED = COIL.replace("SCALEX2 0.000000000e+00", "SCALEX2 5.0e-02")


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

    shape = (1, 16, 16)

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


def test_an_image_of_one_partition_is_corrected_on_the_encoding_matrix(monkeypatch):
    """The plugin drops the partition axis; the correction works on it."""
    pytest.importorskip("bartorch.tools")
    recording = _Recording()
    _patched(monkeypatch, recording)
    image = np.zeros((16, 16), dtype=np.complex64)

    out = gradient_unwarped(image, _Context(header(COIL)), _Data(acquisition()))

    assert tuple(recording.seen.shape) == (1, 16, 16), "the axis must be restored"
    assert out.shape == (16, 16), "and dropped again"
    assert np.iscomplexobj(out), "a complex image must stay complex"
    assert isinstance(out, np.ndarray), "a NumPy image must come back as one"


def test_a_volume_is_passed_through_as_it_stands(monkeypatch):
    pytest.importorskip("bartorch.tools")
    recording = _Recording()
    _patched(monkeypatch, recording)
    image = np.zeros((1, 16, 16), dtype=np.float32)

    gradient_unwarped(image, _Context(header(COIL)), _Data(acquisition()))

    assert tuple(recording.seen.shape) == (1, 16, 16)


def test_a_correction_that_fails_returns_the_image_and_says_so(monkeypatch, caplog):
    pytest.importorskip("bartorch.tools")

    class _Failing(_Recording):
        def __call__(self, tensor):
            raise RuntimeError("no coefficients for this coil")

    _patched(monkeypatch, _Failing())
    image = np.zeros((16, 16), dtype=np.complex64)

    with caplog.at_level("WARNING", logger="pulserver.recon"):
        out = gradient_unwarped(image, _Context(header(COIL)), _Data(acquisition()))

    assert out is image
    assert "not corrected for gradient nonlinearity" in caplog.text
