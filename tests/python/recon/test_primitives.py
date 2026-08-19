"""Tests for Pulserver's reconstruction integration glue."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pulserver.recon.optim as algorithms
import pulserver.recon.calibration as calibration
import pulserver.recon.preprocessing as epi
from pulserver.recon._mrd.grouping import (
    filter_acquisitions,
    group_by_labels,
    split_on_flag,
)
from pulserver.recon._mrd.metadata import (
    MrdMetadata,
    acquisition_labels,
    diffusion_table,
    has_acquisition_flag,
    user_parameter,
)
from pulserver.recon.preprocessing import SmsEpiInputs


def test_polynomial_preconditioner_degree_zero_and_call_count():
    degree_zero = algorithms.PolynomialPreconditioner(
        lambda value: 2 * value,
        degree=0,
    )
    assert degree_zero(2.0) == pytest.approx(3.0)

    calls = []

    def normal(value):
        calls.append(value)
        return 2 * value

    polynomial = algorithms.PolynomialPreconditioner(
        normal,
        degree=3,
        scale=0.5,
    )
    assert np.isfinite(polynomial(1.0))
    assert len(calls) == 3


def test_nlinv_public_api_is_class_based():
    assert calibration.__all__ == [
        "NLINV",
        "NLINVPhysics",
        "NLINVResult",
        "PhasePoleCorrection",
        "WavePSF",
        "WavePSFCalibration",
        "WavePSFResult",
        "calibration_extent",
        "coil_maps_from_reference",
    ]
    assert not hasattr(calibration, "nlinv_sensitivities")
    assert not hasattr(calibration, "estimate_sensitivities")


def _acquisition(slice_number: int, flags: int = 0):
    return SimpleNamespace(
        idx=SimpleNamespace(
            slice=slice_number,
            repetition=0,
            contrast=0,
            phase=0,
            average=0,
            set=0,
            segment=0,
        ),
        encoding_space_ref=0,
        flags=flags,
        is_flag_set=lambda flag: bool(flags & flag),
    )


def test_grouping_and_flag_helpers():
    acquisitions = [_acquisition(0), _acquisition(1, 2), _acquisition(0, 2)]
    groups = group_by_labels(acquisitions, ("repetition", "slice"))
    assert list(groups) == [(0, 0), (0, 1)]
    assert [len(group) for group in groups.values()] == [2, 1]
    assert [len(group) for group in split_on_flag(acquisitions, 2)] == [2, 1]
    assert (
        list(filter_acquisitions(acquisitions, require_flags=(2,))) == acquisitions[1:]
    )
    assert has_acquisition_flag(acquisitions[1], 2)
    assert acquisition_labels(acquisitions[0])["slice"] == 0


def test_metadata_accessors_and_parameter_lookup():
    matrix = SimpleNamespace(x=64, y=32, z=1)
    fov = SimpleNamespace(x=220.0, y=180.0, z=5.0)
    encoding = SimpleNamespace(
        encodedSpace=SimpleNamespace(matrixSize=matrix),
        reconSpace=SimpleNamespace(matrixSize=matrix, fieldOfView_mm=fov),
    )
    header = SimpleNamespace(
        encoding=[encoding],
        userParameters=SimpleNamespace(
            userParameterLong=[SimpleNamespace(name="BitsStored", value=12)]
        ),
    )
    metadata = MrdMetadata(header)
    assert metadata.encoded_matrix() == (64, 32, 1)
    assert metadata.recon_matrix() == (64, 32, 1)
    assert metadata.field_of_view_mm() == (220.0, 180.0, 5.0)
    assert metadata.user_parameter("BitsStored") == 12
    assert user_parameter(header, "missing", "fallback") == "fallback"


def _header_with(**parameters):
    """A header carrying `parameters` as UserParameterStrings."""
    return SimpleNamespace(
        userParameters=SimpleNamespace(
            userParameterString=[
                SimpleNamespace(name=name, value=value)
                for name, value in parameters.items()
            ]
        )
    )


def test_the_diffusion_table_comes_back_out_of_the_header():
    """What the scanner wrote verbatim, read back as the design-side type.

    The tensor is a linear encoding along x with a b-value of 1000 s/mm^2,
    written in s/m^2 the way `write_diffusion_definitions` writes it.
    """
    tensor = np.zeros((2, 3, 3))
    tensor[0, 0, 0] = 1.0e9
    tensor[1, 1, 1] = 5.0e8

    table = diffusion_table(
        _header_with(
            bTensorFixed=" ".join(f"{v!r}" for v in tensor.reshape(-1).tolist()),
            bTensorAxis="SET",
        )
    )
    assert table.axis == "SET"
    np.testing.assert_allclose(table.b_values, [1000.0, 500.0], rtol=1e-12)
    np.testing.assert_allclose(np.abs(table.b_vectors), np.eye(3)[:2], atol=1e-12)
    # Linear encoding: one non-zero eigenvalue, so b_delta is 1.
    np.testing.assert_allclose(table.b_delta, [1.0, 1.0], rtol=1e-12)
    np.testing.assert_allclose(table.mrtrix_table()[:, 3], table.b_values, rtol=1e-12)


def test_a_scan_without_a_diffusion_table_reads_as_none():
    assert diffusion_table(SimpleNamespace(userParameters=None)) is None
    assert diffusion_table(_header_with(bTensorAxis="SET")) is None


def test_the_prescription_turns_the_rotatable_part_only():
    """`NOROT` is why there are two arrays, and this is what it buys.

    The fixed half is what the diffusion preparation played under `NOROT`, so
    the console's FOV rotation must leave it alone; the other half is the
    imaging gradients, which it turns.
    """
    fixed = np.zeros((1, 3, 3))
    fixed[0, 0, 0] = 1.0e9
    rotatable = np.zeros((1, 3, 3))
    rotatable[0, 1, 1] = 1.0e8

    header = _header_with(
        bTensorFixed=" ".join(f"{v!r}" for v in fixed.reshape(-1).tolist()),
        bTensorRotatable=" ".join(f"{v!r}" for v in rotatable.reshape(-1).tolist()),
        bTensorAxis="SET",
    )
    # A quarter turn about z takes the rotatable y term onto x.
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned = diffusion_table(header, rotation=R).b_tensors[0]

    expected = (fixed[0] + R @ rotatable[0] @ R.T) * 1e-6
    np.testing.assert_allclose(turned, expected, atol=1e-12)
    assert turned[0, 0] == pytest.approx(1100.0, rel=1e-12)
    assert turned[1, 1] == pytest.approx(0.0, abs=1e-9)


def test_partition_epi_acquisitions_uses_standard_roles(monkeypatch):
    navigator = _acquisition(0)
    reference = _acquisition(1)
    reverse = _acquisition(2)
    reverse.idx.set = 1
    imaging = _acquisition(3)
    flag_map = {
        id(navigator): {"ACQ_IS_NAVIGATION_DATA"},
        id(reference): {"ACQ_IS_PARALLEL_CALIBRATION"},
    }
    monkeypatch.setattr(
        epi,
        "has_acquisition_flag",
        lambda acquisition, flag: flag in flag_map.get(id(acquisition), set()),
    )

    groups = epi.partition_epi_acquisitions([navigator, reference, reverse, imaging])
    assert groups.phase_correction == [navigator]
    assert groups.single_band_reference == [reference]
    assert groups.reverse_polarity == [reverse]
    assert groups.imaging == [imaging]


def test_sms_inputs_require_caipi_and_a_reconstruction_reference():
    with pytest.raises(ValueError, match="CAIPI"):
        SmsEpiInputs(imaging=object()).validate(multiband_factor=2)
    with pytest.raises(ValueError, match="coil maps or a single-band reference"):
        SmsEpiInputs(imaging=object(), caipi_encoding=object()).validate(
            multiband_factor=2
        )
    SmsEpiInputs(
        imaging=object(), caipi_encoding=object(), coil_maps=object()
    ).validate(multiband_factor=2)
