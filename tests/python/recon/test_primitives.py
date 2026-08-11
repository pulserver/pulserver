"""Tests for Pulserver's reconstruction integration glue."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pulserver.recon.algorithms as algorithms
import pulserver.recon.calibration as calibration
import pulserver.recon._mrd.epi as epi
from pulserver.recon._mrd.grouping import (
    filter_acquisitions,
    group_by_labels,
    split_on_flag,
)
from pulserver.recon._mrd.metadata import (
    MrdMetadata,
    acquisition_labels,
    has_acquisition_flag,
    user_parameter,
)
from pulserver.recon.sms import SmsEpiInputs


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


def test_recon_examples_are_importable_from_pulserver():
    from pulserver.examples.recon import prepare_sms_epi

    inputs = prepare_sms_epi(
        imaging=object(),
        multiband_factor=2,
        caipi_encoding=object(),
        coil_maps=object(),
    )
    assert isinstance(inputs, SmsEpiInputs)
