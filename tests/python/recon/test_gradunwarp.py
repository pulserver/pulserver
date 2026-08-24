"""Tests for vendor-neutral gradient nonlinearity correction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pulserver.recon._server.gradunwarp import MrdCoefficientAccessor
from pulserver.recon._gradunwarp import _evaluate_harmonics
from pulserver.recon.postprocessing import (
    GradientCoefficients,
    Gradunwarp,
    ImageGeometry,
)


def _zero_coefficients(order: int = 1) -> GradientCoefficients:
    values = np.zeros((3, order + 1, order + 1))
    return GradientCoefficients("unnormalized", values, values)


def _dat_payload(*, scale10: float = 0.0) -> str:
    lines = ["GRADWARPTYPE 1"]
    for axis, multiplier in zip("XYZ", (1.0, 2.0, 3.0), strict=True):
        for order in range(1, 11):
            value = multiplier * order if order < 10 else scale10
            lines.append(f"SCALE{axis}{order} {value}")
    lines.append("DELTA 0.0")
    return "\n".join(lines)


def _legacy_mrd_header(key: str, value: str) -> SimpleNamespace:
    parameter = SimpleNamespace(name=key, value=value)
    return SimpleNamespace(
        userParameters=SimpleNamespace(userParameterString=[parameter])
    )


def test_dat_conversion_extends_beta_converter_to_ninth_order() -> None:
    coefficients = GradientCoefficients.from_file(_dat_payload())

    assert coefficients.max_order == 9
    assert coefficients.alpha[0, 3, 1] == pytest.approx(3.0 * 2.0 / 3.0)
    assert coefficients.beta[1, 7, 1] == pytest.approx((2.0 * 7.0) * 16.0 / 7.0)
    assert coefficients.alpha[2, 9, 0] == pytest.approx((3.0 * 9.0) * -128.0)
    assert coefficients.basis == "unnormalized"
    assert "alpha" not in repr(coefficients)
    assert "beta" not in repr(coefficients)


def test_dat_conversion_rejects_undocumented_nonzero_tenth_order() -> None:
    with pytest.raises(ValueError, match=r"SCALE.*10"):
        GradientCoefficients.from_file(_dat_payload(scale10=1e-12))


def test_coefficient_path_and_serialized_text_are_equivalent(tmp_path) -> None:
    path = tmp_path / "coefficients.dat"
    path.write_text(_dat_payload())

    from_path = GradientCoefficients.from_file(path)
    from_text = GradientCoefficients.from_file(_dat_payload())

    np.testing.assert_array_equal(from_path.alpha, from_text.alpha)
    np.testing.assert_array_equal(from_path.beta, from_text.beta)


@pytest.mark.parametrize("binding_style", ["camel", "snake"])
def test_mrd_accessor_reads_keyed_string_parameter(binding_style: str) -> None:
    parameter = SimpleNamespace(name="gradient_coefficients", value=_dat_payload())
    if binding_style == "camel":
        header = SimpleNamespace(
            userParameters=SimpleNamespace(userParameterString=[parameter])
        )
    else:
        header = SimpleNamespace(
            user_parameters=SimpleNamespace(user_parameter_string=[parameter])
        )
    accessor = MrdCoefficientAccessor(header, "gradient_coefficients")

    coefficients = GradientCoefficients.from_file(accessor)

    assert coefficients.max_order == 9
    assert coefficients.source_name == (
        "MRD userParameterString 'gradient_coefficients'"
    )
    assert "SCALEX" not in repr(accessor)


def test_mrd_accessor_reports_missing_key() -> None:
    accessor = MrdCoefficientAccessor(_legacy_mrd_header("other", "value"), "wanted")

    with pytest.raises(KeyError, match="wanted"):
        accessor.read_coefficients()


def test_normalized_coefficient_parser_reads_radius_and_general_degrees() -> None:
    payload = """
    0.25 m = R0, lnorm = 4
      1 A( 3, 0) -0.01 z
    101 A( 3, 1)  0.02 x
    201 B( 5, 3) -0.03 y
    """

    coefficients = GradientCoefficients.from_file(payload)

    assert coefficients.reference_radius_mm == 250.0
    assert coefficients.basis == "normalized"
    assert coefficients.alpha[2, 3, 0] == -0.01
    assert coefficients.alpha[0, 3, 1] == 0.02
    assert coefficients.beta[1, 5, 3] == -0.03


def test_mrd_geometry_uses_reconstructed_matrix_for_zero_fill() -> None:
    header = SimpleNamespace(
        field_of_view=(240.0, 240.0, 5.0),
        position=(10.0, -20.0, 4.0),
        read_dir=(-1.0, 0.0, 0.0),
        phase_dir=(0.0, 1.0, 0.0),
        slice_dir=(0.0, 0.0, 1.0),
    )

    geometry = ImageGeometry.from_mrd(header, (256, 512))

    np.testing.assert_allclose(geometry.fov_mm, (240.0, 240.0))
    np.testing.assert_allclose(geometry.voxel_size_mm, (240 / 256, 240 / 512))
    np.testing.assert_allclose(
        geometry.direction,
        np.asarray(((0.0, -1.0), (1.0, 0.0), (0.0, 0.0))),
    )
    np.testing.assert_allclose(geometry.center_mm, header.position)


def test_mrd_geometry_supports_new_binding_and_3d_axis_order() -> None:
    header = SimpleNamespace(
        field_of_view=np.asarray((60.0, 80.0, 20.0)),
        position=np.asarray((1.0, 2.0, 3.0)),
        col_dir=np.asarray((-1.0, 0.0, 0.0)),
        line_dir=np.asarray((0.0, 1.0, 0.0)),
        slice_dir=np.asarray((0.0, 0.0, 1.0)),
    )

    geometry = ImageGeometry.from_mrd(header, (10, 40, 30))

    np.testing.assert_allclose(geometry.voxel_size_mm, (2.0, 2.0, 2.0))
    np.testing.assert_allclose(
        geometry.direction,
        np.asarray(((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))),
    )


def test_affine_geometry_preserves_oblique_physical_grid() -> None:
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag((1.2, 1.5, 2.0))
    affine[:3, 3] = (4.0, -5.0, 6.0)
    shape = (8, 10, 12)

    geometry = ImageGeometry.from_affine(affine, shape)
    index = np.asarray((2.0, 3.0, 4.0))

    expected = (affine @ np.append(index, 1.0))[:3]
    np.testing.assert_allclose(geometry.indices_to_scanner(index), expected)
    np.testing.assert_allclose(geometry.scanner_to_indices(expected), index)
    np.testing.assert_allclose(geometry.fov_mm, (9.6, 15.0, 24.0))


def test_zero_coefficients_are_identity_for_real_and_complex_batches() -> None:
    angle = np.deg2rad(20.0)
    direction = np.asarray(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
            (0.0, 0.0),
        )
    )
    geometry = ImageGeometry((7, 8), (70.0, 80.0), direction, np.zeros(3))
    correct = Gradunwarp(_zero_coefficients(), geometry)
    rng = np.random.default_rng(4)
    image = rng.normal(size=(2, *geometry.shape)).astype(np.float32)
    complex_image = image + 1j * image[::-1]

    np.testing.assert_allclose(correct(image), image, atol=2e-5)
    np.testing.assert_allclose(correct(complex_image), complex_image, atol=2e-5)
    np.testing.assert_allclose(correct.jacobian_grid(), 1.0)


def test_output_geometry_controls_cubic_resampling_grid() -> None:
    input_geometry = ImageGeometry(
        (9, 9),
        (90.0, 90.0),
        np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
        np.zeros(3),
    )
    output_geometry = ImageGeometry(
        (17, 17),
        (80.0, 80.0),
        input_geometry.direction,
        np.zeros(3),
    )
    physical = input_geometry.scanner_grid()
    image = (2.0 * physical[..., 0] - 3.0 * physical[..., 1] + 7.0).astype(np.float32)
    correct = Gradunwarp(
        _zero_coefficients(),
        input_geometry,
        output_geometry,
        jacobian=False,
    )

    result = correct(image)
    expected_grid = output_geometry.scanner_grid()
    expected_indices = input_geometry.scanner_to_indices(expected_grid)

    assert result.shape == output_geometry.shape
    assert np.isfinite(result).all()
    np.testing.assert_allclose(correct.sampling_grid(), expected_indices, atol=1e-12)


def test_full_3d_jacobian_is_used_for_a_2d_plane() -> None:
    alpha = np.zeros((3, 2, 2))
    beta = np.zeros_like(alpha)
    alpha[0, 1, 1] = 0.01
    coefficients = GradientCoefficients("unnormalized", alpha, beta)
    geometry = ImageGeometry(
        (11, 12),
        (110.0, 120.0),
        np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
        np.zeros(3),
    )
    correct = Gradunwarp(coefficients, geometry)

    # P_1^1(cos(theta)) cos(phi) * r = -x. The output-to-source
    # convention therefore yields source_x = 1.01*x and determinant 1.01.
    np.testing.assert_allclose(correct.jacobian_grid(), 1.01, atol=2e-8)


def test_cartesian_dat_recurrence_matches_second_order_solid_harmonics() -> None:
    alpha = np.zeros((3, 3, 3))
    beta = np.zeros_like(alpha)
    alpha[0, 2, 1] = 0.2
    beta[1, 2, 1] = -0.3
    alpha[2, 2, 0] = 0.4
    coefficients = GradientCoefficients("unnormalized", alpha, beta)
    coordinates = np.asarray(((20.0, -30.0, 40.0), (-10.0, 50.0, -20.0)))
    x, y, z = (coordinates / 10.0).T
    radius_squared = x * x + y * y + z * z
    expected = 10.0 * np.stack(
        (
            0.2 * (-3.0 * x * z),
            -0.3 * (-3.0 * y * z),
            0.4 * (3.0 * z * z - radius_squared) / 2.0,
        ),
        axis=-1,
    )

    np.testing.assert_allclose(
        _evaluate_harmonics(coefficients, coordinates),
        expected,
        atol=1e-12,
    )


def test_compiled_3d_jacobian_uses_all_three_physical_derivatives() -> None:
    alpha = np.zeros((3, 2, 2))
    beta = np.zeros_like(alpha)
    alpha[0, 1, 1] = 0.01
    beta[1, 1, 1] = 0.02
    alpha[2, 1, 0] = 0.03
    coefficients = GradientCoefficients("unnormalized", alpha, beta)
    angle = np.deg2rad(23.0)
    direction = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    geometry = ImageGeometry((9, 10, 11), (90.0, 100.0, 110.0), direction, np.zeros(3))

    determinant = Gradunwarp(coefficients, geometry).jacobian_grid()

    np.testing.assert_allclose(
        determinant[1:-1, 1:-1, 1:-1],
        1.01 * 1.02 * 0.97,
        atol=1e-12,
    )


def test_from_file_accepts_mrd_transport_directly() -> None:
    geometry = ImageGeometry(
        (8, 9),
        (80.0, 90.0),
        np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
        np.zeros(3),
    )
    accessor = MrdCoefficientAccessor(
        _legacy_mrd_header("gradient_coefficients", _dat_payload()),
        "gradient_coefficients",
    )

    correct = Gradunwarp.from_file(accessor, geometry)

    assert correct.coefficients.max_order == 9
