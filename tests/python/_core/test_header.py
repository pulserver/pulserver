"""Test Sequence Definition."""

import mrd
import pytest

from pulserver._core._header import SequenceDefinition


@pytest.fixture
def seq_def_3d():
    """Fixture to create a 3D sequence definition."""
    return SequenceDefinition(3)


@pytest.fixture
def seq_def_2d():
    """Fixture to create a 2D sequence definition."""
    return SequenceDefinition(2)


def test_initialization(seq_def_3d):
    """Test that initialization correctly sets the ndims and other initial attributes."""
    assert seq_def_3d._ndims == 3
    assert len(seq_def_3d._definition.encoding) == 1
    assert seq_def_3d._shape_set is False
    assert seq_def_3d._current_section == 0
    assert isinstance(seq_def_3d._definition.user_parameters, mrd.UserParametersType)
    assert isinstance(
        seq_def_3d._definition.sequence_parameters, mrd.SequenceParametersType
    )


def test_add_section(seq_def_3d):
    """Test that a new section is added and labels are correctly updated."""
    seq_def_3d.section("first_section")
    assert "first_section" in seq_def_3d._section_labels
    assert seq_def_3d._current_section == 0

    # Add another section
    seq_def_3d.section("second_section")
    assert "second_section" in seq_def_3d._section_labels
    assert seq_def_3d._current_section == 1
    assert len(seq_def_3d._definition.encoding) == 2


def test_inheritance_from_section_0(seq_def_3d):
    """Test that the second section correctly inherits values from section 0."""
    seq_def_3d.section("first_section")

    # Set shape and FOV in section 0
    seq_def_3d.set_definition("shape", 128, 128, 64)
    seq_def_3d.set_definition("fov", 240.0, 240.0, 240.0)

    # Add a second section
    seq_def_3d.section("second_section")

    # Ensure fields are copied from section 0 to section 1
    section_0 = seq_def_3d._definition.encoding[0]
    section_1 = seq_def_3d._definition.encoding[1]

    assert (
        section_1.encoded_space.matrix_size.x == section_0.encoded_space.matrix_size.x
    )
    assert section_1.recon_space.matrix_size.x == section_0.recon_space.matrix_size.x
    assert (
        section_1.encoded_space.field_of_view_mm.x
        == section_0.encoded_space.field_of_view_mm.x
    )


def test_set_shape_3d(seq_def_3d):
    """Test setting the encoded and reconstruction space shape for 3D acquisition."""
    seq_def_3d.set_definition("shape", 128, 128, 64)

    # Verify that shape is correctly set
    encoding = seq_def_3d._definition.encoding[0]
    assert encoding.encoded_space.matrix_size.x == 128
    assert encoding.encoded_space.matrix_size.y == 128
    assert encoding.encoded_space.matrix_size.z == 64
    assert encoding.recon_space.matrix_size.x == 128
    assert encoding.recon_space.matrix_size.y == 128
    assert encoding.recon_space.matrix_size.z == 64


def test_set_fov_3d(seq_def_3d):
    """Test setting the field of view for 3D acquisition."""
    seq_def_3d.set_definition("fov", 240.0, 240.0, 240.0)

    # Verify that FOV is correctly set
    encoding = seq_def_3d._definition.encoding[0]
    assert encoding.encoded_space.field_of_view_mm.x == 240.0
    assert encoding.encoded_space.field_of_view_mm.y == 240.0
    assert encoding.encoded_space.field_of_view_mm.z == 240.0
    assert encoding.recon_space.field_of_view_mm.x == 240.0
    assert encoding.recon_space.field_of_view_mm.y == 240.0
    assert encoding.recon_space.field_of_view_mm.z == 240.0


def test_set_fov_2d_before_shape_error(seq_def_2d):
    """Test that setting FOV before shape in 2D raises an error."""
    with pytest.raises(
        KeyError, match="For 2D acquisitions, 'shape' must be set before 'fov'"
    ):
        seq_def_2d.set_definition("fov", 240.0, 240.0, 5.0)


def test_set_fov_2d(seq_def_2d):
    """Test setting the field of view for 2D acquisition."""
    seq_def_2d.set_definition("shape", 128, 128, 1)
    seq_def_2d.set_definition("fov", 240.0, 240.0, 5.0)

    encoding = seq_def_2d._definition.encoding[0]
    assert encoding.encoded_space.field_of_view_mm.x == 240.0
    assert encoding.encoded_space.field_of_view_mm.y == 240.0
    assert encoding.encoded_space.field_of_view_mm.z == 5.0
    assert encoding.recon_space.field_of_view_mm.x == 240.0
    assert encoding.recon_space.field_of_view_mm.y == 240.0
    assert encoding.recon_space.field_of_view_mm.z == 5.0


def test_set_limits(seq_def_3d):
    """Test setting k-space encoding limits."""
    seq_def_3d.set_definition(
        "limits", n_views=128, n_slices=64, n_partitions=32, n_contrasts=1, n_frames=10
    )

    encoding = seq_def_3d._definition.encoding[0]

    # Check the encoding limits for kspace_encoding_step_1 (views)
    assert encoding.encoding_limits.kspace_encoding_step_1.maximum == 127
    assert encoding.encoding_limits.kspace_encoding_step_1.center == 64

    # Check the encoding limits for slice and partition
    assert encoding.encoding_limits.slice.maximum == 63
    assert encoding.encoding_limits.kspace_encoding_step_2.maximum == 31

    # Check the encoding limits for repetition (frames)
    assert encoding.encoding_limits.repetition.maximum == 9


def test_set_trajectory_type(seq_def_3d):
    """Test setting k-space trajectory type."""
    seq_def_3d.set_definition("trajectory-type", "radial")

    encoding = seq_def_3d._definition.encoding[0]
    assert encoding.trajectory == mrd.Trajectory(2)
