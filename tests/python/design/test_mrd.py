"""Test MRD helpers"""

from types import SimpleNamespace

import numpy as np
import pypulseq as pp
import pytest

from ismrmrd import xsd

from pulserver import tools as mod
from pulserver.tools import ISMRMRDBuilder


class FakeRotation:
    def __init__(self, scale=2.0):
        self.scale = scale

    def apply(self, traj):
        return traj * self.scale


class RecordingSequence:
    """Deterministic pypulseq.Sequence stub that records events."""
    created = []

    def __init__(self, system=None):
        self.system = system
        self.blocks = []
        RecordingSequence.created.append(self)

    def add_block(self, *events):
        self.blocks.append(events)

    def calculate_kspace(self):
        # 1D trajectory with center at index 1
        k_traj_adc = np.array(
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        t_adc = np.array([0.0, 1e-3, 2e-3, 3e-3])
        return k_traj_adc, None, None, None, t_adc


@pytest.fixture
def builder():
    return ISMRMRDBuilder()


def test_add_encoding_increments_and_tracks_current(builder):
    initial_len = len(builder.head.encoding)
    builder.add_encoding()
    assert len(builder.head.encoding) == initial_len + 1
    assert builder.current_encoding == len(builder.head.encoding) - 1


def test_set_limits_defaults_center(builder):
    builder.set_limits("k1", maximum=5)
    limits = builder.head.encoding[builder.current_encoding].encodingLimits.kspace_encoding_step_1
    assert limits.minimum == 0
    assert limits.maximum == 5
    assert limits.center == 3  # ceil(5 / 2)


def test_set_trajectory_rejects_invalid_type(builder):
    with pytest.raises(ValueError):
        builder.set_trajectory(traj_type="not-an-enum")
    with pytest.raises(ValueError):
        builder.set_trajectory(traj_type=xsd.calibrationModeType.EXTERNAL)


def test_parallel_imaging_rejects_invalid_calibration(builder):
    with pytest.raises(ValueError):
        builder.set_parallel_imaging_info(calibration_type="bad")


def test_multiband_rejects_invalid_calibration(builder):
    with pytest.raises(ValueError):
        builder.set_multiband_info(
            calibration_type="bad", mb_factor=2, spacing=5.0, calibration_encoding=0
        )


def test_set_diffusion_mismatched_lengths_raises(builder):
    direction = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="bvalues must match"):
        builder.set_diffusion(
            channel=xsd.diffusionDimensionType.SEGMENT,
            scheme="bipolar",
            direction=direction,
            bvalue=np.array([0.0, 1000.0, 2000.0]),
        )


def test_add_user_param_array_goes_to_waveform(builder):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    builder.add_user_param("wave", data)
    assert any(w.waveformName == "wave" for w in builder.head.waveformInformations)
    assert builder.waveforms[-1].data.shape == data.shape


def test_calc_trajectory_returns_expected(monkeypatch, builder):
    RecordingSequence.created.clear()
    monkeypatch.setattr(pp, "Sequence", RecordingSequence)
    ev = SimpleNamespace(id=5)  # id should be stripped
    result = builder.calc_trajectory((ev,))
    assert result.sample_time_us == 1000
    assert result.number_of_samples == 4
    assert result.center_sample == 1
    assert result.trajectory_dimensions == 1
    assert result.traj.shape == (4, 1)

    seq = RecordingSequence.created[-1]
    assert all(not hasattr(e, "id") for block in seq.blocks for e in block)


def test_add_acquisition_sets_fields_and_labels(builder):
    traj = SimpleNamespace(
        sample_time_us=250,
        number_of_samples=3,
        center_sample=1,
        traj=np.arange(3, dtype=float).reshape(3, 1),
        trajectory_dimensions=1,
    )
    label_event = SimpleNamespace(type="labelset", label="k1", value=7)
    builder.add_acquisition(traj, (label_event,))
    acq = builder.acquisitions[-1]

    assert acq.scan_counter == 0
    assert acq.center_sample == 1
    assert acq.encoding_space_ref == builder.current_encoding
    assert acq.sample_time_us == 250
    assert acq.idx.kspace_encode_step_1 == 7
    np.testing.assert_array_equal(acq.traj[:, 0], [0.0, 1.0, 2.0])


def test_add_acquisition_applies_rotation(builder):
    base_traj = np.ones((2, 1))
    traj = SimpleNamespace(
        sample_time_us=10,
        number_of_samples=2,
        center_sample=0,
        traj=base_traj,
        trajectory_dimensions=1,
    )
    rot_event = SimpleNamespace(type="rot3D", rot_quaternion=FakeRotation(scale=3.0))
    label_event = SimpleNamespace(type="labelset", label="k0", value=0)
    builder.add_acquisition(traj, (label_event, rot_event))
    acq = builder.acquisitions[-1]
    np.testing.assert_array_equal(acq.traj[:, 0], base_traj[:, 0] * 3.0)


def test_mode_switch_passthrough_skips_execution():
    b = ISMRMRDBuilder(passthrough=True)
    assert b.calc_trajectory(SimpleNamespace()) is None