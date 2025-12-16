"""Comprehensive tests for ISMRMRDBuilder public API."""

from types import SimpleNamespace

import numpy as np
import pytest
import pypulseq as pp

import ismrmrd.xsd as xsd

from pulserver.tools import ISMRMRDBuilder


# -----------------------------
# Helpers / stubs
# -----------------------------

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
        # 1D trajectory with k-space center at index 1
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


# -----------------------------
# Encoding management
# -----------------------------

def test_add_encoding_increments_and_tracks_current(builder):
    n = len(builder.head.encoding)
    builder.add_encoding()
    assert len(builder.head.encoding) == n + 1
    assert builder.current_encoding == n


def test_set_encoding_switches(builder):
    builder.add_encoding()
    builder.set_encoding(0)
    assert builder.current_encoding == 0


# -----------------------------
# Basic geometry setters
# -----------------------------

def test_set_fov_and_matrix(builder):
    builder.set_fov(220, 210, 5)
    enc = builder.head.encoding[builder.current_encoding]
    assert enc.encodedSpace.fieldOfView_mm.x == 220
    assert enc.reconSpace.fieldOfView_mm.y == 210

    builder.set_matrix(128, 64, 1)
    assert enc.encodedSpace.matrixSize.x == 128
    assert enc.reconSpace.matrixSize.y == 64


def test_set_etl(builder):
    builder.set_etl(8)
    assert builder.head.encoding[builder.current_encoding].echoTrainLength == 8


# -----------------------------
# Sequence parameters
# -----------------------------

@pytest.mark.parametrize("setter,attr,value", [
    ("set_TR", "TR", [5.0, 10.0]),
    ("set_TE", "TE", [3.0]),
    ("set_TI", "TI", [900.0]),
    ("set_flipAngle_deg", "flipAngle_deg", [90.0, 120.0]),
    ("set_echo_spacing", "echo_spacing", [0.6]),
])
def test_sequence_parameter_setters(builder, setter, attr, value):
    fn = getattr(builder, setter)
    fn(value)
    seq = builder.head.sequenceParameters
    assert list(getattr(seq, attr)) == list(value)


def test_set_sequence_type_and_name(builder):
    builder.set_sequence_type("GRE")
    builder.set_name("my_seq")
    assert builder.head.sequenceParameters.sequence_type == "GRE"
    assert builder.head.measurementInformation.sequenceName == "my_seq"


def test_set_resonance_frequency(builder):
    builder.set_H1resonanceFrequency_Hz(42.58e6, 3.0)
    assert builder.head.experimentalConditions.H1resonanceFrequency_Hz > 0


# -----------------------------
# Encoding limits
# -----------------------------

def test_set_limits_defaults_center(builder):
    builder.set_limits("k1", maximum=5)
    limits = builder.head.encoding[builder.current_encoding].encodingLimits.kspace_encoding_step_1
    assert limits.minimum == 0
    assert limits.maximum == 5
    assert limits.center == 3


# -----------------------------
# Trajectory metadata
# -----------------------------

def test_set_trajectory_happy_path(builder):
    desc = {
        "fov": 220.0,
        "lines": 128,
        "name": "spiral",
    }
    builder.set_trajectory(xsd.trajectoryType.OTHER, desc=desc, comment="test")
    enc = builder.head.encoding[builder.current_encoding]

    assert enc.trajectory == xsd.trajectoryType.OTHER
    assert enc.comment == "test"
    assert enc.trajectoryDescription.userParameterDouble[0].name == "fov"
    assert enc.trajectoryDescription.userParameterLong[0].name == "lines"
    assert enc.trajectoryDescription.userParameterString[0].name == "name"


def test_set_trajectory_rejects_invalid(builder):
    with pytest.raises(ValueError):
        builder.set_trajectory(traj_type="bad")
    with pytest.raises(ValueError):
        builder.set_trajectory(traj_type=xsd.calibrationModeType.EXTERNAL)


# -----------------------------
# Parallel imaging / multiband
# -----------------------------

def test_parallel_imaging_happy_path(builder):
    builder.set_parallel_imaging_info(
        calibration_type=xsd.calibrationModeType.EMBEDDED,
        Ry=2,
        Rz=1,
    )
    pi = builder.head.encoding[builder.current_encoding].parallelImaging
    assert pi.calibrationMode == xsd.calibrationModeType.EMBEDDED
    assert pi.accelerationFactor.kspace_encoding_step_1 == 2


def test_parallel_imaging_invalid(builder):
    with pytest.raises(ValueError):
        builder.set_parallel_imaging_info(calibration_type="bad")


def test_multiband_happy_path(builder):
    builder.set_parallel_imaging_info(xsd.calibrationModeType.EXTERNAL)
    builder.set_multiband_info(
        calibration_type=xsd.multibandCalibrationType.FULL3_D,
        mb_factor=2,
        spacing=5.0,
        calibration_encoding=0,
    )
    mb = builder.head.encoding[builder.current_encoding].parallelImaging.multiband
    assert mb.calibration == xsd.multibandCalibrationType.FULL3_D
    assert len(mb.spacing) == 1


def test_multiband_invalid(builder):
    with pytest.raises(ValueError):
        builder.set_multiband_info("bad", 2, 5.0, 0)


# -----------------------------
# Diffusion
# -----------------------------

def test_set_diffusion_happy_path(builder):
    direction = np.array([[1.0, 0.0, 0.0]])
    builder.set_diffusion(
        channel=xsd.diffusionDimensionType.SEGMENT,
        scheme="bipolar",
        direction=direction,
        bvalue=1000.0,
    )
    seq = builder.head.sequenceParameters
    assert seq.diffusionScheme == "bipolar"
    assert len(seq.diffusion) == 1


def test_set_diffusion_mismatched_lengths(builder):
    direction = np.eye(3)
    with pytest.raises(ValueError):
        builder.set_diffusion(
            channel=xsd.diffusionDimensionType.SEGMENT,
            scheme="bipolar",
            direction=direction,
            bvalue=np.array([0.0, 1000.0]),
        )


# -----------------------------
# User parameters
# -----------------------------

def test_add_user_param_scalar_and_update(builder):
    builder.add_user_param("alpha", 1.0)
    builder.add_user_param("alpha", 2.0)
    params = builder.head.userParameters.userParameterDouble
    assert len(params) == 1
    assert params[0].value == 2.0


def test_add_user_param_array_goes_to_waveform(builder):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    builder.add_user_param("wave", data)
    assert builder.waveforms[-1].data.shape == data.shape


# -----------------------------
# Trajectory calculation
# -----------------------------

def test_calc_trajectory(monkeypatch, builder):
    RecordingSequence.created.clear()
    monkeypatch.setattr(pp, "Sequence", RecordingSequence)

    ev = SimpleNamespace(id=5)
    traj = builder.calc_trajectory((ev,))

    assert traj.sample_time_us == 1000
    assert traj.number_of_samples == 4
    assert traj.center_sample == 1
    assert traj.trajectory_dimensions == 1
    assert traj.traj.shape == (4, 1)

    seq = RecordingSequence.created[-1]
    assert all(not hasattr(e, "id") for block in seq.blocks for e in block)


# -----------------------------
# Acquisition creation
# -----------------------------

def test_add_acquisition_with_labels_and_rotation(builder):
    base_traj = np.ones((2, 1))
    traj = SimpleNamespace(
        sample_time_us=10,
        number_of_samples=2,
        center_sample=0,
        traj=base_traj,
        trajectory_dimensions=1,
    )
    events = (
        SimpleNamespace(type="labelset", label="k1", value=7),
        SimpleNamespace(type="labelinc", label="k1", value=1),
        SimpleNamespace(type="rot3D", rot_quaternion=FakeRotation(scale=3.0)),
    )

    builder.add_acquisition(traj, events)
    acq = builder.acquisitions[-1]

    assert acq.idx.kspace_encode_step_1 == 8
    np.testing.assert_array_equal(acq.traj[:, 0], base_traj[:, 0] * 3.0)


# -----------------------------
# Passthrough mode
# -----------------------------

def test_passthrough_mode_skips_execution():
    b = ISMRMRDBuilder(passthrough=True)
    assert b.calc_trajectory(SimpleNamespace()) is None
    assert b.add_encoding() is None