"""Test Ceq structure."""

import pytest

import numpy as np
import pypulseq as pp

from pulserver._core._ceq import (
    PulseqRF,
    PulseqGrad,
    PulseqADC,
    PulseqTrig,
    PulseqBlock,
    Ceq,
)

# Sample Test Data
sample_rf = pp.make_arbitrary_rf(
    signal=np.asarray([1.0, 2.0, 3.0]), flip_angle=1.0, no_signal_scaling=True
)
sample_grad_trap = pp.make_trapezoid("z", area=1e3)
sample_grad_arbitrary = pp.make_arbitrary_grad("z", waveform=np.array([0.0, 0.5, 1.0]))
sample_adc = pp.make_adc(num_samples=128, dwell=10e-6)
sample_trig = pp.make_digital_output_pulse(channel="osc0", delay=0.1, duration=1.0)


@pytest.fixture
def pulseq_rf():
    return PulseqRF.from_struct(sample_rf)


@pytest.fixture
def pulseq_grad_trap():
    return PulseqGrad.from_struct(sample_grad_trap)


@pytest.fixture
def pulseq_grad_arbitrary():
    return PulseqGrad.from_struct(sample_grad_arbitrary)


@pytest.fixture
def pulseq_adc():
    return PulseqADC.from_struct(sample_adc)


@pytest.fixture
def pulseq_trig():
    return PulseqTrig.from_struct(sample_trig)


@pytest.fixture
def pulseq_block(pulseq_rf, pulseq_grad_trap, pulseq_adc, pulseq_trig):
    return PulseqBlock(
        ID=1, rf=sample_rf, gx=sample_grad_trap, adc=sample_adc, trig=sample_trig
    )


# Test PulseqRF Initialization and to_bytes conversion
def test_pulseq_rf_init(pulseq_rf):
    assert isinstance(pulseq_rf, PulseqRF)
    assert pulseq_rf.wav.n_samples == 3


def test_pulseq_rf_to_bytes(pulseq_rf):
    bytes_data = pulseq_rf.to_bytes()
    assert isinstance(bytes_data, bytes)
    assert len(bytes_data) > 0


# Test PulseqGrad Initialization and to_bytes conversion (Trap)
def test_pulseq_grad_trap_init(pulseq_grad_trap):
    assert isinstance(pulseq_grad_trap, PulseqGrad)
    assert pulseq_grad_trap.type == 1


def test_pulseq_grad_trap_to_bytes(pulseq_grad_trap):
    bytes_data = pulseq_grad_trap.to_bytes()
    assert isinstance(bytes_data, bytes)
    assert len(bytes_data) > 0


# Test PulseqGrad Initialization and to_bytes conversion (Arbitrary)
def test_pulseq_grad_arbitrary_init(pulseq_grad_arbitrary):
    assert isinstance(pulseq_grad_arbitrary, PulseqGrad)
    assert pulseq_grad_arbitrary.type == 2


def test_pulseq_grad_arbitrary_to_bytes(pulseq_grad_arbitrary):
    bytes_data = pulseq_grad_arbitrary.to_bytes()
    assert isinstance(bytes_data, bytes)
    assert len(bytes_data) > 0


# Test PulseqADC Initialization and to_bytes conversion
def test_pulseq_adc_init(pulseq_adc):
    assert isinstance(pulseq_adc, PulseqADC)
    assert pulseq_adc.num_samples == 128


def test_pulseq_adc_to_bytes(pulseq_adc):
    bytes_data = pulseq_adc.to_bytes()
    assert isinstance(bytes_data, bytes)
    assert len(bytes_data) > 0


# Test PulseqTrig Initialization and to_bytes conversion
def test_pulseq_trig_init(pulseq_trig):
    assert isinstance(pulseq_trig, PulseqTrig)
    assert pulseq_trig.channel == 0  # Channel 'osc0'


def test_pulseq_trig_to_bytes(pulseq_trig):
    bytes_data = pulseq_trig.to_bytes()
    assert isinstance(bytes_data, bytes)
    assert len(bytes_data) > 0


# Test PulseqBlock Initialization and to_bytes conversion
def test_pulseq_block_init(pulseq_block):
    assert isinstance(pulseq_block, PulseqBlock)
    assert pulseq_block.ID == 1
    assert pulseq_block.rf is not None
    assert pulseq_block.gx is not None
    assert pulseq_block.adc is not None
    assert pulseq_block.trig is not None


def test_pulseq_block_to_bytes(pulseq_block):
    bytes_data = pulseq_block.to_bytes()
    assert isinstance(bytes_data, bytes)
    assert len(bytes_data) > 0


# Test Ceq structure initialization (partial mockup)
def test_ceq_init(pulseq_block):
    loop = [
        [
            1,
            1,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1e-3,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1,
        ]
    ]
    sections_edges = [0, 1]
    ceq = Ceq([pulseq_block], loop, sections_edges)

    assert isinstance(ceq, Ceq)
    assert ceq.n_parent_blocks == 1
    assert ceq.n_segments > 0
    assert ceq.max_b1 == 3.0
