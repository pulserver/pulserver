"""Comprehensive tests for write_mrd public API."""

import pathlib
import numpy as np
import pytest
import h5py
import ismrmrd
import ismrmrd.hdf5 as ismrmrd_hdf5

from pulserver.tools._mrd._mrd_file import write_mrd

# -----------------------------
# Fixtures / helpers
# -----------------------------

@pytest.fixture
def tmp_file(tmp_path):
    return tmp_path / "test.mrd"


@pytest.fixture
def simple_header():
    head = ismrmrd.xsd.ismrmrdHeader()
    head.measurementInformation = ismrmrd.xsd.measurementInformationType()
    head.measurementInformation.sequenceName = "test"
    return head


@pytest.fixture
def simple_acquisition():
    acq = ismrmrd.Acquisition()
    acq.version = 1
    acq.resize(trajectory_dimensions=1, number_of_samples=4)
    acq.center_sample = 1
    acq.sample_time_us = 10

    # minimal data: 1 channel, 4 samples (complex)
    acq.resize(active_channels=1, number_of_samples=4, trajectory_dimensions=1)
    acq.data[:] = np.array([[1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]], dtype=np.complex64)

    # trajectory
    acq.traj[:] = np.arange(4, dtype=np.float32).reshape(4, 1)
    return acq


@pytest.fixture
def simple_waveform():
    head = ismrmrd.WaveformHeader(
        waveform_id=1024, channels=2, number_of_samples=3
    )
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
    return ismrmrd.Waveform(head=head, data=data)


# -----------------------------
# Validation
# -----------------------------

def test_write_mrd_rejects_empty_acquisitions(tmp_file, simple_header):
    with pytest.raises(ValueError):
        write_mrd(tmp_file, "dataset", simple_header, acquisitions=[])


# -----------------------------
# Basic write behavior
# -----------------------------

def test_write_mrd_creates_file(tmp_file, simple_header, simple_acquisition):
    write_mrd(
        tmp_file,
        dataset_name="dataset",
        head=simple_header,
        acquisitions=[simple_acquisition],
    )
    assert tmp_file.exists()


# -----------------------------
# Acquisition dataset integrity
# -----------------------------

def test_write_mrd_writes_acquisition_data(tmp_file, simple_header, simple_acquisition):
    write_mrd(
        tmp_file,
        dataset_name="dataset",
        head=simple_header,
        acquisitions=[simple_acquisition],
    )

    with h5py.File(tmp_file, "r") as f:
        dset = f["dataset/data"]
        assert dset.shape == (1,)
        assert dset.attrs["next"] == 1

        rec = dset[0]
        # header exists
        assert rec["head"].dtype == ismrmrd_hdf5.acquisition_header_dtype

        # data flattened as float32
        data = rec["data"].view(np.float32)
        assert data.size == 8  # 2 * channels * samples

        # trajectory flattened
        traj = rec["traj"].view(np.float32)
        assert traj.size == 4


# -----------------------------
# Waveform dataset integrity
# -----------------------------

def test_write_mrd_writes_waveforms(tmp_file, simple_header, simple_acquisition, simple_waveform):
    write_mrd(
        tmp_file,
        dataset_name="dataset",
        head=simple_header,
        acquisitions=[simple_acquisition],
        waveforms=[simple_waveform],
    )

    with h5py.File(tmp_file, "r") as f:
        wset = f["dataset/waveforms"]
        assert wset.shape == (1,)
        assert wset.attrs["next"] == 1

        rec = wset[0]
        assert rec["head"].dtype == ismrmrd_hdf5.waveform_header_dtype
        data = rec["data"].view(np.uint32)
        assert data.size == 6


# -----------------------------
# Missing data / trajectory handling
# -----------------------------

def test_write_mrd_handles_missing_data_and_traj(tmp_file, simple_header):
    acq = ismrmrd.Acquisition()
    acq.version = 1
    acq.resize(trajectory_dimensions=0, number_of_samples=0)

    write_mrd(
        tmp_file,
        dataset_name="dataset",
        head=simple_header,
        acquisitions=[acq],
    )

    with h5py.File(tmp_file, "r") as f:
        rec = f["dataset/data"][0]
        assert rec["data"].size == 0
        assert rec["traj"].size == 0


# -----------------------------
# Overwrite behavior
# -----------------------------

def test_write_mrd_overwrite(tmp_file, simple_header, simple_acquisition):
    # first write
    write_mrd(tmp_file, "dataset", simple_header, [simple_acquisition])
    size1 = tmp_file.stat().st_size

    # second write without overwrite should keep file
    write_mrd(tmp_file, "dataset", simple_header, [simple_acquisition], overwrite=False)
    size2 = tmp_file.stat().st_size
    assert size2 == size1

    # overwrite=True replaces file
    write_mrd(tmp_file, "dataset", simple_header, [simple_acquisition], overwrite=True)
    size3 = tmp_file.stat().st_size
    assert size3 == size1


# -----------------------------
# Dataset naming
# -----------------------------

def test_write_mrd_custom_dataset_name(tmp_file, simple_header, simple_acquisition):
    write_mrd(
        tmp_file,
        dataset_name="custom",
        head=simple_header,
        acquisitions=[simple_acquisition],
    )

    with h5py.File(tmp_file, "r") as f:
        assert "custom" in f
        assert "data" in f["custom"]