"""Write an ISMRMRD file in one bulk write from in-RAM acquisitions and waveforms,"""

__all__ = ["write_mrd"]

import os
import pathlib

import h5py
import numpy as np

import ismrmrd
import ismrmrd.hdf5 as ismrmrd_hdf5


def write_mrd(
    filepath: str | pathlib.Path,
    dataset_name: str,
    head: ismrmrd.xsd.ismrmrdHeader,
    acquisitions: list[ismrmrd.Acquisition],
    waveforms: list[ismrmrd.Waveform] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Bulk write of MRD object

    Parameters
    ----------
    filepath : str
        Path to disk position where to store the file.
    dataset_name : str
        Path within HDF5 file where to store the dataset.
    head : ismrmrd.xsd.ismrmrdHeader
        ismrmrdHeader object.
    acquisitions : list[ismrmrd.Acquisition]
        List of ismrmrd Acquisitions.
    waveforms : list[ismrmrd.Waveform] | None, optional
        Optional list of ismrmrd Waveforms. The default is None.
    overwrite : bool, optional
        If True, overwrite the file if it is exist. The default is False.

    """
    filepath = pathlib.Path(filepath)
    os.makedirs(filepath.parent, exist_ok=True)

    if os.path.exists(filepath) and overwrite:
        os.remove(filepath)

    acq_list = list(acquisitions)
    if len(acq_list) == 0:
        raise ValueError("acquisitions must contain at least one Acquisition")

    wav_list = list(waveforms) if waveforms is not None else []
    n_acq = len(acq_list)
    n_wav = len(wav_list)

    # Build structured numpy arrays using ISMRMRD dtypes
    acq_dtype = ismrmrd_hdf5.acquisition_dtype
    acq_arr = np.empty((n_acq,), dtype=acq_dtype)
    for i, acq in enumerate(acq_list):
        acq_arr[i]["head"] = np.frombuffer(
            acq.getHead(), dtype=ismrmrd_hdf5.acquisition_header_dtype
        )
        if getattr(acq, "data", None) is not None and acq.data.size:
            acq_arr[i]["data"] = acq.data.view(np.float32).reshape(
                (2 * acq.active_channels * acq.number_of_samples,)
            )
        else:
            acq_arr[i]["data"] = np.asarray([], dtype=np.float32)
        if getattr(acq, "traj", None) is not None and acq.traj.size:
            acq_arr[i]["traj"] = acq.traj.view(np.float32).reshape(
                (acq.number_of_samples * acq.trajectory_dimensions,)
            )
        else:
            acq_arr[i]["traj"] = np.asarray([], dtype=np.float32)

    wav_arr = None
    if n_wav > 0:
        wav_dtype = ismrmrd_hdf5.waveform_dtype
        wav_arr = np.empty((n_wav,), dtype=wav_dtype)
        for i, wav in enumerate(wav_list):
            wav_arr[i]["head"] = np.frombuffer(
                wav.getHead(), dtype=ismrmrd_hdf5.waveform_header_dtype
            )
            if getattr(wav, "data", None) is not None and wav.data.size:
                wav_arr[i]["data"] = wav.data.view(np.uint32).reshape(
                    (wav.channels * wav.number_of_samples,)
                )
            else:
                wav_arr[i]["data"] = np.asarray([], dtype=np.uint32)

    # Write header
    with ismrmrd.Dataset(filepath) as file:
        file.write_xml_header(head.toXML("utf-8"))

    # Write everything in one shot
    with h5py.File(filepath, "w") as file:
        # Acquisitions
        grp = file.require_group(dataset_name)
        dset = grp.create_dataset(
            "data", shape=(n_acq,), maxshape=(None,), dtype=acq_dtype
        )
        dset[:] = acq_arr
        dset.attrs["next"] = n_acq

        # Waveforms
        if wav_arr is not None:
            wset = grp.create_dataset(
                "waveforms", shape=(n_wav,), maxshape=(None,), dtype=wav_dtype
            )
            wset[:] = wav_arr
            wset.attrs["next"] = n_wav
