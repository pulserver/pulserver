"""Reconstructing a file, through the same hooks a scan goes through.

An MRD file holds what the scanner sent, so calling a plugin module on one has
to give what the inline stream would have given. That equality is the whole
claim -- there is no second reconstruction path -- and it is what is checked
here, rather than image quality, which the per-plugin tests already cover.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import ismrmrd
import ismrmrd.xsd

from test_cartesian2D_recon import COILS, N, acquisitions

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def kspace():
    """One slice of a disc phantom, fully sampled."""
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = 0.3 + np.where(x**2 + y**2 < 0.81, 0.7, 0.0)
    maps = (np.ones((COILS, N, N), np.complex64) / np.sqrt(COILS)) * image
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(maps, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64)


def header():
    """The header a scanner writes ahead of the data."""
    space = ismrmrd.xsd.encodingSpaceType(
        matrixSize=ismrmrd.xsd.matrixSizeType(x=N, y=N, z=1),
        fieldOfView_mm=ismrmrd.xsd.fieldOfViewMm(x=256.0, y=256.0, z=5.0),
    )
    return ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=63_750_000
        ),
        encoding=[
            ismrmrd.xsd.encodingType(
                encodedSpace=space,
                reconSpace=space,
                trajectory=ismrmrd.xsd.trajectoryType("cartesian"),
                encodingLimits=ismrmrd.xsd.encodingLimitsType(
                    kspace_encoding_step_1=ismrmrd.xsd.limitType(
                        minimum=0, maximum=N - 1, center=N // 2
                    )
                ),
            )
        ],
        acquisitionSystemInformation=ismrmrd.xsd.acquisitionSystemInformationType(
            receiverChannels=COILS
        ),
    )


def write_file(path, kspace, lines=None):
    """One scan, as an ISMRMRD file."""
    dataset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=True)
    dataset.write_xml_header(header().toXML())
    for acquisition in acquisitions(kspace, list(range(N)) if lines is None else lines):
        dataset.append_acquisition(acquisition)
    dataset.close()
    return str(path)


def pixels(output):
    """The image an emitted output carries, whichever form it took."""
    if hasattr(output, "dset"):
        return np.asarray(output.dset.pixel_array, dtype=float)
    return np.asarray(output.data, dtype=float).squeeze()


def phantom():
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    return 0.3 + np.where(x**2 + y**2 < 0.81, 0.7, 0.0)


def real_values(output):
    """The image an emitted DICOM carries, in the units it was reconstructed in."""
    stored = np.asarray(output.dset.pixel_array, dtype=float)
    slope = float(output.dset.get("RescaleSlope", 1.0) or 1.0)
    intercept = float(output.dset.get("RescaleIntercept", 0.0) or 0.0)
    return stored * slope + intercept


def test_calling_the_module_reconstructs_the_file(tmp_path, kspace):
    """``pulserver.app.cartesian2D_recon(path)`` -- the module is the entry point,
    exactly as a sequence module is."""
    from pulserver.app import cartesian2D_recon

    outputs = cartesian2D_recon(write_file(tmp_path / "scan.h5", kspace))

    assert len(outputs) == 1
    image = real_values(outputs[0]).T
    reference = phantom()
    error = np.linalg.norm(
        image / image.max() - reference / reference.max()
    ) / np.linalg.norm(reference / reference.max())
    assert image.shape == (N, N)
    assert error < 1e-3


def test_the_file_reaches_the_hooks_exactly_as_the_stream_does(tmp_path, kspace):
    """The claim the offline entry point rests on: a file holds what the
    scanner sent, so the plugin sees the same acquisitions in the same order
    and there is no second reconstruction path to keep in step."""
    from pulserver.app import cartesian2D_recon

    seen: list[tuple[int, int]] = []

    class Recording(type(cartesian2D_recon.PLUGIN)):
        def receive(self, acquisition, context):
            super().receive(acquisition, context)
            seen.append(
                (
                    int(acquisition.idx.kspace_encode_step_1),
                    int(acquisition.idx.segment),
                )
            )

    Recording().run(write_file(tmp_path / "scan.h5", kspace))

    expected = [
        (int(a.idx.kspace_encode_step_1), int(a.idx.segment))
        for a in acquisitions(kspace, list(range(N)))
    ]
    assert seen == expected


def test_every_recon_plugin_module_is_callable():
    """No plugin implements the offline entry point and none overrides it, so
    every module that carries a ``PLUGIN`` has it."""
    import pulserver.app as app
    import pulserver.app.recon as family

    called = 0
    for name in family.__all__:
        module = getattr(app, name)
        if hasattr(module, "PLUGIN"):
            assert callable(module), name
            called += 1
    assert called == 7


def test_a_file_with_no_header_says_so(tmp_path):
    """A reconstruction cannot be laid out without one."""
    from pulserver.app import cartesian2D_recon

    path = tmp_path / "headerless.h5"
    dataset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=True)
    dataset.write_xml_header("")
    dataset.close()

    with pytest.raises(ValueError, match="no MRD XML header"):
        cartesian2D_recon(str(path))


def test_a_missing_file_is_not_created(tmp_path):
    """Reading a scan is not a reason to make one."""
    from pulserver.app import cartesian2D_recon

    missing = tmp_path / "nothing.h5"
    with pytest.raises((OSError, RuntimeError)):
        cartesian2D_recon(str(missing))
    assert not missing.exists()


def test_the_exam_cache_defaults_to_the_file(tmp_path, kspace):
    """Two reconstructions of two files must not share artifacts keyed only by
    exam."""
    from pulserver.app import cartesian2D_recon

    seen: list[Any] = []

    class Recording(type(cartesian2D_recon.PLUGIN)):
        def startup(self, context):
            seen.append(context.exam.exam_id)
            super().startup(context)

    path = write_file(tmp_path / "scan.h5", kspace)
    Recording().run(path)
    assert seen == [path]


# %% what a boundary meant


def test_a_bucket_reports_only_the_boundaries_it_ended_on():
    """An acquisition's other flags say what it *is*, not what it closed."""
    from pulserver import AcquisitionBucket, AcquisitionFlag

    acquisition = ismrmrd.Acquisition()
    acquisition.resize(4, 2)
    for flag in (
        ismrmrd.ACQ_LAST_IN_SEGMENT,
        ismrmrd.ACQ_IS_PARALLEL_CALIBRATION,
        ismrmrd.ACQ_IS_REVERSE,
    ):
        acquisition.setFlag(flag)

    trigger = AcquisitionBucket(data=(acquisition,)).trigger
    assert trigger is AcquisitionFlag.LAST_IN_SEGMENT


def test_ending_several_units_at_once_is_not_ending_one():
    """The final acquisition of a scan closes its segment, its slice and the
    measurement, which is what makes ``is`` the right test for a calibration
    boundary and ``in`` the right test for the end of the scan."""
    from pulserver import AcquisitionBucket, AcquisitionFlag

    acquisition = ismrmrd.Acquisition()
    acquisition.resize(4, 2)
    for flag in (
        ismrmrd.ACQ_LAST_IN_SEGMENT,
        ismrmrd.ACQ_LAST_IN_SLICE,
        ismrmrd.ACQ_LAST_IN_MEASUREMENT,
    ):
        acquisition.setFlag(flag)

    trigger = AcquisitionBucket(data=(acquisition,)).trigger
    assert trigger is not AcquisitionFlag.LAST_IN_SEGMENT
    assert AcquisitionFlag.LAST_IN_SEGMENT in trigger
    assert AcquisitionFlag.LAST_IN_MEASUREMENT in trigger


def test_split_on_takes_flags_by_name_or_by_enum():
    """Both spell the same bit, so a plugin is free to say either."""
    from pulserver import AcquisitionFlag, ReconPlugin

    class Sink(ReconPlugin):
        def recon(self, bucket, context):
            del bucket, context

    by_enum = Sink(
        split_on=AcquisitionFlag.LAST_IN_SLICE | AcquisitionFlag.LAST_IN_SEGMENT
    )
    by_name = Sink(split_on=("ACQ_LAST_IN_SEGMENT", "ACQ_LAST_IN_SLICE"))
    assert sorted(by_enum.split_on) == sorted(by_name.split_on)
