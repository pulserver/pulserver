"""Shared pytest fixtures."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not _cuda_available(), reason="no CUDA device"),
            ],
        ),
    ]
)
def device(request):
    """Device name, ``"cpu"`` or ``"cuda"``; the CUDA leg is skipped without a device."""
    return request.param


def exam_header(exam):
    """A header naming an exam, as the proxies of a host read it."""
    return SimpleNamespace(
        studyInformation=SimpleNamespace(studyInstanceUID=exam, studyID=None),
        userParameters=None,
    )


def calibration_header(
    coils=4,
    matrix=16,
    *,
    slices=1,
    labels=True,
    measurement="7",
    calibration=None,
    partitions=1,
    repetitions=1,
    trajectory="cartesian",
):
    """A header of ``coils`` labelled coils and one encoding space of ``matrix`` by ``matrix`` voxels.

    The field of view is 220 mm by 220 mm with 5 mm slices or partitions;
    ``slices``, ``partitions`` and ``repetitions`` vary the counters of those
    names. ``labels=False`` leaves out the coil labels. A ``calibration``
    matrix adds encoding space 1 of that matrix, with the same field of view,
    slices, partitions and repetitions.
    """

    def encoding(size):
        voxels = SimpleNamespace(x=size, y=size, z=partitions)
        fov = SimpleNamespace(x=220.0, y=220.0, z=5.0 * partitions)
        limits = {
            "kspace_encoding_step_1": SimpleNamespace(
                minimum=0, maximum=size - 1, center=size // 2
            )
        }
        if partitions > 1:
            limits["kspace_encoding_step_2"] = SimpleNamespace(
                minimum=0, maximum=partitions - 1, center=partitions // 2
            )
        for name, extent in (("slice", slices), ("repetition", repetitions)):
            if extent > 1:
                limits[name] = SimpleNamespace(minimum=0, maximum=extent - 1, center=0)
        return SimpleNamespace(
            encodedSpace=SimpleNamespace(matrixSize=voxels, fieldOfView_mm=fov),
            reconSpace=SimpleNamespace(matrixSize=voxels, fieldOfView_mm=fov),
            encodingLimits=SimpleNamespace(**limits),
            trajectory=trajectory,
        )

    coil_labels = [
        SimpleNamespace(coilNumber=index, coilName=f"H{index}")
        for index in range(coils)
    ]
    return SimpleNamespace(
        encoding=[encoding(matrix)]
        + ([] if calibration is None else [encoding(calibration)]),
        acquisitionSystemInformation=SimpleNamespace(
            receiverChannels=coils, coilLabel=coil_labels if labels else None
        ),
        measurementInformation=SimpleNamespace(
            measurementID=measurement, frameOfReferenceUID="1.2.3"
        ),
    )


def coil_phantom(coils=6, matrix=32, seed=0):
    """A smooth object, the coils' sensitivities and the centred k-space of their product.

    Returns ``(image, maps, kspace)`` as ``complex64`` arrays of shape
    ``(matrix, matrix)``, ``(coils, matrix, matrix)`` and
    ``(coils, matrix, matrix)``. A sensitivity is a complex plane of the position,
    so the coil images span three dimensions of coil space: k-space is
    compressed to three virtual channels without loss.
    """
    generator = np.random.default_rng(seed)
    axis = (np.arange(matrix) - matrix // 2) / (matrix / 2)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    support = x**2 / 0.8 + y**2 / 0.9 < 1
    image = (support * np.exp(-((x - 0.15) ** 2 + (y + 0.1) ** 2) / 0.35)).astype(
        np.complex64
    )
    plane = generator.standard_normal((coils, 3)) + 1j * generator.standard_normal(
        (coils, 3)
    )
    maps = (
        plane[:, 0, None, None]
        + plane[:, 1, None, None] * x
        + plane[:, 2, None, None] * y
    ).astype(np.complex64)
    return image, maps, centred_kspace(maps * image)


def centred_kspace(images):
    """The unitary two-dimensional transform of ``(coils, y, x)`` images, with the centre at ``n // 2``."""
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(images, axes=(1, 2)), norm="ortho"), axes=(1, 2)
    )
    return kspace.astype(np.complex64)


def combine(kspace, maps):
    """The SENSE image of fully sampled ``(coils, y, x)`` k-space, from maps of unit root sum of squares.

    The conjugate maps times the coil images, summed over the coils: the image
    times the root sum of squares of the sensitivities the maps were
    normalised from.
    """
    images = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(1, 2)), norm="ortho"), axes=(1, 2)
    )
    return (maps.conj() * images).sum(axis=0)


def relative_difference(image, reference):
    """The norm of ``image - reference`` over the norm of ``reference``."""
    return np.linalg.norm(image - reference) / np.linalg.norm(reference)


def readouts(
    kspace,
    *flags,
    lines=None,
    last=(),
    slice_index=0,
    partition=0,
    repetition=0,
    space=0,
    dwell_us=None,
    geometry=None,
):
    """The acquisitions of the lines of ``kspace`` ``(coils, lines, samples)``, in order.

    Each carries ``flags``, the line as its ``kspace_encode_step_1``, ``slice_index``,
    ``partition``, ``repetition`` and the encoding space ``space``; the last also
    carries the flags ``last``. ``lines`` selects the lines, all of them by
    default. ``geometry`` maps acquisition attributes, such as ``position``, to
    the values each takes.
    """
    coils, _, samples = kspace.shape
    acquisitions = []
    selected = list(range(kspace.shape[1]) if lines is None else lines)
    for index, line in enumerate(selected):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(samples, coils)
        acquisition.data[:] = kspace[:, line]
        acquisition.center_sample = samples // 2
        acquisition.idx.kspace_encode_step_1 = line
        acquisition.idx.kspace_encode_step_2 = partition
        acquisition.idx.slice = slice_index
        acquisition.idx.repetition = repetition
        acquisition.encoding_space_ref = space
        for name, value in (geometry or {}).items():
            setattr(acquisition, name, value)
        if dwell_us is not None:
            acquisition.sample_time_us = dwell_us
        names = (*flags, *(last if index == len(selected) - 1 else ()))
        for name in names:
            acquisition.setFlag(getattr(ismrmrd, name))
        acquisitions.append(acquisition)
    return acquisitions


def acs_readouts(kspace, *, acs=16, slice_index=0, repetition=0):
    """The readouts of one slice, whose central ``acs`` lines are also parallel-imaging calibration.

    The slice closes with its last readout.
    """
    lines = kspace.shape[1]
    centre = range(lines // 2 - acs // 2, lines // 2 + acs // 2)
    outside = [line for line in range(lines) if line not in centre]
    return readouts(
        kspace, lines=outside, slice_index=slice_index, repetition=repetition
    ) + readouts(
        kspace,
        "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING",
        lines=centre,
        last=("ACQ_LAST_IN_SLICE",),
        slice_index=slice_index,
        repetition=repetition,
    )


def stream_plugin(step=None, *, gadgets=(), axes=()):
    """A plugin whose ``recon`` calls ``step(context, data)`` for each unit.

    What ``step`` returns, or the unit itself where there is no ``step``, is
    appended to the plugin's ``results``, which a stream's copy shares.
    A unit closes with its slice, and ``axes`` are the counters it is laid out
    along. Reconstructs nothing.
    """
    from pulserver import recon
    from pulserver.mrd import AcquisitionFlag

    class Step(recon.ReconPlugin):
        def __init__(self):
            super().__init__(
                gadgets=gadgets,
                triggers={"imaging": AcquisitionFlag.LAST_IN_SLICE},
                axes=axes,
            )
            self.results = []

        def recon(self, context, branch, data):
            self.results.append(data if step is None else step(context, data))

    return Step()


def play(plugin, header, acquisitions, *, exam=None, device=None):
    """Run ``acquisitions`` through one stream of ``plugin`` and return its context."""
    from pulserver import recon

    plugin = plugin.spawn()
    context = recon.ReconContext(
        header=header,
        exam=recon.ExamCache("exam") if exam is None else exam,
        device=device,
    )
    plugin.startup(context)
    for acquisition in acquisitions:
        plugin.receive(acquisition, context)
    plugin.flush(context)
    return context


#: The vendor the C library is compiled for where a test builds it as a scanner
#: does, and the Pulseq labels filling that vendor's three ADC label columns.
VENDOR = 5
LABELS = (8, 7, 6)

_ROOT = Path(__file__).parents[1]
_C_SOURCES = _ROOT / "src" / "c"


def _build(directory, program, name, *defines):
    """``program`` linked with the C library as a scanner builds it: 32-bit, for one vendor."""
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n")
    toolchain = subprocess.run(
        ["gcc", "-m32", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    if toolchain.returncode != 0:
        pytest.skip("no 32-bit C toolchain")
    folders = ("pulseq", "core", "io", "structure", "cache", "playout")
    sources = [
        str(p) for folder in folders for p in sorted((_C_SOURCES / folder).glob("*.c"))
    ]
    includes = [
        f"-I{_C_SOURCES / sub}"
        for sub in ("", "include", "include/pulseg", "include/pulseq", "pulseq")
    ]
    output = directory / name
    subprocess.run(
        [
            "gcc",
            "-m32",
            "-std=c89",
            f"-DPULSEG_VENDOR={VENDOR}",
            *defines,
            *includes,
            str(_ROOT / "tests" / "native" / program),
            *sources,
            "-lm",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.fixture(name="scanner_build", scope="session")
def scanner_build_fixture():
    """Build a native program against the C library as a scanner builds it.

    ``build(directory, program, name, *defines)``, where ``program`` is a file
    of ``tests/native/``. Skips the test where no 32-bit toolchain is present.
    """
    return _build
