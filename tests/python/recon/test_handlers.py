"""Tests for pulserver.recon.handlers — simplefft and fftrecon reconstruction."""


import ismrmrd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers: fake connection that yields pre-built acquisitions
# ---------------------------------------------------------------------------


class FakeConnection:
    """Minimal Connection stand-in for handler testing.

    Yields the supplied acquisitions, then acts like the connection is closed.
    Captures any sent items in ``self.sent``.
    """

    def __init__(self, acquisitions: list[ismrmrd.Acquisition]) -> None:
        self._items = list(acquisitions)
        self.sent: list = []
        self.socket = self  # handlers call connection.socket.write for close

    def __iter__(self):
        yield from self._items

    def write(self, data: bytes) -> None:
        pass  # swallow close message


def _make_acquisitions(
    n_pe: int = 16,
    n_ro: int = 32,
    n_channels: int = 2,
    n_slices: int = 1,
) -> list[ismrmrd.Acquisition]:
    """Create a list of fake acquisitions for testing."""
    acqs = []
    for slc in range(n_slices):
        for pe in range(n_pe):
            acq = ismrmrd.Acquisition()
            acq.resize(n_ro, n_channels)
            acq.data[:] = np.random.randn(n_channels, n_ro).astype(np.complex64)
            acq.idx.kspace_encode_step_1 = pe
            acq.idx.slice = slc

            # Mark last line in slice
            if pe == n_pe - 1:
                acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)

            acqs.append(acq)

    # Mark very last acquisition as last in measurement
    acqs[-1].setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    return acqs


def _make_metadata(
    n_ro: int = 32,
    n_pe: int = 16,
    fov_x: float = 256.0,
    fov_y: float = 256.0,
    fov_z: float = 5.0,
):
    """Create a minimal ISMRMRD metadata header (as xsd object)."""
    hdr = ismrmrd.xsd.ismrmrdHeader()
    enc = ismrmrd.xsd.encodingType()
    enc.trajectory = ismrmrd.xsd.trajectoryType("cartesian")

    es = ismrmrd.xsd.encodingSpaceType()
    es.matrixSize = ismrmrd.xsd.matrixSizeType(x=n_ro, y=n_pe, z=1)
    es.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm(x=fov_x, y=fov_y, z=fov_z)
    enc.encodedSpace = es

    rs = ismrmrd.xsd.encodingSpaceType()
    rs.matrixSize = ismrmrd.xsd.matrixSizeType(x=n_ro, y=n_pe, z=1)
    rs.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm(x=fov_x, y=fov_y, z=fov_z)
    enc.reconSpace = rs

    lim = ismrmrd.xsd.encodingLimitsType()
    lim.kspace_encoding_step_1 = ismrmrd.xsd.limitType(
        minimum=0,
        maximum=n_pe - 1,
        center=n_pe // 2,
    )
    enc.encodingLimits = lim

    hdr.encoding.append(enc)
    return hdr


# ---------------------------------------------------------------------------
# simplefft handler
# ---------------------------------------------------------------------------


def test_simplefft_produces_image():
    from pulserver.recon.handlers.simplefft import _reconstruct

    n_pe, n_ro, n_ch = 16, 32, 2
    acqs = _make_acquisitions(n_pe=n_pe, n_ro=n_ro, n_channels=n_ch, n_slices=1)
    metadata = _make_metadata(n_ro=n_ro, n_pe=n_pe)

    image = _reconstruct(acqs, metadata)
    assert image is not None
    assert isinstance(image, ismrmrd.Image)
    assert image.data.size > 0


def test_simplefft_empty_group():
    from pulserver.recon.handlers.simplefft import _reconstruct

    metadata = _make_metadata()
    assert _reconstruct([], metadata) is None


def test_simplefft_output_dtype():
    from pulserver.recon.handlers.simplefft import _reconstruct

    acqs = _make_acquisitions(n_pe=8, n_ro=16, n_channels=1)
    metadata = _make_metadata(n_ro=16, n_pe=8)
    image = _reconstruct(acqs, metadata)
    # Image data should be int16
    assert image.data.dtype == np.int16 or image.data.dtype == np.float32


# ---------------------------------------------------------------------------
# fftrecon handler
# ---------------------------------------------------------------------------


def test_fftrecon_multi_slice():
    from pulserver.recon.handlers.fftrecon import _reconstruct

    n_pe, n_ro, n_ch, n_slc = 8, 16, 2, 3
    acqs = _make_acquisitions(n_pe=n_pe, n_ro=n_ro, n_channels=n_ch, n_slices=n_slc)
    metadata = _make_metadata(n_ro=n_ro, n_pe=n_pe)

    result = _reconstruct(acqs, metadata)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == n_slc
    assert result.dtype == np.int16


def test_fftrecon_single_slice():
    from pulserver.recon.handlers.fftrecon import _reconstruct

    n_pe, n_ro = 8, 16
    acqs = _make_acquisitions(n_pe=n_pe, n_ro=n_ro, n_channels=1, n_slices=1)
    metadata = _make_metadata(n_ro=n_ro, n_pe=n_pe)

    result = _reconstruct(acqs, metadata)
    assert result.shape == (1, n_ro, n_pe)


def test_fftrecon_array2image():
    from pulserver.recon.handlers.fftrecon import _array2image

    n_pe, n_ro = 8, 16
    data = np.random.randint(0, 100, (n_ro, n_pe), dtype=np.int16)
    acqs = _make_acquisitions(n_pe=n_pe, n_ro=n_ro, n_channels=1)
    metadata = _make_metadata(n_ro=n_ro, n_pe=n_pe)

    image = _array2image(data, acqs, metadata)
    assert isinstance(image, ismrmrd.Image)
    assert image.attribute_string  # meta should be non-empty


# ---------------------------------------------------------------------------
# savedataonly handler
# ---------------------------------------------------------------------------


def test_savedataonly_drains():
    from pulserver.recon.handlers.savedataonly import process

    acqs = _make_acquisitions(n_pe=4, n_ro=8, n_channels=1)
    conn = FakeConnection(acqs)
    metadata = _make_metadata(n_ro=8, n_pe=4)

    # Should not raise
    process(conn, {}, metadata)
