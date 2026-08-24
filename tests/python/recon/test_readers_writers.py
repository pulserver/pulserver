"""Tests for private MRD reader/writer round trips."""

import io
import struct

import ismrmrd
import numpy as np
from pulserver.recon._server import constants
from pulserver.recon._server.readers import (
    read,
    read_acquisition,
    read_byte_string,
    read_image,
    read_text,
    read_vector,
    read_waveform,
)
from pulserver.recon._server.writers import (
    write_acquisition,
    write_byte_string,
    write_image,
    write_text,
    write_vector,
    write_waveform,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSocket:
    """In-memory read/write buffer that mimics Connection.SocketWrapper."""

    def __init__(self, data: bytes = b"") -> None:
        self._buf = io.BytesIO(data)

    def read(self, nbytes: int) -> bytes:
        return self._buf.read(nbytes)

    def write(self, data: bytes) -> None:
        self._buf.write(data)

    def reset(self) -> None:
        self._buf.seek(0)

    def getvalue(self) -> bytes:
        return self._buf.getvalue()


def _make_write_socket() -> FakeSocket:
    return FakeSocket()


def _make_read_socket(data: bytes) -> FakeSocket:
    return FakeSocket(data)


# ---------------------------------------------------------------------------
# Primitive read/write
# ---------------------------------------------------------------------------


def test_read_uint16():
    src = _make_read_socket(struct.pack("<H", 1008))
    assert read(src, constants.GadgetMessageIdentifier) == 1008


def test_read_uint32():
    src = _make_read_socket(struct.pack("<I", 42))
    assert read(src, constants.uint32) == 42


def test_write_vector_roundtrip():
    dst = _make_write_socket()
    values = [1, 2, 3, 4, 5]
    write_vector(dst, values, constants.uint64)
    src = _make_read_socket(dst.getvalue())
    result = read_vector(src, numpy_type=np.uint64)
    np.testing.assert_array_equal(result, values)


def test_byte_string_roundtrip():
    dst = _make_write_socket()
    payload = b"hello world"
    write_byte_string(dst, payload)
    src = _make_read_socket(dst.getvalue())
    assert read_byte_string(src) == payload


# ---------------------------------------------------------------------------
# Text roundtrip
# ---------------------------------------------------------------------------


def test_text_roundtrip():
    dst = _make_write_socket()
    write_text(dst, "test message")

    raw = dst.getvalue()
    # Skip the 2-byte message identifier that write_text prepends
    src = _make_read_socket(raw[constants.GadgetMessageIdentifier.size :])
    result = read_text(src)
    assert result == "test message"


# ---------------------------------------------------------------------------
# Acquisition roundtrip
# ---------------------------------------------------------------------------


def test_acquisition_roundtrip():
    acq = ismrmrd.Acquisition()
    acq.resize(256, 8)  # 256 samples, 8 channels
    acq.data[:] = np.random.default_rng().standard_normal((8, 256)).astype(np.complex64)
    acq.idx.kspace_encode_step_1 = 42

    dst = _make_write_socket()
    write_acquisition(dst, acq)

    raw = dst.getvalue()
    # Skip the 2-byte message identifier
    src = _make_read_socket(raw[constants.GadgetMessageIdentifier.size :])
    result = read_acquisition(src)

    assert result.idx.kspace_encode_step_1 == 42
    assert result.data.shape == (8, 256)
    np.testing.assert_array_almost_equal(result.data, acq.data)


# ---------------------------------------------------------------------------
# Image roundtrip
# ---------------------------------------------------------------------------


def test_image_roundtrip():
    data = np.random.default_rng().standard_normal((1, 1, 64, 64)).astype(np.float32)
    image = ismrmrd.Image.from_array(data, transpose=False)
    image.image_index = 7

    dst = _make_write_socket()
    write_image(dst, image)

    raw = dst.getvalue()
    src = _make_read_socket(raw[constants.GadgetMessageIdentifier.size :])
    result = read_image(src)

    assert result.image_index == 7
    np.testing.assert_array_almost_equal(result.data, data)


# ---------------------------------------------------------------------------
# Waveform roundtrip
# ---------------------------------------------------------------------------


def test_waveform_roundtrip():
    wav = ismrmrd.Waveform()
    wav.resize(channels=3, number_of_samples=100)
    wav.data[:] = np.arange(300, dtype=np.uint32).reshape(3, 100)

    dst = _make_write_socket()
    write_waveform(dst, wav)

    raw = dst.getvalue()
    src = _make_read_socket(raw[constants.GadgetMessageIdentifier.size :])
    result = read_waveform(src)

    assert result.data.shape == (3, 100)
    np.testing.assert_array_equal(result.data, wav.data)
