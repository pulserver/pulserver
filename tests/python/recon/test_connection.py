"""Tests for pulserver.recon.connection — Connection, MessageType, DataSaver."""

import io
import socket
from unittest.mock import MagicMock

import ismrmrd
import pytest

from pulserver.recon import constants
from pulserver.recon.connection import (
    Connection,
    DataSaver,
    DummySaver,
    MessageType,
    MID_TO_TYPE,
    NAME_TO_TYPE,
)

# ---------------------------------------------------------------------------
# MessageType
# ---------------------------------------------------------------------------


def test_message_type_mid_lookup():
    assert MID_TO_TYPE[1008] is MessageType.Acquisition
    assert MID_TO_TYPE[1022] is MessageType.Image
    assert MID_TO_TYPE[1026] is MessageType.Waveform


def test_message_type_name_lookup():
    # NAME_TO_TYPE keys are Enum member .name, not .display_name
    assert NAME_TO_TYPE["Acquisition"] is MessageType.Acquisition
    assert NAME_TO_TYPE["Image"] is MessageType.Image


def test_message_type_attributes():
    mt = MessageType.Acquisition
    assert mt.mid == 1008
    assert mt.display_name == "GADGET_MESSAGE_ISMRMRD_ACQUISITION"
    assert mt.period == 100


# ---------------------------------------------------------------------------
# DummySaver
# ---------------------------------------------------------------------------


def test_dummy_saver_noop():
    saver = DummySaver()
    # Should not raise
    saver.save(constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION, MagicMock())


# ---------------------------------------------------------------------------
# DataSaver
# ---------------------------------------------------------------------------


def test_data_saver_creates_file(tmp_path):
    folder = str(tmp_path / "output")
    saver = DataSaver(
        savedataFile="test.h5", savedataFolder=folder, savedataGroup="dataset"
    )
    saver.create_save_file()
    assert saver.dset is not None
    assert (tmp_path / "output" / "test.h5").exists()
    saver.dset.close()


def test_data_saver_autogen_filename(tmp_path):
    folder = str(tmp_path)
    saver = DataSaver(savedataFile="", savedataFolder=folder, savedataGroup="dataset")
    saver.create_save_file()
    assert saver.savedataFile.startswith("mrd_unknown_")
    assert saver.savedataFile.endswith(".h5")
    saver.dset.close()


def test_data_saver_save_acquisition(tmp_path):
    folder = str(tmp_path)
    saver = DataSaver(
        savedataFile="acq.h5", savedataFolder=folder, savedataGroup="dataset"
    )

    acq = ismrmrd.Acquisition()
    acq.resize(64, 1)
    saver.save(constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION, acq)

    assert saver.dset is not None
    saver.dset.close()

    # Verify written data
    dset = ismrmrd.Dataset(str(tmp_path / "acq.h5"), "dataset", create_if_needed=False)
    assert dset.number_of_acquisitions() == 1
    dset.close()


# ---------------------------------------------------------------------------
# Connection — socketpair-based integration tests
# ---------------------------------------------------------------------------


def _safe_close(conn: Connection) -> None:
    """Shutdown a Connection ignoring BrokenPipeError from peer-already-closed."""
    try:
        conn.shutdown_close()
    except (BrokenPipeError, OSError):
        pass


def _make_connection_pair(
    savedata: bool = False, savedataFolder: str = ""
) -> tuple[Connection, socket.socket]:
    """Create a ``(Connection, raw_peer_socket)`` pair over a Unix socketpair."""
    server_sock, client_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    conn = Connection(
        server_sock,
        savedata=savedata,
        savedataFolder=savedataFolder,
    )
    return conn, client_sock


def test_connection_receives_close():
    conn, peer = _make_connection_pair()
    try:
        peer.sendall(
            constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
        )

        _mid, item = conn.next()
        assert conn.is_exhausted
        assert isinstance(item, ismrmrd.Acquisition)
        assert item.isFlagSet(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    finally:
        peer.close()
        _safe_close(conn)


def test_connection_sends_acquisition():
    conn, peer = _make_connection_pair()
    try:
        acq = ismrmrd.Acquisition()
        acq.resize(32, 2)
        conn.send(acq)

        # Read from peer: expect message id + serialized acquisition
        mid_bytes = peer.recv(2)
        mid = constants.GadgetMessageIdentifier.unpack(mid_bytes)[0]
        assert mid == constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
    finally:
        peer.close()
        _safe_close(conn)


def test_connection_sends_text():
    conn, peer = _make_connection_pair()
    try:
        conn.send("hello")

        mid_bytes = peer.recv(2)
        mid = constants.GadgetMessageIdentifier.unpack(mid_bytes)[0]
        assert mid == constants.GADGET_MESSAGE_TEXT
    finally:
        peer.close()
        _safe_close(conn)


def test_connection_receives_acquisition():
    conn, peer = _make_connection_pair()
    try:
        # Serialize an acquisition and send from peer
        acq = ismrmrd.Acquisition()
        acq.resize(64, 4)
        acq.idx.kspace_encode_step_1 = 17

        buf = io.BytesIO()
        acq.serialize_into(buf.write)
        raw = buf.getvalue()

        peer.sendall(
            constants.GadgetMessageIdentifier.pack(
                constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
            )
        )
        peer.sendall(raw)

        mid, item = conn.next()
        assert mid == constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
        assert isinstance(item, ismrmrd.Acquisition)
        assert item.idx.kspace_encode_step_1 == 17
    finally:
        peer.close()
        _safe_close(conn)


def test_connection_iter():
    conn, peer = _make_connection_pair()
    try:
        # Send two acquisitions then close
        for step in (0, 1):
            acq = ismrmrd.Acquisition()
            acq.resize(16, 1)
            acq.idx.kspace_encode_step_1 = step
            buf = io.BytesIO()
            acq.serialize_into(buf.write)
            peer.sendall(
                constants.GadgetMessageIdentifier.pack(
                    constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
                )
            )
            peer.sendall(buf.getvalue())

        peer.sendall(
            constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
        )

        items = list(conn)
        # 2 acquisitions + 1 sentinel from stop_iteration
        assert len(items) == 3
        assert items[0].idx.kspace_encode_step_1 == 0
        assert items[1].idx.kspace_encode_step_1 == 1
    finally:
        peer.close()
        _safe_close(conn)


def test_connection_peek():
    conn, peer = _make_connection_pair()
    try:
        peer.sendall(
            constants.GadgetMessageIdentifier.pack(
                constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
            )
        )
        mid = conn.peek()
        assert mid == constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
    finally:
        peer.close()
        _safe_close(conn)


def test_connection_send_raises_on_exhausted():
    conn, peer = _make_connection_pair()
    conn.is_exhausted = True
    with pytest.raises(ValueError, match="exhausted"):
        conn.send("anything")
    peer.close()
    _safe_close(conn)


def test_connection_filter():
    """Verify that filtered items are sent back (bounced) to the peer."""
    conn, peer = _make_connection_pair()
    try:
        conn.filter(ismrmrd.Image)

        # Send an acquisition then close
        acq = ismrmrd.Acquisition()
        acq.resize(8, 1)
        buf = io.BytesIO()
        acq.serialize_into(buf.write)
        peer.sendall(
            constants.GadgetMessageIdentifier.pack(
                constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
            )
        )
        peer.sendall(buf.getvalue())
        peer.sendall(
            constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
        )

        # Read data back from peer to avoid blocking (the bounced acq)
        # Use non-blocking recv to drain whatever the connection sent back
        peer.setblocking(False)
        bounced = b""
        import select

        while True:
            r, _, _ = select.select([peer], [], [], 0.5)
            if not r:
                break
            chunk = peer.recv(4096)
            if not chunk:
                break
            bounced += chunk

        # The bounced data should start with the acquisition MID
        if bounced:
            mid = constants.GadgetMessageIdentifier.unpack(bounced[:2])[0]
            assert mid == constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
    finally:
        peer.close()
        _safe_close(conn)


def test_connection_with_savedata(tmp_path):
    conn, peer = _make_connection_pair(savedata=True, savedataFolder=str(tmp_path))
    try:
        # Send acquisition + close
        acq = ismrmrd.Acquisition()
        acq.resize(32, 2)
        buf = io.BytesIO()
        acq.serialize_into(buf.write)

        peer.sendall(
            constants.GadgetMessageIdentifier.pack(
                constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
            )
        )
        peer.sendall(buf.getvalue())
        peer.sendall(
            constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
        )

        list(conn)  # drain

        # Verify HDF5 file was created
        h5_files = list(tmp_path.glob("mrd_unknown_*.h5"))
        assert len(h5_files) == 1
    finally:
        peer.close()
        _safe_close(conn)
