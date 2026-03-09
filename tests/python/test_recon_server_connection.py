"""Unit tests for recon server connection message parsing."""

from __future__ import annotations

import socket

from pulserver.recon_server import constants
from pulserver.recon_server.connection import Connection


class _FakeSocket:
    def __init__(self, payloads: list[bytes]):
        self._payloads = payloads[:]
        self.closed = False

    def recv(self, nbytes: int, _flags: int = 0) -> bytes:
        if not self._payloads:
            return b''
        data = self._payloads.pop(0)
        assert len(data) == nbytes
        return data

    def shutdown(self, _how):
        return None

    def close(self):
        self.closed = True


def _msg_id(value: int) -> bytes:
    return constants.MrdMessageIdentifier.pack(value)


def _msg_len(value: int) -> bytes:
    return constants.MrdMessageLength.pack(value)


def test_reads_config_text_message():
    text = 'simplefft\0'.encode('utf-8')
    sock = _FakeSocket([
        _msg_id(constants.MRD_MESSAGE_CONFIG_TEXT),
        _msg_len(len(text)),
        text,
    ])
    conn = Connection(sock, savedata=False)
    assert conn.next() == 'simplefft'


def test_reads_metadata_message():
    text = '<xml/>\0'.encode('utf-8')
    sock = _FakeSocket([
        _msg_id(constants.MRD_MESSAGE_METADATA_XML_TEXT),
        _msg_len(len(text)),
        text,
    ])
    conn = Connection(sock, savedata=False)
    assert conn.next() == '<xml/>'


def test_close_message_marks_exhausted():
    sock = _FakeSocket([_msg_id(constants.MRD_MESSAGE_CLOSE)])
    conn = Connection(sock, savedata=False)
    assert conn.next() is None
    assert conn.is_exhausted is True


def test_shutdown_close_closes_socket():
    sock = _FakeSocket([])
    conn = Connection(sock, savedata=False)
    conn.shutdown_close()
    assert sock.closed is True
