"""Unit tests for server utility behavior and plugin dispatch wiring."""

from __future__ import annotations

import socket

from pulserver.recon_server.server import Server


class _FakeConnection:
    def __init__(self, _sock, savedata=False, savedata_folder=''):
        self._items = iter(['simplefft', '<xml/>'])
        self.is_exhausted = False

    def __next__(self):
        return next(self._items)

    def peek_message_identifier(self):
        return None

    def __iter__(self):
        return iter(())

    def shutdown_close(self):
        self.is_exhausted = True


class _DummySocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def shutdown(self, _how):
        return None

    def close(self):
        return None


def test_normalize_config_json():
    cfg = Server._normalize_config('{"plugin": "simplefft"}')
    assert cfg['plugin'] == 'simplefft'


def test_normalize_config_plain_text():
    cfg = Server._normalize_config('simplefft')
    assert cfg['plugin'] == 'simplefft'


def test_handle_uses_default_plugin(monkeypatch):
    from pulserver.recon_server import server as server_module

    monkeypatch.setattr(server_module, 'Connection', _FakeConnection)

    s = Server('127.0.0.1', 0, use_multiprocessing=False)

    dummy_sock = _DummySocket()
    s.handle(dummy_sock)

    # One close message should be written in finally block.
    assert len(dummy_sock.sent) == 1

    s.socket.close()
