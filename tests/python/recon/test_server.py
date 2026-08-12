"""Tests for private MRD reconstruction-app resolution."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

from pulserver import ReconApp
from pulserver.recon._mrd.server import Server, _NullApp


class _TestApp(ReconApp):
    def reconstruct(self, bucket, context):
        del bucket, context


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.PLUGIN = _TestApp()
    return module


def _server() -> Server:
    server = Server.__new__(Server)
    server.default_handler = "savedataonly"
    server.handler_dirs = []
    return server


def test_resolve_app_from_importable_module():
    module = _module("fake_recon")
    with patch.dict(sys.modules, {"fake_recon": module}):
        result = _server()._resolve_app("fake_recon")
    assert result is module.PLUGIN


def test_resolve_app_from_private_built_in_package():
    result = _server()._resolve_app("simplefft")
    assert isinstance(result, ReconApp)
    assert type(result).__name__ == "SimpleFftRecon"


def test_resolve_app_falls_back_to_default():
    module = _module("my_default")
    server = _server()
    server.default_handler = "my_default"
    with patch.dict(sys.modules, {"my_default": module}):
        result = server._resolve_app("missing_recon_module")
    assert result is module.PLUGIN


def test_null_config_uses_default_app():
    result = _server()._resolve_app("null")
    assert type(result).__name__ == "SaveDataOnly"


def test_resolve_app_from_handler_dir(tmp_path):
    plugin_file = tmp_path / "custom_recon.py"
    plugin_file.write_text(
        "from pulserver import ReconApp\n"
        "class CustomRecon(ReconApp):\n"
        "    def reconstruct(self, bucket, context):\n"
        "        return None\n"
        "PLUGIN = CustomRecon()\n"
    )
    server = _server()
    server.handler_dirs = [str(tmp_path)]

    result = server._resolve_app("custom_recon.py")

    assert isinstance(result, ReconApp)
    assert type(result).__name__ == "CustomRecon"


def test_file_without_plugin_is_skipped(tmp_path):
    plugin_file = tmp_path / "bad_recon.py"
    plugin_file.write_text("value = 42\n")
    server = _server()
    server.handler_dirs = [str(tmp_path)]

    result = server._resolve_app("bad_recon")

    assert type(result).__name__ == "SaveDataOnly"


def test_missing_app_and_default_use_null_app():
    server = _server()
    server.default_handler = "missing_default_recon"
    result = server._resolve_app("also_missing")
    assert isinstance(result, _NullApp)


def test_server_init_owns_exam_cache_and_binds_socket():
    server = Server(host="127.0.0.1", port=0)
    address = server._socket.getsockname()
    assert address[0] == "127.0.0.1"
    assert address[1] > 0
    assert server._exam_caches.current_exam_id is None
    server._exam_caches.close()
    server._socket.close()
