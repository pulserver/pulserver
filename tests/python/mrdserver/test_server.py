"""Tests for mrdserver.server — handler resolution logic."""

import sys
import types
from unittest.mock import MagicMock, patch


from mrdserver.server import Server, _NullHandler

# ---------------------------------------------------------------------------
# Handler resolution
# ---------------------------------------------------------------------------


def test_resolve_handler_from_importable_module():
    """A config that matches an importable module with process() is loaded."""
    fake_module = types.ModuleType("fake_handler")
    fake_module.process = lambda conn, cfg, meta: None

    server = Server.__new__(Server)
    server.default_handler = "savedataonly"
    server.handler_dirs = []

    with patch.dict(sys.modules, {"fake_handler": fake_module}):
        result = server._resolve_handler("fake_handler")
    assert result is fake_module


def test_resolve_handler_fallback_to_default():
    """Unknown config falls back to default_handler."""
    fake_default = types.ModuleType("my_default")
    fake_default.process = lambda conn, cfg, meta: None

    server = Server.__new__(Server)
    server.default_handler = "my_default"
    server.handler_dirs = []

    with patch.dict(sys.modules, {"my_default": fake_default}):
        result = server._resolve_handler("nonexistent_module_xyz")
    assert result is fake_default


def test_resolve_handler_null_config():
    """Config 'null' goes directly to default handler."""
    fake_default = types.ModuleType("savedataonly")
    fake_default.process = lambda conn, cfg, meta: None

    server = Server.__new__(Server)
    server.default_handler = "savedataonly"
    server.handler_dirs = []

    with patch.dict(sys.modules, {"savedataonly": fake_default}):
        result = server._resolve_handler("null")
    assert result is fake_default


def test_resolve_handler_from_handler_dir(tmp_path):
    """Resolves a handler from a .py file in handler_dirs."""
    handler_file = tmp_path / "custom_recon.py"
    handler_file.write_text("def process(connection, config, metadata):\n    pass\n")

    server = Server.__new__(Server)
    server.default_handler = "savedataonly"
    server.handler_dirs = [str(tmp_path)]

    result = server._resolve_handler("custom_recon")
    assert hasattr(result, "process")
    assert result.__name__ == "custom_recon"


def test_resolve_handler_file_without_process_skipped(tmp_path):
    """A .py file without process() is skipped."""
    handler_file = tmp_path / "bad_handler.py"
    handler_file.write_text("x = 42\n")

    fake_default = types.ModuleType("savedataonly")
    fake_default.process = lambda conn, cfg, meta: None

    server = Server.__new__(Server)
    server.default_handler = "savedataonly"
    server.handler_dirs = [str(tmp_path)]

    with patch.dict(sys.modules, {"savedataonly": fake_default}):
        result = server._resolve_handler("bad_handler")
    assert result is fake_default


def test_resolve_handler_ultimate_fallback():
    """When nothing resolves—even default—returns _NullHandler."""
    server = Server.__new__(Server)
    server.default_handler = "totally_missing_module"
    server.handler_dirs = []

    result = server._resolve_handler("also_missing")
    assert result is _NullHandler


# ---------------------------------------------------------------------------
# _NullHandler
# ---------------------------------------------------------------------------


def test_null_handler_has_process():
    assert hasattr(_NullHandler, "process")
    assert callable(_NullHandler.process)


def test_null_handler_drains_connection():
    mock_conn = MagicMock()
    mock_conn.__iter__ = MagicMock(return_value=iter([1, 2, 3]))
    mock_conn.socket = MagicMock()

    _NullHandler.process(mock_conn, "null", None)
    mock_conn.socket.write.assert_called_once()


# ---------------------------------------------------------------------------
# Server.__init__
# ---------------------------------------------------------------------------


def test_server_init_binds_socket():
    """Server.__init__ should bind to a port without error."""
    server = Server(host="127.0.0.1", port=0)  # port=0 → OS picks a free port
    # Check that the socket is bound
    addr = server._socket.getsockname()
    assert addr[0] == "127.0.0.1"
    assert addr[1] > 0
    server._socket.close()
