"""Minimal server built on top of Gadgetron python server flow."""

from __future__ import annotations

import json
import logging
import multiprocessing
import socket
from typing import Any

from . import constants
from .config_loader import load_config_text, resolve_plugin_spec
from .connection import Connection
from .plugin_loader import PluginLoadError, PluginRegistry


class Server:
    """Socket server dispatching each connection to a recon plugin."""

    def __init__(
        self,
        address: str,
        port: int,
        default_plugin: str = 'pulserver.recon_server.plugins.simplefft:SimpleFFTPlugin',
        savedata: bool = False,
        savedata_folder: str = '/tmp/share/saved_data',
        use_multiprocessing: bool = False,
        plugin_directory: str | None = None,
    ) -> None:
        self.default_plugin = default_plugin
        self.savedata = savedata
        self.savedata_folder = savedata_folder
        self.use_multiprocessing = use_multiprocessing
        self.plugin_registry = PluginRegistry()
        self.plugin_registry.register_spec('simplefft', default_plugin)
        self.plugin_registry.register_spec(
            'sigpy_reference',
            'pulserver.recon_server.plugins.sigpy_reference:SigPyReferencePlugin',
        )
        if plugin_directory:
            self.plugin_registry.discover_directory(plugin_directory)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((address, port))

    def serve(self) -> None:
        self.socket.listen(0)
        while True:
            sock, (remote_addr, remote_port) = self.socket.accept()
            logging.info('Accepting connection from: %s:%d', remote_addr, remote_port)
            if self.use_multiprocessing:
                proc = multiprocessing.Process(target=self.handle, args=[sock])
                proc.daemon = True
                proc.start()
            else:
                self.handle(sock)

    def handle(self, sock: socket.socket) -> None:
        connection = Connection(
            sock,
            savedata=self.savedata,
            savedata_folder=self.savedata_folder,
        )
        try:
            config_msg = next(connection)
            if config_msg is None and connection.is_exhausted:
                return

            metadata_xml = next(connection)
            if metadata_xml is None and connection.is_exhausted:
                return

            config_dict = self._normalize_config(config_msg)
            if connection.peek_message_identifier() == constants.MRD_MESSAGE_TEXT:
                extra_text = next(connection)
                extra = load_config_text(extra_text)
                if isinstance(extra, dict):
                    config_dict.update(extra)

            plugin_spec = resolve_plugin_spec(config_dict, fallback=self.default_plugin)
            plugin = self.plugin_registry.resolve(plugin_spec)
            plugin.validate_config(config_dict)
            plugin.process(connection, config_dict, metadata_xml)
        except PluginLoadError as exc:
            logging.error('Plugin resolution failed: %s', exc)
        except StopIteration:
            pass
        except Exception:
            logging.exception('Unhandled recon server exception')
        finally:
            try:
                sock.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_CLOSE))
            except Exception:
                pass
            connection.shutdown_close()

    @staticmethod
    def _normalize_config(config_msg: Any) -> dict[str, Any]:
        if isinstance(config_msg, dict):
            return config_msg
        if isinstance(config_msg, bytes):
            config_msg = config_msg.decode('utf-8')
        if isinstance(config_msg, str):
            # Fast path: JSON payload from client.
            try:
                parsed = json.loads(config_msg)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return load_config_text(config_msg)
        return {}
