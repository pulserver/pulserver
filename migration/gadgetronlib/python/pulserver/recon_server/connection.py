"""Socket connection wrapper for MRD message stream."""

from __future__ import annotations

import logging
import os
from datetime import datetime
import random
import socket
from typing import Any

from . import constants


class Connection:
    """Small compatibility wrapper for config/metadata/acquisition stream."""

    def __init__(
        self,
        sock: socket.socket,
        savedata: bool = False,
        savedata_file: str = '',
        savedata_folder: str = '/tmp/share/saved_data',
        savedata_group: str = 'dataset',
    ) -> None:
        self.socket = sock
        self.savedata = savedata
        self.savedata_file = savedata_file
        self.savedata_folder = savedata_folder
        self.savedata_group = savedata_group
        self.mrd_file_path: str | None = None
        self.dset: Any = None
        self.is_exhausted = False

        self.handlers = {
            constants.MRD_MESSAGE_CONFIG_FILE: self.read_config_file,
            constants.MRD_MESSAGE_CONFIG_TEXT: self.read_config_text,
            constants.MRD_MESSAGE_METADATA_XML_TEXT: self.read_metadata,
            constants.MRD_MESSAGE_CLOSE: self.read_close,
            constants.MRD_MESSAGE_TEXT: self.read_text,
            constants.MRD_MESSAGE_ISMRMRD_ACQUISITION: self.read_acquisition,
            constants.MRD_MESSAGE_ISMRMRD_WAVEFORM: self.read_waveform,
            constants.MRD_MESSAGE_ISMRMRD_IMAGE: self.read_image,
        }

    def __iter__(self):
        while not self.is_exhausted:
            value = self.next()
            if value is None and self.is_exhausted:
                return
            yield value

    def __next__(self):
        value = self.next()
        if value is None and self.is_exhausted:
            raise StopIteration
        return value

    def read(self, nbytes: int) -> bytes:
        return self.socket.recv(nbytes, socket.MSG_WAITALL)

    def peek(self, nbytes: int) -> bytes:
        return self.socket.recv(nbytes, socket.MSG_PEEK)

    def next(self):
        msg_id = self.read_message_identifier()
        if self.is_exhausted:
            return None
        handler = self.handlers.get(msg_id, self.unknown_message_identifier)
        return handler()

    def shutdown_close(self) -> None:
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        self.socket.close()

    def read_message_identifier(self):
        try:
            identifier_bytes = self.read(constants.SIZEOF_MRD_MESSAGE_IDENTIFIER)
        except ConnectionResetError:
            self.is_exhausted = True
            return None

        if not identifier_bytes:
            self.is_exhausted = True
            return None

        return constants.MrdMessageIdentifier.unpack(identifier_bytes)[0]

    def peek_message_identifier(self):
        try:
            identifier_bytes = self.peek(constants.SIZEOF_MRD_MESSAGE_IDENTIFIER)
        except ConnectionResetError:
            self.is_exhausted = True
            return None

        if not identifier_bytes:
            self.is_exhausted = True
            return None

        return constants.MrdMessageIdentifier.unpack(identifier_bytes)[0]

    def read_message_length(self) -> int:
        length_bytes = self.read(constants.SIZEOF_MRD_MESSAGE_LENGTH)
        return constants.MrdMessageLength.unpack(length_bytes)[0]

    def read_config_file(self) -> str:
        config_file_bytes = self.read(constants.SIZEOF_MRD_MESSAGE_CONFIGURATION_FILE)
        config_file = constants.MrdMessageConfigurationFile.unpack(config_file_bytes)[0]
        config_file_text = config_file.split(b'\x00', 1)[0].decode('utf-8')

        if config_file_text == 'savedataonly':
            self.savedata = True

        self._ensure_save_file()
        return config_file_text

    def read_config_text(self) -> str:
        length = self.read_message_length()
        config = self.read(length)
        config_text = config.split(b'\x00', 1)[0].decode('utf-8')
        self._ensure_save_file()
        return config_text

    def read_metadata(self) -> str:
        length = self.read_message_length()
        metadata = self.read(length)
        metadata_xml = metadata.split(b'\x00', 1)[0].decode('utf-8')
        self._ensure_save_file()
        if self.savedata and self.dset is not None:
            self.dset.write_xml_header(bytes(metadata_xml, 'utf-8'))
        return metadata_xml

    def read_text(self) -> str:
        length = self.read_message_length()
        text = self.read(length)
        return text.split(b'\x00', 1)[0].decode('utf-8')

    def read_close(self):
        self.is_exhausted = True
        if self.savedata and self.dset is not None:
            self.dset.close()
            self.dset = None
        return None

    def read_acquisition(self):
        try:
            import ismrmrd  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError('ismrmrd is required for acquisition decoding') from exc

        acq = ismrmrd.Acquisition.deserialize_from(self.read)
        self._ensure_save_file()
        if self.savedata and self.dset is not None:
            self.dset.append_acquisition(acq)
        return acq

    def read_image(self):
        try:
            import ismrmrd  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError('ismrmrd is required for image decoding') from exc

        image = ismrmrd.Image.deserialize_from(self.read)
        self._ensure_save_file()
        if self.savedata and self.dset is not None:
            self.dset.append_image(f'image_{image.image_series_index}', image)
        return image

    def read_waveform(self):
        try:
            import ismrmrd  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError('ismrmrd is required for waveform decoding') from exc

        waveform = ismrmrd.Waveform.deserialize_from(self.read)
        self._ensure_save_file()
        if self.savedata and self.dset is not None:
            self.dset.append_waveform(waveform)
        return waveform

    @staticmethod
    def unknown_message_identifier(*_args):
        logging.error('Received unknown message type')
        raise StopIteration

    def _ensure_save_file(self) -> None:
        if not self.savedata or self.dset is not None:
            return

        try:
            import ismrmrd  # type: ignore
        except ImportError:
            logging.warning('Savedata enabled but ismrmrd is unavailable; skipping persistence')
            self.savedata = False
            return

        os.makedirs(self.savedata_folder, exist_ok=True)
        if self.savedata_file:
            self.mrd_file_path = self.savedata_file
        else:
            stamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
            suffix = random.randint(0, 100)
            self.mrd_file_path = os.path.join(self.savedata_folder, f'MRD_input_{stamp}_{suffix}.h5')

        self.dset = ismrmrd.Dataset(self.mrd_file_path, self.savedata_group)
        self.dset._file.require_group(self.savedata_group)
