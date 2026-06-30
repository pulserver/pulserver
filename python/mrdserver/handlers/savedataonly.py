"""Save-data-only handler: drains all acquisitions without reconstruction."""

import logging
from typing import Any

from .. import constants


def process(connection: Any, config: Any, metadata: Any) -> None:
    """Drain all incoming data without reconstruction.

    Parameters
    ----------
    connection : Connection
        Active MRD connection.
    config : Any
        Unused configuration payload.
    metadata : Any
        Unused ISMRMRD XML header.
    """
    logging.info("savedataonly handler — draining all incoming data")
    try:
        for _msg in connection:
            pass
    finally:
        end = constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
        connection.socket.write(end)
