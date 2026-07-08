"""Serialization of Sequence to binary stream via temporary file.

On Linux the temp file is placed on the RAM-backed ``/dev/shm`` when
available, eliminating disk I/O entirely.
"""

__all__ = ['write_to_stream']

import os
import platform
import tempfile
from pathlib import Path

import pypulseq as pp


def _tmpdir() -> str | None:
    """Return a RAM-disk directory on Linux, or ``None`` elsewhere."""
    if platform.system() == 'Linux':
        shm = Path('/dev') / 'shm'
        if shm.is_dir() and os.access(shm, os.W_OK):
            return str(shm)
    return None  # tempfile will choose a platform default


def write_to_stream(
    seq: pp.Sequence,
    create_signature: bool = True,
    remove_duplicates: bool = False,
    check_timing: bool = False,
) -> bytes:
    """
    Serialize a pypulseq Sequence to bytes via a temporary file.

    Uses pypulseq's native ``seq.write()`` to a safe temporary file
    (backed by ``/dev/shm`` on Linux when available) and reads the
    result back as raw bytes.

    Parameters
    ----------
    seq : pp.Sequence
        Sequence object to serialize.
    create_signature : bool, default=True
        Whether the file should be signed (MD5).
    remove_duplicates : bool, default=False
        Remove duplicate events before writing.
    check_timing : bool, default=False
        Run timing checks before writing.

    Returns
    -------
    bytes
        Raw bytes of the ``.seq`` file.
    """
    _ = check_timing
    tmpdir = _tmpdir()

    # Write using pypulseq's native method (handles all sections)
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.seq',
        dir=tmpdir,
        delete=True,
    ) as tmp:
        seq.write(
            tmp.name,
            create_signature=create_signature,
            remove_duplicates=remove_duplicates,
        )
        # Read back as bytes
        with open(tmp.name, 'rb') as f:
            data = f.read()

    return data
