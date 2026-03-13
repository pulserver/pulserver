"""I/O helpers for pulserver sequence writing."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO


def _write_to_temp_and_collect(
    seq,
    *,
    create_signature: bool,
    remove_duplicates: bool,
    check_timing: bool,
    v141_compat: bool,
) -> tuple[bytes, str | None]:
    with NamedTemporaryFile(suffix=".seq", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        signature = seq.write(
            tmp_path,
            create_signature=create_signature,
            remove_duplicates=remove_duplicates,
            check_timing=check_timing,
            v141_compat=v141_compat,
        )
        payload = Path(tmp_path).read_bytes()
        return payload, signature
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def write(
    seq,
    output: str | os.PathLike | BinaryIO | None = None,
    *,
    create_signature: bool = True,
    remove_duplicates: bool = True,
    check_timing: bool = True,
    v141_compat: bool = False,
):
    """Write a Pulseq sequence to path or return binary payload.

    When ``output`` is ``None``, bytes are returned instead of writing to disk.
    When ``output`` is a binary file-like object, bytes are written to it.
    """
    if output is None:
        payload, _signature = _write_to_temp_and_collect(
            seq,
            create_signature=create_signature,
            remove_duplicates=remove_duplicates,
            check_timing=check_timing,
            v141_compat=v141_compat,
        )
        return payload

    if hasattr(output, "write"):
        payload, signature = _write_to_temp_and_collect(
            seq,
            create_signature=create_signature,
            remove_duplicates=remove_duplicates,
            check_timing=check_timing,
            v141_compat=v141_compat,
        )
        output.write(payload)
        return signature

    return seq.write(
        str(output),
        create_signature=create_signature,
        remove_duplicates=remove_duplicates,
        check_timing=check_timing,
        v141_compat=v141_compat,
    )
