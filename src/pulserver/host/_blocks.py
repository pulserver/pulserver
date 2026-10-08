"""The blocks of the host commands that carry scanner limits and a file to import."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..protocol import (
    FOV_OFFSET,
    FOV_ROTATION,
    format_prescription,
    parse_prescription,
    prescribed_rotation,
)

LIMITS_BEGIN = "[Limits]"
LIMITS_END = "[Limits End]"
IMPORT_BEGIN = "[Import]"
IMPORT_END = "[Import End]"


def _limit(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def parse_limits(block: str) -> dict[str, Any]:
    """Read a limits block: ``pypulseqpp.Opts`` keyword arguments, one per line."""
    limits = {}
    for line in block.splitlines():
        if ": " in line and LIMITS_BEGIN not in line:
            name, value = line.split(": ", 1)
            limits[name.strip()] = _limit(value.strip())
    return limits


def format_limits(limits: dict[str, Any]) -> str:
    lines = [f"{name}: {value}" for name, value in limits.items()]
    return "\n".join([LIMITS_BEGIN, *lines, LIMITS_END]) + "\n"


def parse_import(
    block: str,
) -> tuple[Path, tuple[float, float, float], np.ndarray]:
    """Read an import block: the first file of a chain and its prescription.

    The offset is in mm along the logical readout, phase and slice axes, and
    zero along an axis the block leaves out. The rotation is that of
    :func:`~pulserver.protocol.prescribed_rotation`, from the block's
    ``fov_rotation_ij`` lines.

    Raises
    ------
    ValueError
        If the block has no ``file`` line, a prescription line is not a
        number, or the rotation is not orthonormal.
    """
    path = None
    for line in block.splitlines():
        name, _, value = line.partition(": ")
        if name == "file":
            path = Path(value.strip())
    prescribed = parse_prescription(block)
    if path is None:
        raise ValueError("IMPORT needs a file line")
    x, y, z = (prescribed.get(key, 0.0) for key in FOV_OFFSET)
    return path, (x, y, z), prescribed_rotation(prescribed)


def import_averages(block: str) -> int:
    """Return how many times an import block plays the chain's main sequence: its ``nex`` line, 1 without one.

    Raises
    ------
    ValueError
        If the ``nex`` line is not a whole number from 1.
    """
    for line in block.splitlines():
        name, _, value = line.partition(": ")
        if name == "nex":
            count = float(value)
            if count != round(count) or count < 1:
                raise ValueError(
                    f"nex is a whole number of averages from 1, not {value.strip()}"
                )
            return int(count)
    return 1


def format_import(
    path: Path | str,
    fov_offset_mm: tuple[float, float, float] | None = None,
    fov_rotation: np.ndarray | None = None,
    averages: int = 1,
) -> str:
    prescribed = {}
    if fov_offset_mm is not None:
        prescribed.update(zip(FOV_OFFSET, fov_offset_mm, strict=True))
    if fov_rotation is not None:
        matrix = np.asarray(fov_rotation, dtype=float).ravel()
        prescribed.update(zip(FOV_ROTATION, matrix, strict=True))
    lines = [IMPORT_BEGIN, f"file: {path}", *format_prescription(prescribed)]
    if averages != 1:
        lines.append(f"nex: {averages}")
    return "\n".join([*lines, IMPORT_END]) + "\n"
