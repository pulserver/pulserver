"""The blocks of the host commands that carry scanner limits and a file to import."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..protocol import FOV_OFFSET, FOV_ROTATION, prescribed_rotation

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
    path, offset, rotation = None, dict.fromkeys(FOV_OFFSET, 0.0), {}
    for line in block.splitlines():
        name, _, value = line.partition(": ")
        if name == "file":
            path = Path(value.strip())
        elif name in offset:
            offset[name] = float(value)
        elif name in FOV_ROTATION:
            rotation[name] = float(value)
    if path is None:
        raise ValueError("IMPORT needs a file line")
    x, y, z = offset.values()
    return path, (x, y, z), prescribed_rotation(rotation)


def format_import(
    path: Path | str,
    fov_offset_mm: tuple[float, float, float] | None = None,
    fov_rotation: np.ndarray | None = None,
) -> str:
    lines = [IMPORT_BEGIN, f"file: {path}"]
    if fov_offset_mm is not None:
        lines += [
            f"{name}: {value!r}"
            for name, value in zip(FOV_OFFSET, fov_offset_mm, strict=True)
        ]
    if fov_rotation is not None:
        matrix = np.asarray(fov_rotation, dtype=float).ravel()
        lines += [
            f"{name}: {float(value)!r}"
            for name, value in zip(FOV_ROTATION, matrix, strict=True)
        ]
    return "\n".join([*lines, IMPORT_END]) + "\n"
