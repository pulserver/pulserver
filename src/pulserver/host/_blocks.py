"""The blocks of the host commands that carry scanner limits and a file to import."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def parse_import(block: str) -> Path:
    """Read an import block: the ``file`` line naming the first file of a chain.

    Raises
    ------
    ValueError
        If the block has no ``file`` line.
    """
    for line in block.splitlines():
        if line.startswith("file: "):
            return Path(line.removeprefix("file: ").strip())
    raise ValueError("IMPORT needs a file line")


def format_import(path: Path | str) -> str:
    return f"{IMPORT_BEGIN}\nfile: {path}\n{IMPORT_END}\n"
