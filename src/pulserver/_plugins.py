"""Plugin files, found by name along an ordered list of directories."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

#: Directories searched for ``<name>.py``, in order; a single directory is a
#: search path of one.
PluginPath = Path | str | Sequence[Path | str]

NAME = re.compile(r"[A-Za-z0-9_\-]+")

#: The shipped scanner sequences and reconstructions, searched after the
#: directories a caller gives, so a file of the same name there replaces one.
SEQUENCES = Path(__file__).parent / "_zoo" / "sequences"
RECONSTRUCTIONS = Path(__file__).parent / "_zoo" / "recon"


def directories(plugins: PluginPath) -> tuple[Path, ...]:
    """Return the directories of a search path, in search order."""
    if isinstance(plugins, (str, Path)):
        return (Path(plugins),)
    return tuple(Path(directory) for directory in plugins)


def find(plugins: PluginPath, name: str) -> Path:
    """Return ``<directory>/<name>.py`` of the first directory holding it, the shipped plugins last.

    A symbolic link is followed, so a link names the file it points to.

    Raises
    ------
    ValueError
        If ``name`` is not a plugin name.
    FileNotFoundError
        If no directory holds it.
    """
    if not NAME.fullmatch(name):
        raise ValueError(f"invalid plugin name {name!r}")
    searched = (*directories(plugins), SEQUENCES, RECONSTRUCTIONS)
    for directory in searched:
        path = directory / f"{name}.py"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no plugin {name!r} in {', '.join(str(d) for d in searched)}"
    )


def names(plugins: PluginPath) -> list[str]:
    """Return the name of each scanner sequence :func:`find` can return, once, sorted.

    The shipped sequences are listed; the shipped reconstructions are not. A
    directory that does not exist holds none.
    """
    return sorted(
        {
            path.stem
            for directory in (*directories(plugins), SEQUENCES)
            for path in directory.glob("*.py")
            if NAME.fullmatch(path.stem) and path.is_file()
        }
    )
