"""Log configuration shared by the services."""

from __future__ import annotations

import faulthandler
import logging
import os
import sys
from pathlib import Path

FORMAT = "%(asctime)s %(process)d %(levelname)s %(name)s: %(message)s"


def configure(
    level: int | str = logging.INFO, logfile: Path | str | None = None
) -> None:
    """Log at ``level`` to the standard error, and send that to ``logfile`` when given.

    ``logfile`` is appended to by redirecting the standard output and error
    descriptors, so ``print``, warnings, native output, a crash's traceback
    and every child process the caller starts afterwards land in it too.
    Output is unbuffered in this process and in spawned children, so lines
    from several processes interleave in the order they were written.
    """
    if logfile is not None:
        redirect(logfile)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=FORMAT, force=True)
    logging.captureWarnings(True)
    faulthandler.enable()


def redirect(logfile: Path | str) -> None:
    """Append the standard output and error of this process to ``logfile``."""
    sys.stdout.flush()
    sys.stderr.flush()
    descriptor = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    os.environ["PYTHONUNBUFFERED"] = "1"
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(line_buffering=True)
