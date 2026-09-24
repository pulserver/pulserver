"""Pushing a stored design to the design intake of a reconstruction computer."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ._store import DesignStore

_TIMEOUT = 60.0


def push(
    store: DesignStore, design: str, url: str, *, timeout: float = _TIMEOUT
) -> bool:
    """Send a stored design to the intake at ``url``; return whether it was sent.

    A design the intake holds already is not sent again.

    Raises
    ------
    ValueError
        If ``url`` is not an ``http`` or ``https`` URL.
    FileNotFoundError
        If the store holds no such design.
    OSError
        If the intake cannot be reached, or refuses the design.
    """
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(f"a design intake is an http or https URL, not {url!r}")
    target = f"{url.rstrip('/')}/designs/{design}"
    try:
        _open(target, timeout, method="HEAD")
        return False
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise OSError(
                f"the design intake at {url} answered {error.code} {error.reason}"
            ) from None
    bundle = store.pack(design)
    try:
        _open(
            target,
            timeout,
            data=bundle,
            method="PUT",
            headers={"Content-Type": "application/gzip"},
        )
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace").strip()
        raise OSError(
            f"the design intake at {url} refused design {design}: "
            f"{detail or error.reason}"
        ) from None
    return True


def _open(target: str, timeout: float, **options: Any) -> None:
    # push() admits http and https targets only.
    request = urllib.request.Request(target, **options)  # noqa: S310 # nosec B310
    with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310 # nosec B310
        pass
