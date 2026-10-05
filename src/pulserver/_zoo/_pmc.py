"""A shipped sequence playing its navigators, for prospective motion correction."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import pypulseqpp as pp


def navigated(app: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``app`` playing its three-plane navigators, its main sequence setting ``EnablePmc``.

    ``EnablePmc`` asks the proxy to publish the pose the ``pmc``
    reconstruction estimates from each navigator to the scan.
    """

    @functools.wraps(app)
    def design(*args: Any, **kwargs: Any) -> Any:
        designed = app(*args, **kwargs, navigator=True)
        main = designed if isinstance(designed, pp.Sequence) else designed[-1]
        main.set_definition(key="EnablePmc", value=1)
        return designed

    return design
