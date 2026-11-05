"""A shipped sequence playing its navigators, for prospective motion correction."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import pypulseqpp as pp


def navigated(app: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``app`` playing its three-plane navigators, its main sequence setting ``EnablePmc``.

    ``EnablePmc`` asks the proxy to publish the pose the ``pmc``
    reconstruction estimates from each navigator to the scan. The returned
    function's ``enable_pmc`` attribute states as much before any design, so
    that a site following poses designs it under the limits that hold for
    every rotation (:func:`~pulserver.host._limits.design_system`).
    """

    @functools.wraps(app)
    def design(*args: Any, **kwargs: Any) -> Any:
        designed = app(*args, **kwargs, navigator=True)
        main = designed if isinstance(designed, pp.Sequence) else designed[-1]
        main.set_definition(key="EnablePmc", value=1)
        return designed

    design.enable_pmc = True  # type: ignore[attr-defined]
    return design
