"""State containers for iterative reconstruction algorithms."""

from __future__ import annotations

__all__ = ["OptimResult", "OptimState"]

from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Any

import torch


@dataclass(frozen=True)
class OptimState:
    """State of an iterative reconstruction algorithm.

    Parameters
    ----------
    variables
        Algorithm-owned primal and auxiliary variables.
    iteration
        Number of completed iterations.
    """

    variables: dict[str, Any]
    iteration: int = 0

    @property
    def estimate(self) -> torch.Tensor:
        """Return the current primal estimate."""
        return self.variables["est"][0]

    def detach(self) -> OptimState:
        """Detach every tensor, for example at a truncated-BPTT boundary."""
        return replace(self, variables=_detach(self.variables))


@dataclass(frozen=True)
class OptimResult:
    """Reconstruction result with final state and selected iterates.

    Parameters
    ----------
    reconstruction
        Final reconstructed image.
    state
        Final optimizer state.
    history
        Mapping from iteration number to requested intermediate estimates.
    """

    reconstruction: torch.Tensor
    state: OptimState
    history: dict[int, torch.Tensor]


# %% private module subroutines


def _detach(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _detach(item) for key, item in value.items()}
    detach = getattr(value, "detach", None)
    return detach() if callable(detach) else value
