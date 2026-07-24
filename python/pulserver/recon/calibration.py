"""Torch-native coil-sensitivity calibration adapters.

NLINV delegates to :mod:`pygrog`, the preferred maintained implementation in
this workspace.  PyGROG preserves a Torch tensor's CPU/CUDA device. Pulserver
supplies the dispatch boundary, not a calibration implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = ["estimate_sensitivities", "nlinv_sensitivities"]


def nlinv_sensitivities(
    kspace: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    trajectory: torch.Tensor | None = None,
    image_shape: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Estimate sensitivity maps with PyGROG's NLINV calibration.

    Parameters
    ----------
    kspace : torch.Tensor
        Cartesian data of shape ``(coils, *matrix)`` or non-Cartesian data of
        shape ``(coils, samples)``.
    mask : torch.Tensor, optional
        Cartesian acquisition mask.
    trajectory : torch.Tensor, optional
        Non-Cartesian trajectory with shape ``(samples, dimensions)``.  Its
        presence selects the non-Cartesian NLINV path.
    image_shape : tuple[int, ...], optional
        Required for non-Cartesian data; ignored for Cartesian data.
    **kwargs
        Additional keyword arguments accepted by ``pygrog.utils.nlinv_calib``.

    Returns
    -------
    torch.Tensor
        Coil maps with shape ``(coils, *image_shape)`` on ``kspace.device``.
    """
    try:
        from pygrog.utils import nlinv_calib
    except ImportError as error:
        raise ImportError("NLINV calibration requires pygrog; install pulserver[recon-nlinv].") from error

    if kspace.ndim < 2:
        raise ValueError("kspace must have shape (coils, ...)")
    if trajectory is None:
        result = nlinv_calib(kspace, ndim=kspace.ndim - 1, mask=mask, **kwargs)
    else:
        if image_shape is None:
            raise ValueError("image_shape is required for non-Cartesian NLINV")
        result = nlinv_calib(kspace, shape=image_shape, coords=trajectory, **kwargs)
    return result[0] if isinstance(result, tuple) else result


estimate_sensitivities = nlinv_sensitivities
"""Alias for :func:`nlinv_sensitivities`.

PyGROG NLINV is the supported Torch-native sensitivity calibration path.
"""
