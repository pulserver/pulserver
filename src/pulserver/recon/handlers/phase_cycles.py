"""Phase-cycled balanced SSFP: each cycle solved as ``pics`` solves it, over the same coil sensitivities, and the cycles combined into one image free of off-resonance bands."""

from __future__ import annotations

__all__ = [
    "COMBINATIONS",
    "PLUGIN",
    "PhaseCycleRecon",
    "combined",
]

from typing import Any

import numpy as np

from .._buffers import ReconData
from .._calibration import coil_maps
from ..plugin import ReconContext, ReconResult
from .pics import PicsRecon, averaged, compressed, nlinv_estimate

#: Ways the phase cycles are combined (:func:`combined`).
COMBINATIONS = ("rss", "maximum", "complex")


def combined(images: np.ndarray, combination: str = "rss") -> np.ndarray:
    """Return the phase cycles ``(cycles, ...)`` of a balanced SSFP combined into one magnitude image.

    ``rss`` is the root sum of squares of the cycles, the combination of
    highest SNR efficiency that leaves a small residual band modulation;
    ``maximum`` the largest magnitude of any cycle (maximum-intensity
    projection, as CISS combines its two cycles), which flattens the bands
    furthest at a cost in noise; ``complex`` the magnitude of the complex sum,
    which with cycles equally spaced over 2 pi cancels the odd harmonics of
    the off-resonance profile.

    Parameters
    ----------
    images : ndarray
        Complex images, one per phase cycle along the first axis.
    combination : {'rss', 'maximum', 'complex'}, default='rss'
        How the cycles are combined.

    Returns
    -------
    ndarray
        Real, non-negative image of the shape of one cycle.

    Raises
    ------
    ValueError
        If ``combination`` is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> cycles = np.array([[3.0 + 0j], [4.0j]])
    >>> [float(combined(cycles, c)[0]) for c in COMBINATIONS]
    [5.0, 4.0, 5.0]
    """
    if combination not in COMBINATIONS:
        raise ValueError(
            f"combination must be one of {COMBINATIONS}, got {combination!r}"
        )
    if combination == "rss":
        return np.sqrt((np.abs(images) ** 2).sum(axis=0))
    if combination == "maximum":
        return np.abs(images).max(axis=0)
    return np.abs(images.sum(axis=0))


class PhaseCycleRecon(PicsRecon):
    """Balanced SSFP whose phase cycles, the ``SET`` counter, are combined into one image.

    The cycles of a slice or volume are buffered together. Each is solved by
    :meth:`~pulserver.recon.handlers.pics.PicsRecon.image` over the coil
    sensitivities of the first cycle's calibration, so that the cycles share
    one receive phase, and the complex images are combined by
    :func:`combined`. A scan of one cycle gives the ``pics`` image.

    Parameters
    ----------
    combination : {'rss', 'maximum', 'complex'}, default='rss'
        How the phase cycles are combined.
    **kwargs
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.

    Raises
    ------
    ValueError
        If ``combination`` is unknown.
    """

    def __init__(self, combination: str = "rss", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if combination not in COMBINATIONS:
            raise ValueError(
                f"combination must be one of {COMBINATIONS}, got {combination!r}"
            )
        self.axes = ("set",)
        self.combination = combination

    def recon(
        self, context: ReconContext, branch: str, data: ReconData
    ) -> list[ReconResult] | ReconResult | None:
        del branch
        data = compressed(context, data, self.virtual_coils)
        if data.data is None and data.ref is not None:
            coil_maps(context, data, estimate=nlinv_estimate)
        buffer = data.data
        if buffer is None:
            return None
        kspace = averaged(buffer)
        axis = buffer.axes.index("set") if "set" in buffer.axes else None
        cycles = [kspace] if axis is None else np.moveaxis(kspace, axis, 0)
        cycles = [c for c in cycles if np.any(c)]
        images = np.stack(
            [self.image(c, buffer.image_shape, context, data) for c in cycles]
        )
        image = combined(images, self.combination)
        corrections = self.corrections
        history = [
            "PULSERVER",
            "PYTHON",
            "PICS",
            f"PHASECYCLES_{self.combination.upper()}",
        ]
        if corrections is not None:
            image = corrections.unwarped(image, context, data)
            history += corrections.applied()
        results = [ReconResult(image, attributes={"ImageProcessingHistory": history})]
        if corrections is not None:
            results = corrections.intensity_corrected(results, data)
        return results[0]


PLUGIN = PhaseCycleRecon()
