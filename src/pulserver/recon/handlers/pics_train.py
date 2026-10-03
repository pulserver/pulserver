"""Cartesian parallel-imaging compressed-sensing reconstruction of the image the readouts of a train fill, as ``bart nlinv``, ``bart pics`` and ``bart homodyne``."""

from __future__ import annotations

__all__ = ["PLUGIN", "PicsTrainRecon"]

from .pics import PicsRecon


class PicsTrainRecon(PicsRecon):
    """Images of :class:`~pulserver.recon.handlers.pics.PicsRecon` from readouts whose ``contrast`` counter numbers their place in a train.

    The echoes of a fast spin echo train and the excitations of one inversion
    shot are labelled ``ECO`` by their place in the train, and fill different
    phase encodes of one k-space. ``contrast`` is therefore neither an image
    selector nor an axis: it is merged, so a unit holds one k-space that every
    value of it fills, and closes when ``LAST_IN_SLICE`` has arrived for each.

    Two readouts of a unit that share their phase encode and partition replace
    one another, with a warning. A scan whose contrasts are images, such as a
    multi-echo gradient echo, is reconstructed by
    :class:`~pulserver.recon.handlers.pics.PicsRecon`.

    Parameters
    ----------
    wavelet
        ``lambda``, relative to the data scaling ``pics`` estimates.
    iterations
        Iterations of the solve.
    """

    def __init__(self, wavelet: float = 0.005, iterations: int = 30) -> None:
        super().__init__(wavelet, iterations)
        self.merge = ("contrast",)


PLUGIN = PicsTrainRecon()
