"""``nufft`` of readouts that number their place in a train of excitations, the train taken as one image."""

from __future__ import annotations

__all__ = ["PLUGIN", "NufftTrainRecon"]

from .nufft import NufftRecon


class NufftTrainRecon(NufftRecon):
    """:class:`NufftRecon` of the readouts of every place in a train of excitations as one image.

    A sequence that plays a train of excitations after one preparation, an
    inversion, and numbers each readout by its place in the train
    (``contrast``, ``ECO``) would be reconstructed as one image per place,
    each from a single view per partition. Here the place is a merged counter
    (:class:`~pulserver.recon.ReconPlugin`): readouts are placed by their other
    counters, which differ between places, and a unit closes once its closing
    flag has arrived at every place.

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


PLUGIN = NufftTrainRecon()
