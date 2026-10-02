"""EPI reconstruction with bartorch: ramp sampling and odd/even phase corrected per readout, then ``bart pics``."""

from __future__ import annotations

__all__ = ["PLUGIN", "EpiPhaseCorrection", "EpiRecon", "RampSampling"]

import numpy as np

from ...mrd._header import EncodingSpace
from ...mrd._metadata import acquisition_label, has_acquisition_flag
from .._buffers import ReconData, discards, echo_centre
from ..plugin import Gadget
from .pics import PicsRecon


class RampSampling(Gadget):
    """Resample each readout from the k-space positions it was sampled at onto its space's readout grid.

    The positions are the readout's trajectory, ramps included, which the
    proxy's enrichment writes in 1/m; the grid is the reconstruction matrix's
    readout at its field of view, so readout oversampling is removed with it
    (:func:`bartorch.tools.epi_ramp_operator`). A reversed readout is
    resampled onto the grid in its own direction. The resampled readout is a
    full echo of the matrix's readout: its ``center_sample`` is the grid's
    centre, and it has no discards. A space that states no matrix, such as the
    navigators', takes the geometry of the first space that does. Readouts
    without a trajectory pass unchanged.
    """

    def startup(self, context):
        super().startup(context)
        encodings = getattr(context.header, "encoding", None) or ()
        spaces = [
            EncodingSpace.from_header(context.header, index)
            for index in range(len(encodings))
        ]
        stated = [
            (space.recon_matrix[-1], space.recon_fov[-1])
            for space in spaces
            if space.recon_fov and space.recon_fov[-1] > 0
        ]
        self.geometry = [
            (space.recon_matrix[-1], space.recon_fov[-1])
            if space.recon_fov and space.recon_fov[-1] > 0
            else stated[0]
            if stated
            else None
            for space in spaces
        ]
        self.operators = {}

    def __call__(self, acquisition, data):
        traj = getattr(acquisition, "traj", None)
        space = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
        if traj is None or np.size(traj) == 0 or space >= len(self.geometry):
            return data
        if self.geometry[space] is None:
            return data
        columns, fov = self.geometry[space]
        positions = np.asarray(traj)[:, 0] * fov / columns
        key = positions.tobytes()
        if key not in self.operators:
            import torch
            from bartorch import tools

            grid = (np.arange(columns) - columns // 2) / columns
            if positions[-1] < positions[0]:
                grid = grid[::-1]
            self.operators[key] = (
                tools.epi_ramp_operator(
                    torch.from_numpy(positions), torch.from_numpy(grid.copy()), columns
                )
                .numpy()
                .T
            )
        acquisition.center_sample = (
            columns - 1 - columns // 2 if positions[-1] < positions[0] else columns // 2
        )
        acquisition.discard_pre = acquisition.discard_post = 0
        return data @ self.operators[key]


class EpiPhaseCorrection(Gadget):
    """Remove the odd/even phase of EPI readouts, fitted to each shot's navigator.

    Navigator readouts (``NAV``) are consumed: each run of three, of
    alternating polarity, gives the phase
    (:func:`bartorch.tools.estimate_epi_phase`) applied to the reversed
    readouts (``REV``) that follow it. Each of those is flipped into forward
    order (:func:`bartorch.tools.correct_lines`), and its ``center_sample``
    and discards are mirrored with it. A run of polarity ``- + -`` measures
    the phase of the forward readout against the reversed ones, the negative
    of the correction.
    """

    def startup(self, context):
        super().startup(context)
        self.navigator = []
        self.phase = None

    def __call__(self, acquisition, data):
        import torch
        from bartorch import tools

        reverse = has_acquisition_flag(acquisition, "ACQ_IS_REVERSE")
        line = (torch.from_numpy(np.ascontiguousarray(data)), reverse)
        if has_acquisition_flag(acquisition, "ACQ_IS_NAVIGATION_DATA"):
            self.navigator.append(line)
            if len(self.navigator) == 3:
                phase = tools.estimate_epi_phase(tools.correct_lines(self.navigator))
                self.phase = -phase if self.navigator[0][1] else phase
                self.navigator = []
            return None
        if reverse:
            samples = data.shape[-1]
            acquisition.center_sample = samples - 1 - echo_centre(acquisition, samples)
            before, after = discards(acquisition)
            acquisition.discard_pre, acquisition.discard_post = after, before
        return tools.correct_lines([line], self.phase)[0].numpy()


class EpiRecon(PicsRecon):
    """Images of :class:`~pulserver.recon.handlers.pics.PicsRecon` from EPI readouts.

    Each readout is ramp-resampled (:class:`RampSampling`) and has its
    odd/even phase removed (:class:`EpiPhaseCorrection`) on arrival, so the
    k-space is Cartesian, forward-ordered and at the reconstruction matrix
    when its slice closes. The phase-encode-reversed reference volume
    (``SET`` 1) is reconstructed as an image of its own, and when PyHySCO is
    installed (bartorch's ``pyhysco`` extra, GPL-3.0) each later image of the
    same slice is corrected for susceptibility distortion against it
    (:func:`bartorch.tools.correct_susceptibility`).
    """

    def __init__(self, wavelet: float = 0.005, iterations: int = 30) -> None:
        super().__init__(wavelet, iterations)
        self.gadgets = (RampSampling(), EpiPhaseCorrection())

    def startup(self, context):
        super().startup(context)
        self.references = {}

    def image(self, kspace, shape, device, data: ReconData | None = None):
        image = super().image(kspace, shape, device)
        if data is None:
            return image
        if int(data.counters.get("set", 0)) == 1:
            self.references[int(data.counters.get("slice", 0))] = image
            return image
        reference = self.references.get(int(data.counters.get("slice", 0)))
        if reference is None or image.ndim != 2:
            return image
        space = data.data.space
        if not space.recon_fov:
            return image
        voxel = tuple(
            1e3 * fov / n
            for fov, n in zip(space.recon_fov, space.recon_matrix, strict=False)
        )
        return undistorted(image, reference, voxel[-2:])


PLUGIN = EpiRecon()


def undistorted(image, reverse, voxel_size):
    """Return the ``(phase encode, readout)`` image corrected for susceptibility distortion against its phase-encode-reversed ``reverse``.

    PyHySCO corrects volumes, so the pair is passed as three identical
    slices, which its smoothness penalty leaves uncoupled. ``image`` is
    returned unchanged when PyHySCO is not installed.
    """
    import torch
    from bartorch import tools

    def volume(plane):
        plane = np.ascontiguousarray(plane, dtype=np.float64)
        return torch.from_numpy(np.repeat(plane[:, :, None], 3, axis=2))

    try:
        corrected = tools.correct_susceptibility(
            volume(image),
            volume(reverse * (np.sum(image) / max(np.sum(reverse), 1e-30))),
            voxel_size=(*voxel_size, 1.0),
            phase_encoding_axis=0,
        )
    except ImportError:
        return image
    return np.clip(corrected.blip_up[:, :, 1].cpu().numpy(), 0.0, None)
