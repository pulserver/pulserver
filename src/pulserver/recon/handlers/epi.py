"""EPI reconstruction with bartorch: ramp sampling and odd/even phase corrected per readout, then ``bart pics``."""

from __future__ import annotations

__all__ = ["PLUGIN", "EpiPhaseCorrection", "EpiRecon", "RampSampling"]

import logging
import time

import numpy as np

from ...mrd._header import EncodingSpace
from ...mrd._metadata import acquisition_label, has_acquisition_flag
from .._buffers import ReconData, discards, echo_centre
from ..gadgets import AsymmetricEcho, Prewhiten, RemoveReadoutOversampling
from ..plugin import Gadget, ReconContext
from .pics import PicsRecon

_log = logging.getLogger("pulserver.recon")


class RampSampling(Gadget):
    """Resample each readout from the k-space positions it was sampled at onto its space's readout grid.

    The positions are the readout's trajectory, ramps included, which the
    proxy's enrichment writes in 1/m; the grid is the reconstruction matrix's
    readout at its field of view, so readout oversampling is removed with it
    (:func:`bartorch.tools.epi_ramp_operator`). A reversed readout is
    resampled onto the grid in its own direction. The resampled readout is a
    full echo of the matrix's readout: its ``center_sample`` is the grid's
    centre, and it has no discards. A space that states no matrix, such as the
    navigators', takes the geometry of the first space that does. A readout
    without a trajectory, such as a line of the gradient-echo calibration
    that precedes an accelerated scan, is sampled uniformly: it is completed
    to a full echo (:class:`~pulserver.recon.AsymmetricEcho`) and its readout
    oversampling removed (:class:`~pulserver.recon.RemoveReadoutOversampling`),
    so it lands on the same readout grid as the EPI lines and the coil
    sensitivities estimated from it serve them.
    """

    def startup(self, context):
        super().startup(context)
        self.uniform = (AsymmetricEcho(), RemoveReadoutOversampling())
        for gadget in self.uniform:
            gadget.startup(context)
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
        if traj is None or np.size(traj) == 0:
            for gadget in self.uniform:
                data = gadget(acquisition, data)
            return data
        if space >= len(self.geometry):
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


def _echo_index(acquisition) -> int:
    """Position of a readout in its EPI train: the ``user[0]`` counter (``USER0``), 0 where absent."""
    counters = getattr(getattr(acquisition, "idx", None), "user", None)
    try:
        return int(counters[0])
    except (TypeError, IndexError):
        return 0


class EpiPhaseCorrection(Gadget):
    """Remove the odd/even phase of EPI readouts, fitted to a blip-free reference train or to each shot's navigator.

    Reference readouts (``ACQ_IS_PHASECORR_DATA``) are consumed: the lines of
    one train, each at its position in the train (``user[0]``), give a
    constant and linear phase per echo
    (:func:`bartorch.tools.estimate_epi_echo_phase`) once the first readout
    after them arrives. Each reversed readout (``REV``) of the same slice is
    corrected by the phase of its own echo; a readout past the reference's
    last echo takes the last reversed echo's, and a slice without a reference
    of its own the latest reference of any slice. Without a reference,
    navigator readouts (``NAV``) are used: each run of three, of alternating
    polarity, gives the phase (:func:`bartorch.tools.estimate_epi_phase`)
    applied to the reversed readouts that follow it, a run of polarity
    ``- + -`` measuring the negative of the correction. With neither, the
    reversed readouts are only flipped, and a ``WARNING`` says so once. Each
    reversed readout is flipped into forward order
    (:func:`bartorch.tools.correct_lines`), and its ``center_sample`` and
    discards are mirrored with it. Navigator readouts are consumed whether or
    not a reference serves.
    """

    def startup(self, context):
        super().startup(context)
        self.navigator = []
        self.phase = None
        self.pending = {}
        self.references = {}
        self.latest = None
        self._warned = False

    def _close_references(self) -> None:
        """Fit every reference train still collecting."""
        from bartorch import tools

        for location, lines in self.pending.items():
            lines = sorted(lines, key=lambda item: item[0])
            polarity = [reverse for _, (_, reverse) in lines]
            try:
                fit = tools.estimate_epi_echo_phase(
                    tools.correct_lines([line for _, line in lines]), polarity
                )
            except ValueError as error:
                _log.warning(
                    "the odd/even phase reference of slice %d is unusable (%s); "
                    "it is ignored",
                    location,
                    error,
                )
                continue
            echoes = [echo for echo, _ in lines]
            table = {echo: fit[row] for row, echo in enumerate(echoes)}
            last = max(
                (e for e, r in zip(echoes, polarity, strict=True) if r), default=None
            )
            self.references[location] = (table, last)
            self.latest = location
            slope = float(fit[[r for r, flag in enumerate(polarity) if flag], 1].mean())
            _log.info(
                "odd/even phase of slice %d fitted to a %d-echo reference train: "
                "mean linear term %.4f rad across the readout",
                location,
                len(lines),
                slope,
            )
        self.pending = {}

    def _reference_phase(self, acquisition):
        if not self.references:
            return None
        location = int(acquisition_label(acquisition, "slice", 0) or 0)
        table, last = self.references.get(location, self.references[self.latest])
        echo = _echo_index(acquisition)
        if echo in table:
            return table[echo]
        return None if last is None else table[last]

    def __call__(self, acquisition, data):
        import torch
        from bartorch import tools

        reverse = has_acquisition_flag(acquisition, "ACQ_IS_REVERSE")
        line = (torch.from_numpy(np.ascontiguousarray(data)), reverse)
        if has_acquisition_flag(acquisition, "ACQ_IS_PHASECORR_DATA"):
            location = int(acquisition_label(acquisition, "slice", 0) or 0)
            echo = _echo_index(acquisition)
            collected = self.pending.get(location)
            if collected and echo <= collected[-1][0]:
                # A new train of a slice still collecting closes the last.
                self._close_references()
            self.pending.setdefault(location, []).append((echo, line))
            return None
        if self.pending:
            self._close_references()
        if has_acquisition_flag(acquisition, "ACQ_IS_NAVIGATION_DATA"):
            self.navigator.append(line)
            if len(self.navigator) == 3:
                phase = tools.estimate_epi_phase(tools.correct_lines(self.navigator))
                self.phase = -phase if self.navigator[0][1] else phase
                self.navigator = []
            return None
        phase = None
        if reverse:
            samples = data.shape[-1]
            acquisition.center_sample = samples - 1 - echo_centre(acquisition, samples)
            before, after = discards(acquisition)
            acquisition.discard_pre, acquisition.discard_post = after, before
            phase = self._reference_phase(acquisition)
            if phase is None:
                phase = self.phase
            if phase is None and not self._warned:
                self._warned = True
                _log.warning(
                    "the EPI readouts carry neither an odd/even phase reference nor "
                    "a navigator: reversed lines are flipped but not phase "
                    "corrected, and the images will show an N/2 ghost"
                )
        return tools.correct_lines([line], phase)[0].numpy()


class EpiRecon(PicsRecon):
    """Images of :class:`~pulserver.recon.handlers.pics.PicsRecon` from EPI readouts.

    Each readout is whitened (:class:`~pulserver.recon.Prewhiten`),
    ramp-resampled (:class:`RampSampling`) and has its odd/even phase removed
    (:class:`EpiPhaseCorrection`, from the blip-free reference train
    ``ACQ_IS_PHASECORR_DATA`` the sequence plays in a dummy shot, else from
    the navigator) on arrival, so the k-space is Cartesian,
    forward-ordered and at the reconstruction matrix when its slice closes.
    The phase-encode-reversed reference volume (``SET`` 1) is reconstructed and
    kept, Fermi filtered as the images are, but sent as no image of the
    series: when PyHySCO is installed (bartorch's ``pyhysco`` extra, GPL-3.0)
    the first later image of the same slice, or volume of a 3D scan, and the
    reference give the slice's susceptibility displacement field
    (:func:`bartorch.tools.correct_susceptibility`), which is kept and applied
    to that image and every later one of the slice
    (:func:`bartorch.tools.apply_susceptibility`, as FSL's ``applytopup
    --method=jac``), history ``HYSCO``. The
    correction is logged as the other corrections are, and an image left
    uncorrected, for want of PyHySCO or of its reference, is logged with the
    reason at ``WARNING``. The calibration readouts of an accelerated scan
    reach the same readout grid as the EPI lines (:class:`RampSampling`), so
    the coil sensitivities estimated from them unfold it. A FLEET
    calibration's readouts are EPI lines, ramp-resampled and phase-corrected
    as the imaging lines are, read once in each phase-encode polarity: each set is unfolded by the maps of its own polarity, distorted
    as its image is (:func:`~pulserver.recon.coil_maps`). Each image is Fermi
    filtered before its magnitude is taken, and corrected for gradient
    nonlinearity and divided by the receive field after the distortion
    correction, the reference set by a field of its own
    (:class:`~pulserver.recon.Corrections`).

    Parameters
    ----------
    wavelet : float, default=0.005
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.
    iterations : int, default=30
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.
    kspace_filter, gradient_unwarp, intensity_correction : bool, default=True
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.
    """

    def __init__(
        self,
        wavelet: float = 0.005,
        iterations: int = 30,
        *,
        kspace_filter: bool = True,
        gradient_unwarp: bool = True,
        intensity_correction: bool = True,
    ) -> None:
        super().__init__(
            wavelet,
            iterations,
            kspace_filter=kspace_filter,
            gradient_unwarp=gradient_unwarp,
            intensity_correction=intensity_correction,
        )
        self.gadgets = (Prewhiten(), RampSampling(), EpiPhaseCorrection())
        # The odd/even phase reference is consumed by EpiPhaseCorrection.
        self.reject_flags = ()

    def startup(self, context):
        super().startup(context)
        self.references = {}
        # The displacement field of each slice, estimated once from its first
        # image and its reference.
        self.fields = {}
        self._skips_logged = set()

    def recon(self, context: ReconContext, branch: str, data: ReconData):
        if int(data.counters.get("set", 0)) != 1:
            return super().recon(context, branch, data)
        # The reference only corrects the images of its slice (image() keeps
        # it); it is not an image of the series.
        self.unit_image(context, data)
        return None

    def image(
        self,
        kspace: np.ndarray,
        shape: tuple[int, ...],
        context: ReconContext,
        data: ReconData,
    ) -> np.ndarray:
        image = np.abs(super().image(kspace, shape, context, data))
        location = int(data.counters.get("slice", 0))
        if int(data.counters.get("set", 0)) == 1:
            self.references[location] = image
            self.fields.pop(location, None)
            return image
        reference = self.references.get(location)
        space = data.data.space
        if reference is None:
            reason = (
                f"no phase-encode-reversed reference (set 1) of slice {location} "
                "came before it"
            )
        elif image.shape != reference.shape:
            reason = f"its reference is {reference.shape}, the image {image.shape}"
        elif image.ndim not in (2, 3):
            reason = f"it has {image.ndim} dimensions, not 2 or 3"
        elif not space.recon_fov:
            reason = "the header states no reconstruction field of view"
        else:
            voxel = tuple(
                1e3 * fov / n
                for fov, n in zip(space.recon_fov, space.recon_matrix, strict=False)
            )
            voxel = voxel[-image.ndim :]
            start = time.perf_counter()
            field = self.fields.get(location)
            estimated = field is None
            try:
                if estimated:
                    field = susceptibility_field(image, reference, voxel)
                corrected = undistorted(image, field, voxel)
            except ImportError as error:
                reason = f"PyHySCO is not installed ({error})"
            except Exception:
                _log.exception(
                    "the susceptibility distortion correction failed; the image is "
                    "sent uncorrected"
                )
                return image
            else:
                self.fields[location] = field
                if self.corrections is not None:
                    self.corrections.ran(
                        "HYSCO",
                        "susceptibility distortion of a %s image corrected in %.2f s "
                        "by %s",
                        image.shape,
                        time.perf_counter() - start,
                        (
                            f"the field of slice {location} estimated against its "
                            "phase-encode-reversed reference"
                            if estimated
                            else f"applying the field of slice {location}"
                        ),
                    )
                return corrected
        level = logging.DEBUG if reason in self._skips_logged else logging.WARNING
        self._skips_logged.add(reason)
        _log.log(
            level,
            "slice %d is not corrected for susceptibility distortion: %s",
            location,
            reason,
        )
        return image


PLUGIN = EpiRecon()


def susceptibility_field(image, reverse, voxel_size):
    """Return the displacement field of ``image`` against its phase-encode-reversed ``reverse``.

    A slice is ``(phase encode, readout)`` and a volume ``(partition, phase
    encode, readout)``, with ``voxel_size`` in mm along the same axes. The
    field is PyHySCO's (:func:`bartorch.tools.correct_susceptibility`), in mm
    on the cell edges of the phase encode, in the axes of ``image``: one row
    (a volume: one phase-encode line) longer. PyHySCO corrects volumes: a
    volume is corrected as one, its phase encode first, and a slice as three
    identical slices, which its smoothness penalty leaves uncoupled, the
    middle one's field kept.

    Raises
    ------
    ImportError
        If PyHySCO is not installed.
    """
    import torch
    from bartorch import tools

    image = np.asarray(image, dtype=np.float64)
    reverse = np.asarray(reverse, dtype=np.float64)
    scale = np.sum(image) / max(np.sum(reverse), 1e-30)
    if image.ndim == 2:
        up = np.repeat(image[:, :, None], 3, axis=2)
        down = np.repeat(reverse[:, :, None], 3, axis=2)
        voxel = (*voxel_size, 1.0)
    else:
        # (partition, phase, readout) to (phase, readout, partition).
        up, down = image.transpose(1, 2, 0), reverse.transpose(1, 2, 0)
        voxel = (voxel_size[1], voxel_size[2], voxel_size[0])
    corrected = tools.correct_susceptibility(
        torch.from_numpy(np.ascontiguousarray(up)),
        torch.from_numpy(np.ascontiguousarray(down * scale)),
        voxel_size=voxel,
        phase_encoding_axis=0,
    )
    field = corrected.field_map.detach().cpu().numpy()
    return field[:, :, 1] if image.ndim == 2 else field.transpose(2, 0, 1)


def undistorted(image, field, voxel_size):
    """Return ``image`` corrected for susceptibility distortion by its slice's displacement ``field``.

    ``field`` is :func:`susceptibility_field`'s, of the first image of the
    slice: the image is sampled at the displaced position along its phase
    encode and its intensity modulated by the field's derivative
    (:func:`bartorch.tools.apply_susceptibility`).
    """
    import torch
    from bartorch import tools

    image = np.asarray(image, dtype=np.float64)
    corrected = tools.apply_susceptibility(
        torch.from_numpy(np.ascontiguousarray(image)),
        torch.from_numpy(np.ascontiguousarray(field)),
        voxel_size=tuple(voxel_size),
        phase_encoding_axis=image.ndim - 2,
    )
    return np.clip(corrected.cpu().numpy(), 0.0, None)
