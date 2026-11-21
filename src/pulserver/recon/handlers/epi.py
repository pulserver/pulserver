"""EPI reconstruction with bartorch: ramp sampling and odd/even phase corrected per readout, then ``bart pics``."""

from __future__ import annotations

__all__ = ["PLUGIN", "EpiPhaseCorrection", "EpiRecon", "RampSampling"]

import importlib.util
import logging
import time

import numpy as np

from ...mrd._header import EncodingSpace
from ...mrd._metadata import acquisition_label, has_acquisition_flag
from .._buffers import ReconData, discards, echo_centre
from .._calibration import (
    MissingCalibration,
    _zero_filled,
    coil_maps,
)
from ..gadgets import AsymmetricEcho, Prewhiten, RemoveReadoutOversampling
from ..plugin import Gadget, ReconContext, ReconResult
from .pics import (
    VIRTUAL_COILS,
    PicsRecon,
    averaged,
    center_crop,
    compressed,
    nlinv_estimate,
    pics_image,
)

_log = logging.getLogger("pulserver.recon")


class RampSampling(Gadget):
    """Resample each readout from the k-space positions it was sampled at onto its space's readout grid.

    The positions are the readout's trajectory, ramps included, which the
    proxy's enrichment writes in 1/m: the integral of the played gradients at
    the sample times. Delayed by the read-gradient delay :attr:`delay_s`, which
    :class:`EpiPhaseCorrection` fits to the navigator or the phase reference,
    the trajectory of a sample taken at ``t`` is the one designed for
    ``t - delay_s``. The grid is the reconstruction matrix's readout at its
    field of view, so readout oversampling is removed with it
    (:func:`bartorch.tools.epi_ramp_operator`). A reversed readout is
    resampled onto the grid in its own direction. The resampled readout is a
    full echo of the matrix's readout: its ``center_sample`` is the grid's
    centre, and it has no discards. A space that states no matrix, such as the
    navigators', takes the geometry of the first space that does. Navigator
    and phase-reference readouts pass on as they were sampled, for
    :class:`EpiPhaseCorrection` to resample once it has fitted the delay to
    them. A readout without a trajectory, such as a line of the gradient-echo
    calibration that precedes an accelerated scan, is sampled uniformly: it is
    completed to a full echo (:class:`~pulserver.recon.AsymmetricEcho`) and its
    readout oversampling removed
    (:class:`~pulserver.recon.RemoveReadoutOversampling`), so it lands on the
    same readout grid as the EPI lines and the coil sensitivities estimated
    from it serve them.

    Attributes
    ----------
    delay_s : float
        Delay of the played read gradient behind the designed one (s).
    """

    delay_s = 0.0

    def startup(self, context):
        super().startup(context)
        self.delay_s = 0.0
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
        if traj is None or np.size(traj) == 0:
            for gadget in self.uniform:
                data = gadget(acquisition, data)
            return data
        if has_acquisition_flag(
            acquisition, "ACQ_IS_PHASECORR_DATA"
        ) or has_acquisition_flag(acquisition, "ACQ_IS_NAVIGATION_DATA"):
            return data
        return self.resample(acquisition, data)

    def resample(self, acquisition, data, delay_s=None):
        """Return one readout resampled onto its space's readout grid, the trajectory delayed by ``delay_s``.

        ``None`` is :attr:`delay_s`. The acquisition's ``center_sample`` and
        discards are set to those of the grid. A readout of a space without
        geometry, or without a trajectory, is returned as it is.
        """
        traj = getattr(acquisition, "traj", None)
        space = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
        if traj is None or np.size(traj) == 0 or space >= len(self.geometry):
            return data
        if self.geometry[space] is None:
            return data
        columns, fov = self.geometry[space]
        delay = self.delay_s if delay_s is None else float(delay_s)
        k = np.asarray(traj, dtype=np.float64)[:, 0]
        dwell = acquisition_label(acquisition, "sample_time_us", None)
        if delay and dwell:
            k = delayed(k, delay / (1e-6 * float(dwell)))
        positions = k * fov / columns
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


def delayed(k: np.ndarray, samples: float) -> np.ndarray:
    """Return the trajectory ``k`` sampled ``samples`` sample intervals earlier, as a delayed gradient plays it.

    ``k[i]`` becomes the designed trajectory at ``i - samples``, interpolated
    linearly between samples and extrapolated linearly past the first and the
    last.
    """
    index = np.arange(k.size, dtype=np.float64) - samples
    inner = np.interp(index, np.arange(k.size), k)
    before, after = index < 0, index > k.size - 1
    inner[before] = k[0] + (k[1] - k[0]) * index[before]
    inner[after] = k[-1] + (k[-1] - k[-2]) * (index[after] - (k.size - 1))
    return inner


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

    A delay of the read gradient moves every sample along the trajectory by
    the gradient times the delay, which on the ramps is not the shift a linear
    phase removes. The first reference train, or the first navigator, of the
    stream gives the delay: the one that, applied to the trajectory before the
    ramp resampling (:class:`RampSampling`), leaves no linear term in the fit.
    It is found by the secant method on that term, starting from no delay, and
    the gadget's :class:`RampSampling` resamples every later readout with it;
    the phase of each reference and navigator is then fitted to its readouts
    resampled with the delay. Without a :class:`RampSampling` (``ramp`` is
    ``None``), the reference and navigator readouts are taken as they come.

    Attributes
    ----------
    ramp : RampSampling or None
        The resampling the reference and navigator readouts go through, and
        whose delay is set.
    """

    ramp = None

    def startup(self, context):
        super().startup(context)
        self.navigator = []
        self.phase = None
        self.pending = {}
        self.references = {}
        self.latest = None
        self._warned = False
        self._delay_fitted = False

    def _lines(self, collected, delay=None):
        """Return the readouts of ``collected``, ``(acquisition, data, reverse)``, resampled and in forward order."""
        import torch
        from bartorch import tools

        lines = []
        for acquisition, data, reverse in collected:
            if self.ramp is not None:
                data = self.ramp.resample(acquisition, data, delay)
            lines.append((torch.from_numpy(np.ascontiguousarray(data)), reverse))
        return tools.correct_lines(lines)

    def _fit_delay(self, linear_term, collected) -> None:
        """Set the ramp resampling's delay to the one that zeroes ``linear_term(delay)``, once a stream."""
        if self._delay_fitted or self.ramp is None:
            return
        self._delay_fitted = True
        if not any(np.size(getattr(a, "traj", None)) > 1 for a, _, _ in collected):
            return
        step = DELAY_STEP_S
        delays, terms = [0.0, step], [linear_term(0.0), linear_term(step)]
        for _ in range(DELAY_ITERATIONS):
            slope = (terms[-1] - terms[-2]) / (delays[-1] - delays[-2])
            if not np.isfinite(slope) or slope == 0.0:
                break
            delay = delays[-1] - terms[-1] / slope
            if abs(delay) > MAX_DELAY_S:
                _log.warning(
                    "the read-gradient delay fitted to the odd/even phase, "
                    "%.2f us, is beyond %.0f us and is not applied",
                    1e6 * delay,
                    1e6 * MAX_DELAY_S,
                )
                return
            delays.append(delay)
            terms.append(linear_term(delay))
        self.ramp.delay_s = delays[-1]
        _log.info(
            "read-gradient delay of %.2f us fitted to the odd/even phase; the "
            "ramp resampling delays the trajectory by it",
            1e6 * self.ramp.delay_s,
        )

    def _close_references(self) -> None:
        """Fit every reference train still collecting."""
        from bartorch import tools

        for location, collected in self.pending.items():
            collected = sorted(collected, key=lambda item: item[0])
            echoes = [echo for echo, _ in collected]
            lines = [line for _, line in collected]
            polarity = [reverse for _, _, reverse in lines]
            rows = [r for r, flag in enumerate(polarity) if flag]

            def fit(delay=None, lines=lines, polarity=polarity):
                return tools.estimate_epi_echo_phase(
                    self._lines(lines, delay), polarity
                )

            try:
                self._fit_delay(
                    lambda d, rows=rows: float(fit(d)[rows, 1].mean()), lines
                )
                coefficients = fit()
            except ValueError as error:
                _log.warning(
                    "the odd/even phase reference of slice %d is unusable (%s); "
                    "it is ignored",
                    location,
                    error,
                )
                continue
            table = {echo: coefficients[row] for row, echo in enumerate(echoes)}
            last = max(
                (e for e, r in zip(echoes, polarity, strict=True) if r), default=None
            )
            self.references[location] = (table, last)
            self.latest = location
            _log.info(
                "odd/even phase of slice %d fitted to a %d-echo reference train: "
                "mean linear term %.4f rad across the readout",
                location,
                len(lines),
                float(coefficients[rows, 1].mean()),
            )
        self.pending = {}

    def _navigator_phase(self, delay=None):
        from bartorch import tools

        phase = tools.estimate_epi_phase(self._lines(self.navigator, delay))
        return -phase if self.navigator[0][2] else phase

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
        if has_acquisition_flag(acquisition, "ACQ_IS_PHASECORR_DATA"):
            location = int(acquisition_label(acquisition, "slice", 0) or 0)
            echo = _echo_index(acquisition)
            collected = self.pending.get(location)
            if collected and echo <= collected[-1][0]:
                # A new train of a slice still collecting closes the last.
                self._close_references()
            self.pending.setdefault(location, []).append(
                (echo, (acquisition, data, reverse))
            )
            return None
        if self.pending:
            self._close_references()
        if has_acquisition_flag(acquisition, "ACQ_IS_NAVIGATION_DATA"):
            self.navigator.append((acquisition, data, reverse))
            if len(self.navigator) == 3:
                self._fit_delay(
                    lambda d: float(self._navigator_phase(d)[1]), self.navigator
                )
                self.phase = self._navigator_phase()
                self.navigator = []
            return None
        line = (torch.from_numpy(np.ascontiguousarray(data)), reverse)
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


#: First trial delay of the read gradient (s), and the secant steps after it.
DELAY_STEP_S, DELAY_ITERATIONS = 1e-6, 3

#: Largest read-gradient delay (s) applied to the trajectory; a larger fit
#: points at a reference that does not describe the train.
MAX_DELAY_S = 20e-6


class EpiRecon(PicsRecon):
    """Images of EPI readouts: coil sensitivities of the whole stack from its calibration, a linear SENSE solve, one distortion field for every image.

    Each readout is whitened (:class:`~pulserver.recon.Prewhiten`),
    ramp-resampled along its k-space trajectory (:class:`RampSampling`) and
    has its odd/even phase removed (:class:`EpiPhaseCorrection`, from the
    blip-free reference train ``ACQ_IS_PHASECORR_DATA`` the sequence plays in
    a dummy shot, else from the navigator) on arrival, so the k-space is
    Cartesian, forward-ordered and at the reconstruction matrix when its slice
    closes. Each unit's channels are compressed to ``virtual_coils``
    (:class:`~pulserver.recon.CompressChannels`).

    Coil sensitivities come from the calibration the scan opens with, once
    for the whole stack: the calibration units of every slice, held until the
    first unit that needs maps, are each estimated on their own
    (:func:`~pulserver.recon.coil_maps` with
    :func:`epi_nlinv_estimate`), each phase-encode
    polarity of a FLEET calibration apart, so each set is unfolded by maps
    distorted as its image is. A slice stack with gaps is no contiguous
    volume, and maps fitted smooth across it fit no slice well enough for a
    multiband or in-plane unfolding. Every later image of a slice takes its
    slice's maps (:func:`~pulserver.recon.coil_maps`). A scan without a
    calibration has the maps of each slice and set fitted once, to its first
    unit, and kept.

    A unit whose lines fill k-space within the range acquired along the phase
    encode is combined as the least-squares SENSE image of its maps,
    ``sum_c conj(S_c) F^-1 y_c / sum_c |S_c|^2``; an undersampled one is the
    wavelet-regularised ``pics`` solve of
    :class:`~pulserver.recon.handlers.pics.PicsRecon` at its ``wavelet`` on
    maps carrying the smooth phase (:func:`smooth_phase`) of its
    ``iterations``-step conjugate-gradient image, so the prior acts on a near-real image, or
    with ``wavelet`` zero the solution of ``min |P F S x - y|^2`` by
    ``iterations`` conjugate-gradient steps (:class:`bartorch.optim.CG`). A
    partial-Fourier phase encode is completed by ``pocs_iterations`` of POCS
    (:func:`pocs`), which keeps the image phase.

    A multiband unit whose bands each lie on a calibrated slice has its bands
    separated by split-slice GRAPPA (:func:`slice_grappa_bands`), with kernels
    trained on those slices' calibration k-space, and each band is then
    solved as a slice of its own, with its slice's maps. A multiband unit
    without that calibration is solved as one volume of its bands.

    When PyHySCO is installed (bartorch's ``pyhysco`` extra, GPL-3.0), one
    susceptibility displacement field is estimated for the whole stack
    (:func:`bartorch.tools.correct_susceptibility`, :func:`stack_field`) from
    the root-sum-of-squares images of the FLEET calibration's two
    phase-encode polarities. They share the imaging's echo spacing, and so
    its distortion, at the calibration's short echo time, where the
    signal near air-tissue interfaces has not yet dephased. Without them,
    the phase-encode-reversed reference volume (``SET`` 1), reconstructed and
    kept, Fermi filtered as the images are, but sent as no image of the
    series, is the reference: the first image of every slice of the stack is
    held until the stack is complete, and that volume and the reference
    volume give the field. The field is estimated once and applied to every
    image of every slice
    (:func:`bartorch.tools.apply_susceptibility`, as FSL's ``applytopup
    --method=jac``), history ``HYSCO``. The correction is logged as the other
    corrections are, and an image left uncorrected, for want of PyHySCO or of
    its reference, is logged with the reason at ``WARNING``. Each image is
    Fermi filtered before its magnitude is taken, and corrected for gradient
    nonlinearity and divided by the receive field after the distortion
    correction (:class:`~pulserver.recon.Corrections`).

    Parameters
    ----------
    wavelet : float | None, default=None
        ``lambda`` of a wavelet ``pics`` solve of undersampled units; zero
        solves them by conjugate gradients. ``None`` takes ``EPI_WAVELET_SLAB``
        for a unit of more than one partition and ``EPI_WAVELET_SLICE`` for a
        slice.
    iterations : int, default=30
        Conjugate-gradient steps, or ``pics`` iterations with ``wavelet``.
    pocs_iterations : int, default=8
        POCS iterations of a partial-Fourier unit.
    l2 : float, default=0.0
        Tikhonov weight ``lambda`` of the conjugate-gradient solve, on data
        scaled as ``pics`` scales them.
    virtual_coils : int | None, default=VIRTUAL_COILS
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.
    kspace_filter, gradient_unwarp, intensity_correction : bool, default=True
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.
    """

    def __init__(
        self,
        wavelet: float | None = None,
        iterations: int = 30,
        *,
        pocs_iterations: int = 8,
        l2: float = 0.0,
        virtual_coils: int | None = VIRTUAL_COILS,
        kspace_filter: bool = True,
        gradient_unwarp: bool = True,
        intensity_correction: bool = True,
    ) -> None:
        super().__init__(
            0.0 if wavelet is None else wavelet,
            iterations,
            virtual_coils,
            kspace_filter=kspace_filter,
            gradient_unwarp=gradient_unwarp,
            intensity_correction=intensity_correction,
        )
        self.automatic_wavelet = wavelet is None
        self.pocs_iterations = int(pocs_iterations)
        self.l2 = float(l2)
        self.gadgets = (Prewhiten(), RampSampling(), EpiPhaseCorrection())
        # The odd/even phase reference is consumed by EpiPhaseCorrection.
        self.reject_flags = ()

    def startup(self, context):
        super().startup(context)
        self.gadget(EpiPhaseCorrection).ramp = self.gadget(RampSampling)
        self.calibrations = {}
        self.slice_calibrations = []
        self.own_maps = {}
        self.references = {}
        self.field = None
        self.calibration_field = False
        self.held = []
        self._skips_logged = set()
        self.hysco = (
            None
            if importlib.util.find_spec(PYHYSCO_PACKAGE) is not None
            else f"PyHySCO is not installed (no package {PYHYSCO_PACKAGE})"
        )

    def unit_image(self, context, data):
        data = compressed(context, data, self.virtual_coils)
        buffer = data.data
        if buffer is None:
            if data.ref is not None:
                polarity = int(data.counters.get("set", 0))
                self.calibrations.setdefault(polarity, []).append(data)
            return None
        if self.calibrations:
            self._calibrate(context)
        image = self.image(averaged(buffer), buffer.image_shape, context, data)
        return buffer, data, image

    def image(self, kspace, shape, context, data):
        """Return the complex image of ``(coils, [partitions,] phase encodes, readout)`` k-space, cropped to ``shape`` and Fermi filtered."""
        import torch

        volume = kspace if kspace.ndim == 4 else kspace[:, None]
        volume = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.complex64))
        maps = self._maps(context, data, volume)
        bands = None
        if volume.shape[1] > 1:
            training = band_calibration(self.slice_calibrations, data, volume.shape)
            if training is not None:
                bands = slice_grappa_bands(volume, torch.from_numpy(training))
        if bands is None:
            image = self._solved(volume, maps, context)
        else:
            image = torch.cat(
                [
                    self._solved(bands[:, b : b + 1], maps[:, b : b + 1], context)
                    for b in range(bands.shape[1])
                ]
            )
        image = image.numpy()
        if kspace.ndim == 3:
            image = image[0]
        return self._filtered(np.array(center_crop(image, shape[-image.ndim :])))

    def _calibrate(self, context):
        """Estimate the maps of each held calibration slice on its own, keep its k-space for split-slice GRAPPA, and fit the stack's field to its polarities."""
        polarities = {}
        for polarity, units in self.calibrations.items():
            for unit in units:
                coil_maps(context, unit, estimate=epi_nlinv_estimate)
                kspace = _zero_filled(unit.ref)
                polarities.setdefault(int(unit.counters.get("slice", 0)), {})[
                    polarity
                ] = (unit, kspace)
                if int(np.prod(kspace.shape[1:-2])) == 1:
                    self.slice_calibrations.append(
                        (
                            polarity,
                            _along_slice(unit.ref.reference),
                            kspace.reshape(kspace.shape[0], *kspace.shape[-2:]),
                        )
                    )
        self.calibrations = {}
        pairs = [
            (location, by_polarity[0], by_polarity[1])
            for location, by_polarity in polarities.items()
            if {0, 1} <= set(by_polarity)
        ]
        if pairs and self.hysco is None:
            self._estimate_field(
                [
                    (
                        location,
                        forward.ref.space,
                        forward.ref.reference,
                        self._calibration_image(forward.ref, forward_kspace),
                        self._calibration_image(reverse.ref, reverse_kspace),
                    )
                    for location, (forward, forward_kspace), (
                        reverse,
                        reverse_kspace,
                    ) in pairs
                ],
                "its FLEET calibration's two phase-encode polarities",
            )
            self.calibration_field = self.field is not None

    def _calibration_image(self, ref, kspace):
        """Return the root-sum-of-squares image of zero-filled calibration k-space on the grid of the images, Fermi filtered."""
        import bartorch
        import torch

        axes = tuple(range(-1, -kspace.ndim, -1))
        coils = bartorch.fft(
            torch.from_numpy(np.ascontiguousarray(kspace, dtype=np.complex64)),
            axes,
            inverse=True,
            unitary=True,
        ).reshape(kspace.shape)
        image = coils.abs().pow(2).sum(dim=0).sqrt().numpy()
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        image = center_crop(image, ref.image_shape[-image.ndim :])
        return np.abs(self._filtered(np.array(image)))

    def _wavelet(self, volume):
        """Return the wavelet ``lambda`` of ``(coils, z, y, x)`` k-space."""
        if not self.automatic_wavelet:
            return self.wavelet
        return EPI_WAVELET_SLAB if volume.shape[1] > 1 else EPI_WAVELET_SLICE

    def _solved(self, volume, maps, context):
        """Return the image of ``(coils, z, y, x)`` k-space by the solve its sampling calls for, partial Fourier completed."""
        sampled = volume.abs().sum(dim=0) > 0
        if _fills_acquired_range(sampled.numpy()):
            image = sense_combined(volume, maps)
        elif self._wavelet(volume) > 0:
            phase = smooth_phase(
                cg_sense(volume, maps, sampled, self.iterations, context.device)
            )
            image = phase * pics_image(
                volume,
                maps * phase,
                sampled,
                self._wavelet(volume),
                self.iterations,
                context.device,
            )
        else:
            image = cg_sense(
                volume, maps, sampled, self.iterations, context.device, l2=self.l2
            )
        return pocs(image, sampled, self.pocs_iterations)

    def _maps(self, context, data, volume):
        """Return the unit's coil maps shaped as ``volume``: its slice's, its bands' slices', else fitted once to it and kept."""
        try:
            return coil_maps(context, data, estimate=epi_nlinv_estimate).reshape(
                volume.shape
            )
        except MissingCalibration as missing:
            bands = band_maps(context, data, volume.shape)
            if bands is not None:
                return bands
            key = (int(data.counters.get("slice", 0)), int(data.counters.get("set", 0)))
            if key not in self.own_maps:
                self._self_calibrated(context, data, missing)
                self.own_maps[key] = epi_nlinv_estimate(volume)
            return self.own_maps[key]

    def recon(self, context: ReconContext, branch: str, data: ReconData):
        del branch
        unit = self.unit_image(context, data)
        if unit is None:
            return None
        buffer, data, image = unit
        location = int(data.counters.get("slice", 0))
        if int(data.counters.get("set", 0)) == 1:
            # The reference only corrects the images (_corrected); it is not
            # an image of the series.
            self.references[location] = (np.abs(image), buffer.reference)
            if not self.calibration_field:
                self.field = None
            return None
        image = np.abs(image)
        if self.hysco is not None or not self.references or self.field is not None:
            return self._results(context, data, self._corrected(image, location))
        self.held.append((data, image))
        if {int(d.counters.get("slice", 0)) for d, _ in self.held} < set(
            self.references
        ):
            return None
        held, self.held = self.held, []
        pairs = []
        for unit_data, unit_image in held:
            location = int(unit_data.counters.get("slice", 0))
            reference, acquisition = self.references[location]
            pairs.append(
                (location, unit_data.data.space, acquisition, unit_image, reference)
            )
        self._estimate_field(pairs, "its phase-encode-reversed reference")
        results = []
        for unit_data, unit_image in held:
            location = int(unit_data.counters.get("slice", 0))
            results += self._results(
                context, unit_data, self._corrected(unit_image, location)
            )
        return results

    def finish(self, context):
        if self.held:
            _log.warning(
                "the stack never completed its first volume; %d images are sent "
                "without susceptibility distortion correction",
                len(self.held),
            )
        return super().finish(context)

    def flush(self, context):
        emitted = super().flush(context)
        held, self.held = self.held, []
        for data, image in held:
            emitted.append((data, self._results(context, data, image)))
        return emitted

    def _results(self, context, data, image):
        """Return the image's results, its geometry its own unit's whatever unit closed."""
        corrections = self.corrections
        history = [
            "PULSERVER",
            "PYTHON",
            "PICS" if self.automatic_wavelet or self.wavelet > 0 else "SENSE",
        ]
        if corrections is not None:
            image = corrections.unwarped(image, context, data)
            history += corrections.applied()
        results = [
            ReconResult(
                np.abs(image),
                reference=data.data.reference,
                attributes={"ImageProcessingHistory": history},
            )
        ]
        if corrections is not None:
            results = corrections.intensity_corrected(results, data)
        return results

    def _estimate_field(self, pairs, against) -> None:
        """Estimate the stack's displacement field from pairs of images of opposite phase-encode polarity.

        ``pairs`` holds ``(slice, space, acquisition, image, reference)`` per
        slice of the stack, ``reference`` read with the phase encode reversed;
        ``against`` names the reference in the log. Every plane of the pairs,
        a slice, a band of a multiband group or a partition of a 3D volume, is
        placed at its position along the slice direction
        (:func:`plane_positions`); the planes are stacked in that order and
        corrected as one volume, the median spacing of their positions its
        third voxel size, and each slice takes the field of its own planes.
        """
        start = time.perf_counter()
        planes = []
        for location, space, acquisition, image, reference in pairs:
            if not space.recon_fov:
                self.hysco = "the header states no reconstruction field of view"
                return
            images = image if image.ndim == 3 else image[None]
            references = reference if reference.ndim == 3 else reference[None]
            for index, position in enumerate(
                plane_positions(acquisition, space, images.shape[0])
            ):
                planes.append(
                    (position, location, index, images[index], references[index])
                )
        planes.sort(key=lambda plane: plane[0])
        space = pairs[0][1]
        voxel = tuple(
            1e3 * fov / n
            for fov, n in zip(space.recon_fov, space.recon_matrix, strict=False)
        )[-2:]
        image = np.stack([plane[3] for plane in planes])
        reference = np.stack([plane[4] for plane in planes])
        if len(planes) == 1:
            image, reference = image[0], reference[0]
        else:
            spacing = float(np.median(np.diff([plane[0] for plane in planes])))
            voxel = (spacing, *voxel)
        try:
            field = susceptibility_field(image, reference, voxel)
        except ImportError as error:
            self.hysco = f"PyHySCO is not installed ({error})"
            return
        except Exception:
            _log.exception(
                "the susceptibility distortion field could not be estimated; the "
                "images are sent uncorrected"
            )
            self.hysco = "its field could not be estimated"
            return
        field = field if field.ndim == 3 else field[None]
        by_location = {}
        for (_, location, index, _, _), plane in zip(planes, field, strict=True):
            by_location.setdefault(location, {})[index] = plane
        depth = voxel[0] if len(voxel) == 3 else 1.0
        volumes = {pair[0]: pair[3].ndim == 3 for pair in pairs}
        self.field = {
            location: (
                (np.stack([fields[i] for i in sorted(fields)]), (depth, *voxel[-2:]))
                if volumes[location]
                else (fields[0], voxel[-2:])
            )
            for location, fields in by_location.items()
        }
        _log.info(
            "susceptibility displacement field of a stack of %d planes %s "
            "estimated in %.2f s against %s",
            len(planes),
            image.shape,
            time.perf_counter() - start,
            against,
        )

    def _corrected(self, image, location):
        """Return ``image`` corrected by the stack's field, or as it is with the reason logged."""
        if self.field is not None and location in self.field:
            field, voxel = self.field[location]
            if self.corrections is not None:
                self.corrections.ran(
                    "HYSCO",
                    "susceptibility distortion of a %s image corrected by the "
                    "stack's displacement field",
                    image.shape,
                )
            return undistorted(image, field, voxel)
        if self.hysco is not None:
            reason = self.hysco
        elif location not in self.references:
            reason = (
                f"no phase-encode-reversed reference (set 1) of slice {location} "
                "came before it"
            )
        else:
            reason = f"the stack's field holds no slice {location}"
        level = logging.DEBUG if reason in self._skips_logged else logging.WARNING
        self._skips_logged.add(reason)
        _log.log(
            level,
            "slice %d is not corrected for susceptibility distortion: %s",
            location,
            reason,
        )
        return image


def plane_positions(acquisition, space, count) -> list[float]:
    """Return the position (mm) along the slice direction of each of a unit's ``count`` planes.

    The reference acquisition's position is that of plane ``count // 2``, the
    voxel an FFT centres; the planes, the partitions of the unit's
    reconstruction space, are one voxel of that space's field of view apart.
    """
    centre = _along_slice(acquisition)
    if count == 1:
        return [centre]
    depth = 1e3 * float(space.recon_fov[0]) / count
    return [centre + (index - count // 2) * depth for index in range(count)]


#: Wavelet ``lambda`` of an undersampled 3D EPI slab and 2D EPI slice, on
#: data scaled as ``pics`` scales them. A slab's partition and CAIPI
#: unfolding amplifies noise and converges slowly without the prior.
EPI_WAVELET_SLAB = 0.1
EPI_WAVELET_SLICE = 0.03

#: Standard deviation, in lines, of the in-plane Gaussian k-space window that
#: smooths that phase. At 7T the off-resonance phase near the sinuses wraps
#: within a few voxels by a BOLD echo time; a wavelet prior on that complex
#: image turns each wrap into a ring.
PHASE_SMOOTHING_LINES = 8.0

#: Sobolev weight of the coil maps of a 3D EPI slab on their calibration
#: grid. Below BART's, the maps follow the sensitivity across the end
#: partitions of a slab whose signal reaches them and wraps onto each other,
#: where the partition and in-plane aliases are unfolded.
EPI_SOBOLEV = 1.0


def epi_nlinv_estimate(kspace):
    """Return the coil maps of EPI calibration k-space by :func:`~pulserver.recon.handlers.pics.nlinv_estimate`: a slab's at ``EPI_SOBOLEV``, a slice's at BART's weight."""
    slab = kspace.ndim == 4 and kspace.shape[1] > 1
    if not slab or EPI_SOBOLEV is None:
        return nlinv_estimate(kspace)
    return nlinv_estimate(kspace, sobolev=EPI_SOBOLEV)


def band_maps(context, data, shape):
    """Return the coil maps of a multiband group, its bands' slices' maps stacked as its partitions; ``None`` where the stream holds no maps for some band.

    Band ``p`` of a group of ``shape[1]`` bands lies at the position of
    partition ``p`` (:func:`plane_positions`), and takes the maps the stream
    stored for the slice of that position and the unit's set, else of that
    position in set 0, within ``BAND_TOLERANCE_MM``.
    """
    import torch

    stored = []
    for key, maps in context.coil_maps.items():
        position = getattr(maps, "geometry", {}).get("position")
        normal = getattr(maps, "geometry", {}).get("slice_dir")
        if position is None or normal is None:
            continue
        key_set = key[1] if isinstance(key, tuple) else 0
        along = float(np.asarray(position) @ np.asarray(normal))
        stored.append((key_set, along, maps.maps))
    planes = _band_planes(stored, data, shape)
    return None if planes is None else torch.from_numpy(planes)


def band_calibration(stored, data, shape):
    """Return the calibration k-space of a multiband group's bands, ``(bands, coils, y, x)``; ``None`` where some band's slice has none.

    ``stored`` holds ``(set, position along the slice direction in mm,
    (coils, y, x) k-space)`` for each calibrated slice; band ``p`` takes the
    slice at the position of partition ``p``, as :func:`band_maps` does.
    """
    planes = _band_planes(stored, data, shape)
    return None if planes is None else np.ascontiguousarray(planes.swapaxes(0, 1))


def _band_planes(stored, data, shape):
    """Return the entries of ``stored`` at the planes of a multiband unit, stacked as ``(coils, bands, y, x)``; ``None`` where a band has none.

    ``stored`` holds ``(set, position in mm, (coils, *plane) array)``. A
    band takes the entry of the unit's set at its plane's position within
    ``BAND_TOLERANCE_MM``, else that of set 0.
    """
    buffer = data.data
    if buffer is None or len(shape) != 4 or shape[1] < 2:
        return None
    polarity = int(data.counters.get("set", 0))
    planes = []
    for position in plane_positions(buffer.reference, buffer.space, shape[1]):
        found = None
        for key_set in (polarity, 0):
            found = next(
                (
                    values
                    for entry_set, along, values in stored
                    if entry_set == key_set
                    and abs(along - position) <= BAND_TOLERANCE_MM
                ),
                None,
            )
            if found is not None:
                break
        if found is None or found.shape[-2:] != tuple(shape[-2:]):
            return None
        planes.append(found.reshape(found.shape[0], *shape[-2:]))
    return np.ascontiguousarray(np.stack(planes, axis=1))


def slice_grappa_bands(volume, training, size=(3, 5)):
    """Return the bands of multiband k-space separated by split-slice GRAPPA, ``(coils, bands, y, x)``; ``None`` where the calibration cannot train the kernels.

    Each acquired line of ``volume`` holds one partition encode, its
    collapsed bands each weighted by the centred, unitary Fourier transform
    along the partitions of a unit image at that band, the encoding the
    joint solve models. The kernels
    (:func:`bartorch.tools.slice_grappa_kernel`) are trained on the
    calibration lines on the acquired lattice, each band's single-band
    k-space carrying the weight of the line's encode, and applied
    (:func:`bartorch.tools.slice_grappa`) to the collapsed lines, whose
    weights they then remove. Lines not acquired stay zero.

    Parameters
    ----------
    volume : torch.Tensor
        ``(coils, bands, y, x)`` multiband k-space, each acquired line in one
        partition.
    training : torch.Tensor
        ``(bands, coils, y, x)`` single-band calibration k-space of each band's
        slice, zero outside its calibration lines.
    size : tuple of int, default=(3, 5)
        Kernel extent along lines and readout.

    Returns
    -------
    torch.Tensor or None
        Separated k-space, each band's lines unweighted.
    """
    import bartorch
    import torch
    from bartorch import tools

    count = volume.shape[1]
    lines = volume.abs().sum(dim=(0, 3)) > 0
    acquired = torch.nonzero(lines.any(dim=0)).flatten()
    encode = lines[:, acquired].to(torch.int64).argmax(dim=0)
    unit = torch.eye(count, dtype=torch.complex64)
    weights = bartorch.fft(unit, (1,), unitary=True).reshape(count, count)[:, encode]
    calibrated = (training.abs().sum(dim=(0, 1, 3)) > 0)[acquired]
    taught = torch.nonzero(calibrated).flatten()
    if taught.numel() < size[0] or not bool(
        (taught[-1] - taught[0] + 1) == taught.numel()
    ):
        return None
    window = slice(int(taught[0]), int(taught[-1]) + 1)
    kernels = tools.slice_grappa_kernel(
        training[:, :, acquired[window]].to(torch.complex64), weights[:, window], size
    )
    collapsed = volume.sum(dim=1)[:, acquired]
    separated = tools.slice_grappa(collapsed, kernels, size) / weights[:, None, :, None]
    bands = torch.zeros_like(volume)
    bands[:, :, acquired] = separated.transpose(0, 1).to(volume.dtype)
    return bands


#: Distance (mm) within which a stored slice's position is a band's.
BAND_TOLERANCE_MM = 0.01


def _along_slice(acquisition) -> float:
    """Return an acquisition's position along its slice direction (mm)."""
    position = np.asarray(getattr(acquisition, "position", (0, 0, 0)), float)
    normal = np.asarray(getattr(acquisition, "slice_dir", (0, 0, 1)), float)
    return float(position @ normal)


def _fills_acquired_range(sampled: np.ndarray) -> bool:
    """Whether every line between the first and last acquired along each encoded axis was acquired."""
    lines = sampled.any(axis=-1)
    for axis in range(lines.ndim):
        profile = lines.any(axis=tuple(a for a in range(lines.ndim) if a != axis))
        acquired = np.flatnonzero(profile)
        if acquired.size and not profile[acquired[0] : acquired[-1] + 1].all():
            return False
    inside = np.ones(lines.shape, dtype=bool)
    for axis in range(lines.ndim):
        profile = lines.any(axis=tuple(a for a in range(lines.ndim) if a != axis))
        acquired = np.flatnonzero(profile)
        shape = [1] * lines.ndim
        shape[axis] = -1
        window = np.zeros(profile.size, dtype=bool)
        if acquired.size:
            window[acquired[0] : acquired[-1] + 1] = True
        inside = inside & window.reshape(shape)
    return bool(np.array_equal(lines, inside))


def sense_combined(kspace, maps):
    """Return the least-squares SENSE image ``sum_c conj(S_c) F^-1 y_c / sum_c |S_c|^2`` of fully sampled k-space.

    ``kspace`` and ``maps`` are ``(coils, z, y, x)``; the transform is
    centred and unitary over ``(y, x)``, and over ``z`` where it has more than
    one point. Where no coil is sensitive the image is zero.
    """
    import bartorch
    import torch

    axes = (-1, -2, -3) if kspace.shape[1] > 1 else (-1, -2)
    coils = bartorch.fft(kspace, axes, inverse=True, unitary=True).reshape(kspace.shape)
    maps = maps.to(coils.dtype)
    weight = (maps.abs() ** 2).sum(dim=0)
    image = (maps.conj() * coils).sum(dim=0)
    return torch.where(weight > 0, image / weight.clamp_min(1e-12), 0)


def cg_sense(kspace, maps, sampled, iterations, device, *, l2=0.0):
    """Return the image ``(z, y, x)`` minimising ``|P F S x - y|^2 + l2 |x|^2`` by conjugate gradients from zero.

    The encoding is :class:`bartorch.linop.CartesianSense` of ``maps`` over
    the ``sampled`` lines, on ``device``; stopping after ``iterations`` steps
    regularizes the solve. The data are scaled as ``pics`` scales them.
    """
    import bartorch
    import torch
    from bartorch import linop, optim

    grid = tuple(kspace.shape[2:] if kspace.shape[1] == 1 else kspace.shape[1:])
    pattern = sampled.reshape(grid).to(torch.complex64)
    measured = kspace * sampled
    encoding = linop.CartesianSense(
        maps.reshape(maps.shape[0], *grid), grid, pattern=pattern, device=device
    )
    shifted = bartorch.fftmod(measured, axes=(-1, -2, -3), inverse=True)
    scaling = optim.data_scaling(shifted.reshape(kspace.shape[0], *grid), A=encoding)
    scaling = scaling or 1.0
    data = (measured * (1.0 / scaling)).reshape(kspace.shape[0], *grid)
    solve = optim.CG(lambda_=float(l2), maxiter=int(iterations))
    return (scaling * solve(data, encoding)).reshape(kspace.shape[1:])


def smooth_phase(image):
    """Return the unit-magnitude phase of ``image`` low-passed in plane.

    Parameters
    ----------
    image : torch.Tensor
        Complex ``(z, y, x)`` image, the transform centred.

    Returns
    -------
    torch.Tensor
        ``exp(i angle(g * image))``, complex64, ``g`` a Gaussian k-space
        window of ``PHASE_SMOOTHING_LINES`` lines over ``y`` and ``x``.
    """
    import bartorch
    import torch

    image = image.to(torch.complex64)
    ny, nx = image.shape[-2:]
    y = torch.arange(ny, dtype=torch.float32, device=image.device) - ny // 2
    x = torch.arange(nx, dtype=torch.float32, device=image.device) - nx // 2
    window = torch.exp(
        -(y[:, None] ** 2 + x[None, :] ** 2) / (2 * PHASE_SMOOTHING_LINES**2)
    )
    axes = (-1, -2)
    kspace = bartorch.fft(image, axes, unitary=True).reshape(image.shape)
    smooth = bartorch.fft(kspace * window, axes, inverse=True, unitary=True)
    return torch.exp(1j * torch.angle(smooth.reshape(image.shape))).to(torch.complex64)


def pocs(image, sampled, iterations):
    """Return ``image`` completed by POCS over every partial-Fourier axis of ``sampled``; ``image`` where none is.

    The partial-Fourier axes and the acquired range ``R`` along each are those
    :func:`~pulserver.recon.handlers.pics.partial_fourier` finds. The phase
    is that of the image of the samples whose conjugates are also in ``R``;
    each iteration imposes it on the image's magnitude and restores the
    k-space of ``image`` inside ``R``. The image keeps its phase.

    Parameters
    ----------
    image : torch.Tensor
        Complex ``(z, y, x)`` image of the sampled k-space, the transform
        centred.
    sampled : torch.Tensor
        Boolean ``(z, y, x)`` mask of the acquired samples.
    iterations : int
        POCS iterations.
    """
    import bartorch
    import torch

    from .pics import _acquired_range

    mask = sampled.cpu().numpy()
    kept = np.ones(mask.shape, dtype=bool)
    axes = []
    for axis in range(mask.ndim):
        lines = _acquired_range(mask, axis)
        if lines is None:
            continue
        axes.append(axis)
        shape = [1] * mask.ndim
        shape[axis] = mask.shape[axis]
        inside = np.zeros(mask.shape[axis], dtype=bool)
        inside[lines[0] : lines[1] + 1] = True
        kept = kept & inside.reshape(shape)
    if not axes or iterations < 1:
        return image
    every = tuple(range(-1, -mask.ndim - 1, -1))
    mirrored = np.roll(np.flip(kept), [1 - n % 2 for n in mask.shape], axis=every)
    symmetric = torch.from_numpy((kept & mirrored).astype(np.float32)).to(image.device)
    inside = torch.from_numpy(kept).to(image.device)

    from bartorch import optim

    def forward(x):
        return bartorch.fft(x, every, unitary=True).reshape(x.shape)

    def inverse(k):
        return bartorch.fft(k, every, inverse=True, unitary=True).reshape(k.shape)

    measured = forward(image.to(torch.complex64))
    phase = torch.exp(1j * torch.angle(inverse(measured * symmetric)))
    return optim.POCS(
        [
            lambda x: x.abs() * phase,
            lambda x: inverse(torch.where(inside, measured, forward(x))),
        ],
        maxiter=int(iterations),
    )(measured, x0=image.to(torch.complex64))


#: The package PyHySCO installs, which the susceptibility correction imports.
PYHYSCO_PACKAGE = "EPI_MRI"

PLUGIN = EpiRecon()


def susceptibility_field(image, reverse, voxel_size):
    """Return the displacement field of ``image`` against its phase-encode-reversed ``reverse``.

    A slice is ``(phase encode, readout)`` and a volume ``(partition, phase
    encode, readout)``, with ``voxel_size`` in mm along the same axes. The
    field is PyHySCO's (:func:`bartorch.tools.correct_susceptibility`), in mm
    on the cell edges of the phase encode, in the axes of ``image``: one row
    (a volume: one phase-encode line) longer. PyHySCO corrects volumes: a
    volume is corrected as one, its phase encode first, at the smoothness
    weight ``SLAB_FIELD_SMOOTHNESS``, and a slice as three identical slices,
    which its smoothness penalty leaves uncoupled, the middle one's field
    kept, at PyHySCO's own weight.

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
    smoothness = {}
    if image.ndim == 2:
        up = np.repeat(image[:, :, None], 3, axis=2)
        down = np.repeat(reverse[:, :, None], 3, axis=2)
        voxel = (*voxel_size, 1.0)
    else:
        smoothness = {"alpha": SLAB_FIELD_SMOOTHNESS}
        # (partition, phase, readout) to (phase, readout, partition), the
        # outer partitions repeated so the smoothness penalty does not pull
        # the field of the end partitions towards zero.
        edges = ((FIELD_EDGE_PARTITIONS,) * 2, (0, 0), (0, 0))
        up = np.pad(image, edges, mode="edge").transpose(1, 2, 0)
        down = np.pad(reverse, edges, mode="edge").transpose(1, 2, 0)
        voxel = (voxel_size[1], voxel_size[2], voxel_size[0])
    corrected = tools.correct_susceptibility(
        torch.from_numpy(np.ascontiguousarray(up)),
        torch.from_numpy(np.ascontiguousarray(down * scale)),
        voxel_size=voxel,
        phase_encoding_axis=0,
        **smoothness,
    )
    field = corrected.field_map.detach().cpu().numpy()
    if image.ndim == 2:
        return field[:, :, 1]
    inner = slice(FIELD_EDGE_PARTITIONS, field.shape[2] - FIELD_EDGE_PARTITIONS)
    return field[:, :, inner].transpose(2, 0, 1)


#: Partitions repeated beyond each end of a volume whose field is estimated.
FIELD_EDGE_PARTITIONS = 2

#: PyHySCO smoothness weight of a volume's field. At PyHySCO's own weight a
#: slab whose blip-up and blip-down images differ only by their T2* blur is
#: fitted a field that steps by over half a voxel between neighbouring lines,
#: and the Jacobian of that step darkens a line of the corrected image.
SLAB_FIELD_SMOOTHNESS = 3000.0


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
