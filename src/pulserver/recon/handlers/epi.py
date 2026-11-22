"""EPI reconstruction with bartorch: ramp sampling and odd/even phase corrected per readout, then ``bart pics``."""

from __future__ import annotations

from typing import NamedTuple

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
from ..gadgets import (
    AsymmetricEcho,
    CompressChannels,
    Prewhiten,
    RemoveReadoutOversampling,
)
from ..plugin import Gadget, ReconContext, ReconResult
from .pics import (
    CG_MIN_L2,
    VIRTUAL_COILS,
    PicsRecon,
    averaged,
    center_crop,
    nlinv_estimate,
    partial_fourier_axes,
    pics_image,
    smooth_phase,
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

    Each readout is whitened (:class:`~pulserver.recon.Prewhiten`), projected
    onto ``virtual_coils`` channels (:class:`~pulserver.recon.CompressChannels`,
    one basis for the stream from the calibration of every slice, so the
    bands of a multiband unit and the slices its kernels are trained on share
    their virtual channels, and no later step holds every receive channel),
    ramp-resampled along its k-space trajectory (:class:`RampSampling`) and
    has its odd/even phase removed (:class:`EpiPhaseCorrection`, from the
    blip-free reference train ``ACQ_IS_PHASECORR_DATA`` the sequence plays in
    a dummy shot, else from the navigator) on arrival, so the k-space is
    Cartesian, forward-ordered and at the reconstruction matrix when its slice
    closes.

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
    maps carrying the smooth phase (:func:`~pulserver.recon.handlers.pics.smooth_phase`) of its
    ``iterations``-step conjugate-gradient image, so the prior acts on a
    near-real image, started from that image, or
    with ``wavelet`` zero the solution of ``min |P F S x - y|^2`` by
    ``iterations`` conjugate-gradient steps (:class:`bartorch.optim.CG`). A
    partial-Fourier unit is solved for a real image times a known phase
    (``pics_image`` with ``phase``), so the k-space it leaves unsampled is the
    conjugate of what it samples. The phase is that of the full-k phase
    reference volume of its slice (``SET`` 4, played at the echo time of the
    scan, reconstructed and kept but sent as no image) times the smooth phase
    of the unit's own image against it, which follows the drift of the field
    from volume to volume. Without a reference it is the smooth phase of the
    unit's image, the phase of a spin echo; where that image keeps more than
    ``SMOOTH_PHASE_RESIDUAL`` of its energy in quadrature to it, as a
    gradient echo at 7T does near air-tissue interfaces, the unit is left
    zero-filled and a warning says why.

    A multiband unit whose bands each lie on a calibrated slice has its bands
    separated by split-slice GRAPPA (:func:`slice_grappa_bands`), with kernels
    trained on those slices' calibration k-space, and each band is then
    solved as a slice of its own, with its slice's maps. A multiband unit
    without that calibration is solved as one volume of its bands.

    When PyHySCO is installed (bartorch's ``pyhysco`` extra, GPL-3.0), one
    susceptibility displacement field is estimated for the whole stack
    (:func:`bartorch.tools.correct_susceptibility`) from its spin-echo field
    pair (``SET`` 2 forward, 3 reversed), reconstructed and kept but sent as
    no image of the series: refocused, its two volumes are displaced in
    opposite directions by the field and differ in nothing else. A pair of
    gradient echoes, such as the FLEET calibration's two polarities or the
    reversed reference volume (``SET`` 1), also differs in the intensity the
    field gradient moves along the phase encode at 7T, and gives no field;
    the reversed reference volume is sent as no image. The field is
    estimated once, on the planes of the pair at their positions along the
    slice direction (:class:`PlaneField`), and applied to every image of
    every slice, a plane between the pair's taking the linear interpolation
    of the two, so a 3D slab takes the field of a 2D pair over it. It is
    kept for the exam: a later series without a pair of its own takes it
    where its slices are parallel to the pair's and it shares the pair's
    in-plane matrix, field of view, dwell time and phase-encode line step,
    and so its distortion
    (:func:`bartorch.tools.apply_susceptibility`, as FSL's ``applytopup
    --method=jac``), history ``HYSCO``. The correction is logged as the other
    corrections are, and an image left uncorrected, for want of PyHySCO or of
    the field pair, is logged with the reason at ``WARNING``. Each image is
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
        self.l2 = float(l2)
        self.phase_references = {}
        self._skips_logged = set()
        compression = (
            ()
            if self.virtual_coils is None
            else (CompressChannels(self.virtual_coils),)
        )
        self.gadgets = (
            Prewhiten(),
            *compression,
            RampSampling(),
            EpiPhaseCorrection(),
        )
        # The odd/even phase reference is consumed by EpiPhaseCorrection.
        self.reject_flags = ()

    def startup(self, context):
        super().startup(context)
        self.gadget(EpiPhaseCorrection).ramp = self.gadget(RampSampling)
        self.calibrations = {}
        self.slice_calibrations = []
        self.own_maps = {}
        self.field = None
        self.field_pair = {}
        self.phase_references = {}
        self._skips_logged = set()
        self.hysco = (
            None
            if importlib.util.find_spec(PYHYSCO_PACKAGE) is not None
            else f"PyHySCO is not installed (no package {PYHYSCO_PACKAGE})"
        )

    def unit_image(self, context, data):
        if data.data is None:
            if data.ref is not None:
                polarity = int(data.counters.get("set", 0))
                self.calibrations.setdefault(polarity, []).append(data)
            return None
        buffer = data.data
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
        location = int(data.counters.get("slice", 0))
        reference = self.phase_references.get(location)
        bands = None
        if volume.shape[1] > 1:
            training = band_calibration(self.slice_calibrations, data, volume.shape)
            if training is not None:
                bands = slice_grappa_bands(volume, torch.from_numpy(training))
        if bands is None:
            image = self._solved(volume, maps, context, reference)
        else:
            image = torch.cat(
                [
                    self._solved(
                        bands[:, b : b + 1],
                        maps[:, b : b + 1],
                        context,
                        None if reference is None else reference[b : b + 1],
                    )
                    for b in range(bands.shape[1])
                ]
            )
        if int(data.counters.get("set", 0)) == FULL_K_REFERENCE_SET:
            self.phase_references[location] = image
        image = image.numpy()
        if kspace.ndim == 3:
            image = image[0]
        return self._filtered(np.array(center_crop(image, shape[-image.ndim :])))

    def _calibrate(self, context):
        """Estimate the maps of each held calibration slice on its own, and keep its k-space for split-slice GRAPPA."""
        for polarity, units in self.calibrations.items():
            for unit in units:
                coil_maps(context, unit, estimate=epi_nlinv_estimate)
                kspace = _zero_filled(unit.ref)
                if int(np.prod(kspace.shape[1:-2])) == 1:
                    self.slice_calibrations.append(
                        (
                            polarity,
                            _along_slice(unit.ref.reference),
                            kspace.reshape(kspace.shape[0], *kspace.shape[-2:]),
                        )
                    )
        self.calibrations = {}

    def _wavelet(self, volume):
        """Return the wavelet ``lambda`` of ``(coils, z, y, x)`` k-space."""
        if not self.automatic_wavelet:
            return self.wavelet
        return EPI_WAVELET_SLAB if volume.shape[1] > 1 else EPI_WAVELET_SLICE

    def _solved(self, volume, maps, context, reference=None):
        """Return the image of ``(coils, z, y, x)`` k-space by the solve its sampling calls for, a partial-Fourier unit solved for a real image times its phase."""
        sampled = volume.abs().sum(dim=0) > 0
        partial = partial_fourier_axes(sampled.numpy())
        if _fills_acquired_range(sampled.numpy()) and not partial:
            return sense_combined(volume, maps)
        wavelet = self._wavelet(volume)
        start = cg_sense(
            volume,
            maps,
            sampled,
            self.iterations,
            context.device,
            l2=0.0 if wavelet > 0 else self.l2,
        )
        if partial:
            phase = self._phase(start, reference, sampled)
            if phase is not None:
                return pics_image(
                    volume,
                    maps,
                    sampled,
                    wavelet,
                    self.iterations,
                    context.device,
                    x0=(start * phase.conj()).real.to(start.dtype),
                    phase=phase,
                    l2=self.l2,
                )
        if wavelet <= 0:
            return start
        phase = smooth_phase(start)
        return phase * pics_image(
            volume,
            maps * phase,
            sampled,
            wavelet,
            self.iterations,
            context.device,
            x0=start * phase.conj(),
        )

    def _phase(self, start, reference, sampled):
        """Return the phase of a partial-Fourier unit whose zero-filled image is ``start``; ``None`` where it has none to take.

        The drift against the reference is the smooth phase of ``start``
        against the reference cut to the same range of k-space, so the two
        share the blur of the lines the unit leaves out.
        """
        import torch

        if reference is not None:
            unit = torch.exp(1j * torch.angle(reference)).to(torch.complex64)
            cut = _within_acquired_range(reference.to(torch.complex64), sampled)
            return unit * smooth_phase(start * cut.conj())
        phase = smooth_phase(start)
        energy = float((start.abs() ** 2).sum())
        quadrature = float(((start * phase.conj()).imag ** 2).sum())
        if energy > 0 and quadrature <= SMOOTH_PHASE_RESIDUAL * energy:
            return phase
        if "phase" not in self._skips_logged:
            self._skips_logged.add("phase")
            _log.warning(
                "a partial-Fourier image keeps %.3f of its energy in quadrature to "
                "its smooth phase and the scan played no full-k phase reference "
                "(set %d): its unsampled k-space is left empty",
                quadrature / max(energy, 1e-30),
                FULL_K_REFERENCE_SET,
            )
        return None

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
        polarity = int(data.counters.get("set", 0))
        if polarity in FIELD_PAIR_SETS:
            self.field_pair.setdefault(location, {})[polarity] = (
                np.abs(image),
                buffer.space,
                buffer.reference,
                _train(buffer),
            )
            return None
        if self.field_pair:
            self._field_from_pair(context)
        if polarity in (1, FULL_K_REFERENCE_SET):
            # The reversed and the full-k reference volumes are no images of
            # the series.
            return None
        return self._results(
            context, data, self._corrected(context, np.abs(image), location, buffer)
        )

    def _field_from_pair(self, context) -> None:
        """Estimate the stack's field from its spin-echo field pair and keep it for the exam."""
        forward_set, reversed_set = FIELD_PAIR_SETS
        pairs = []
        train = None
        for location, by_set in sorted(self.field_pair.items()):
            if {forward_set, reversed_set} <= set(by_set):
                image, space, acquisition, train = by_set[forward_set]
                pairs.append(
                    (location, space, acquisition, image, by_set[reversed_set][0])
                )
        self.field_pair = {}
        if not pairs or self.hysco is not None:
            return
        self._estimate_field(pairs, "its spin-echo field pair")
        if self.field is not None:
            self.field = self.field._replace(train=train)
            context.exam.set(SUSCEPTIBILITY_FIELD, self.field)

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
        self.field = PlaneField(
            positions=np.array([plane[0] for plane in planes]),
            fields=np.asarray(field),
            voxel=tuple(voxel[-2:]),
            slice_dir=tuple(
                float(v) for v in getattr(pairs[0][2], "slice_dir", (0, 0, 1))
            ),
            train=None,
        )
        _log.info(
            "susceptibility displacement field of a stack of %d planes %s "
            "estimated in %.2f s against %s",
            len(planes),
            image.shape,
            time.perf_counter() - start,
            against,
        )

    def _corrected(self, context, image, location, buffer):
        """Return ``image`` corrected by the stack's field, the exam's where the stack has none, or as it is with the reason logged."""
        field = self.field
        if field is None and self.hysco is None:
            field = context.exam.get(SUSCEPTIBILITY_FIELD)
        if field is not None and self.hysco is None:
            planes = image if image.ndim == 3 else image[None]
            displacement, reason = field.at(
                plane_positions(buffer.reference, buffer.space, planes.shape[0]),
                _train(buffer),
                getattr(buffer.reference, "slice_dir", (0, 0, 1)),
                planes.shape[-2:],
            )
            if displacement is not None:
                if self.corrections is not None:
                    self.corrections.ran(
                        "HYSCO",
                        "susceptibility distortion of a %s image corrected by the "
                        "stack's displacement field",
                        image.shape,
                    )
                if image.ndim == 3:
                    depth = 1e3 * float(buffer.space.recon_fov[0]) / image.shape[0]
                    return undistorted(image, displacement, (depth, *field.voxel))
                return undistorted(image, displacement[0], field.voxel)
        elif self.hysco is not None:
            reason = self.hysco
        else:
            reason = "no spin-echo field pair (sets 2 and 3) came before it"
        level = logging.DEBUG if reason in self._skips_logged else logging.WARNING
        self._skips_logged.add(reason)
        _log.log(
            level,
            "slice %d is not corrected for susceptibility distortion: %s",
            location,
            reason,
        )
        return image


class PlaneField(NamedTuple):
    """A susceptibility displacement field on the planes of a stack, at their positions along its slice direction.

    Attributes
    ----------
    positions : numpy.ndarray
        Position (mm) of each plane along ``slice_dir``, ascending.
    fields : numpy.ndarray
        Displacement (mm) along the phase encode of each plane, as
        :func:`susceptibility_field` gives it.
    voxel : tuple of float
        In-plane voxel size (mm), phase encode then readout.
    slice_dir : tuple of float
        Slice direction of the stack.
    train : tuple
        :func:`_train` of the field pair, which a series has to share to
        take the field.
    """

    positions: np.ndarray
    fields: np.ndarray
    voxel: tuple
    slice_dir: tuple
    train: tuple | None

    def at(self, positions, train, slice_dir, shape):
        """Return the displacement at the planes ``positions`` (mm) along ``slice_dir``, linear between this field's planes, and ``None`` with the reason where it holds none.

        A plane takes the nearest plane of the field within half the field's
        spacing beyond its end planes.
        """
        if self.train is not None and train != self.train:
            return (
                None,
                "its phase-encode sampling or geometry differs from the field pair's",
            )
        if abs(abs(float(np.dot(slice_dir, self.slice_dir))) - 1.0) > 1e-3:
            return None, "its slices are not parallel to the field pair's"
        if self.fields.shape[-1] != shape[-1]:
            return None, "its matrix differs from the field pair's"
        positions = np.asarray(positions, float)
        if len(self.positions) == 1:
            spacing = np.inf
        else:
            spacing = float(np.median(np.diff(self.positions)))
        margin = 0.5 * spacing if np.isfinite(spacing) else 0.5
        low, high = self.positions[0] - margin, self.positions[-1] + margin
        if np.any(positions < low - 1e-6) or np.any(positions > high + 1e-6):
            return None, "it lies outside the field pair's stack"
        if len(self.positions) == 1:
            return np.repeat(self.fields[:1], len(positions), axis=0), None
        upper = np.clip(
            np.searchsorted(self.positions, positions), 1, len(self.positions) - 1
        )
        lower = upper - 1
        weight = (positions - self.positions[lower]) / (
            self.positions[upper] - self.positions[lower]
        )
        weight = np.clip(weight, 0.0, 1.0)[:, None, None]
        return (1 - weight) * self.fields[lower] + weight * self.fields[upper], None


def _train(buffer) -> tuple:
    """Return what a series shares with a field pair whose field it takes: in-plane matrix and field of view, dwell time and phase-encode line step."""
    profile = buffer.mask.reshape(-1, *buffer.mask.shape[-2:]).any(axis=(0, 2))
    lines = np.flatnonzero(profile)
    step = int(np.diff(lines).min()) if lines.size > 1 else 1
    space = buffer.space
    return (
        tuple(int(n) for n in space.recon_matrix[-2:]),
        tuple(round(float(f), 6) for f in space.recon_fov[-2:]),
        None if buffer.sample_time is None else round(float(buffer.sample_time), 9),
        step,
    )


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

#: Sobolev weight of the EPI coil maps on their calibration grid. Below
#: BART's, the maps follow the steep sensitivity next to the coil elements:
#: along the scalp of a close-fitting array, where the scalp is far brighter
#: than the brain and smoother maps fold it into the brain at R=3, and across
#: the end partitions of a slab whose signal reaches them and wraps onto each
#: other.
EPI_SOBOLEV = 1.0


def epi_nlinv_estimate(kspace):
    """Return the coil maps of EPI calibration k-space by :func:`~pulserver.recon.handlers.pics.nlinv_estimate` at ``EPI_SOBOLEV``."""
    if EPI_SOBOLEV is None:
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


def _within_acquired_range(image, sampled):
    """Return ``image``, ``(z, y, x)``, with its k-space outside the range of lines ``sampled`` acquires along each axis removed."""
    import bartorch
    import torch

    from .pics import _acquired_range

    mask = sampled.cpu().numpy()
    kept = np.ones(mask.shape, dtype=bool)
    for axis in range(mask.ndim):
        lines = _acquired_range(mask, axis)
        if lines is None:
            continue
        inside = np.zeros(mask.shape[axis], dtype=bool)
        inside[lines[0] : lines[1] + 1] = True
        shape = [1] * mask.ndim
        shape[axis] = mask.shape[axis]
        kept = kept & inside.reshape(shape)
    axes = tuple(range(-1, -mask.ndim - 1, -1))
    kspace = bartorch.fft(image, axes, unitary=True).reshape(image.shape)
    kspace = kspace * torch.from_numpy(kept).to(kspace.device)
    return bartorch.fft(kspace, axes, inverse=True, unitary=True).reshape(image.shape)


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
    """Return the image ``(z, y, x)`` minimising ``|P F S x - y|^2 + l2 |x|^2``, ``l2`` at least ``CG_MIN_L2``, by conjugate gradients from zero.

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
    solve = optim.CG(lambda_=max(float(l2), CG_MIN_L2), maxiter=int(iterations))
    return (scaling * solve(data, encoding)).reshape(kspace.shape[1:])


def susceptibility_field(image, reverse, voxel_size):
    """Return the displacement field of ``image`` against its phase-encode-reversed ``reverse``.

    A slice is ``(phase encode, readout)`` and a volume ``(partition, phase
    encode, readout)``, with ``voxel_size`` in mm along the same axes. The
    field is PyHySCO's (:func:`bartorch.tools.correct_susceptibility`), in mm
    on the cell edges of the phase encode, in the axes of ``image``: one row
    (a volume: one phase-encode line) longer. A volume is corrected as one,
    its phase encode first, at the smoothness weight
    ``SLAB_FIELD_SMOOTHNESS``, and a slice in its plane, at PyHySCO's own
    weight.

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
        up, down, voxel = image, reverse, tuple(voxel_size)
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
        return field
    inner = slice(FIELD_EDGE_PARTITIONS, field.shape[2] - FIELD_EDGE_PARTITIONS)
    return field[:, :, inner].transpose(2, 0, 1)


#: Partitions repeated beyond each end of a volume whose field is estimated.
FIELD_EDGE_PARTITIONS = 2

#: PyHySCO smoothness weight of a volume's field, on the intensity scale of
#: :func:`bartorch.tools.correct_susceptibility`. A 7T spin-echo pair of a
#: 2 mm stack is corrected best near this weight; ten times stiffer, the field
#: recovers under half of the displacement near the sinuses.
SLAB_FIELD_SMOOTHNESS = 300.0


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


#: ``SET`` of the forward and of the reversed volume of a spin-echo field pair.
FIELD_PAIR_SETS = (2, 3)

#: Exam key of the last spin-echo field pair's :class:`PlaneField`.
SUSCEPTIBILITY_FIELD = "susceptibility_field"

#: ``SET`` of the full-k phase reference volume.
FULL_K_REFERENCE_SET = 4

#: Fraction of a partial-Fourier image's energy in quadrature to its smooth
#: phase above which that phase does not describe the image. A spin echo
#: keeps about 0.001 there, a gradient echo at 7T near air-tissue interfaces
#: 0.015 or more.
SMOOTH_PHASE_RESIDUAL = 0.01

#: The package PyHySCO installs, which the susceptibility correction imports.
PYHYSCO_PACKAGE = "EPI_MRI"

PLUGIN = EpiRecon()
