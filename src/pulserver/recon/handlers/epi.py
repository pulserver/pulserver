"""EPI reconstruction with bartorch: ramp sampling and odd/even phase corrected per readout, then ``bart pics``."""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["PLUGIN", "EpiPhaseCorrection", "EpiRecon", "RampSampling"]

import dataclasses
import importlib.util
import logging
import math
import time

import numpy as np

from ...mrd._header import EncodingSpace
from ...mrd._metadata import (
    ACCELERATION_PER_SOLVE,
    acquisition_label,
    has_acquisition_flag,
    user_parameter,
)
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

    In a 3D encoding space, each excitation of a slab is read over a few
    partitions only, and a shot of a segmented slab is a run of excitations
    through every partition, so the phase a breath or a field drift adds
    changes from excitation to excitation within a shot. Each excitation's
    phase, a constant and a linear term across the readout, is fitted from
    its navigator: the mean of its forward navigator readouts, in hybrid
    space, against the first such mean of its navigator's encoding space and
    slice (:func:`bartorch.tools.estimate_epi_phase` of
    ``[first, this, first]``), and removed from every imaging readout of the
    excitation in a 3D encoding space, forward and reversed.
    ``excitation_phase=False`` leaves it.

    A breath changes the field, and the phase a field offset adds grows with
    the time since the excitation, so the fitted phase is scaled from the
    navigator's echo time to the train's: by the ``TrainEchoTime`` over the
    ``NavigatorEchoTime`` the header states (in s, as user parameters), the
    constant taken within half a turn of zero first. A header without them
    has the phase removed as the navigator reads it, and the log says so.

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

    Parameters
    ----------
    excitation_phase : bool, default=True
        Remove each excitation's navigator phase from the readouts of a 3D
        encoding space.

    Attributes
    ----------
    ramp : RampSampling or None
        The resampling the reference and navigator readouts go through, and
        whose delay is set.
    """

    ramp = None

    def __init__(self, excitation_phase: bool = True) -> None:
        super().__init__()
        self.excitation_phase = bool(excitation_phase)

    def startup(self, context):
        super().startup(context)
        encodings = getattr(getattr(context, "header", None), "encoding", None) or ()
        self.slabs = {
            index
            for index in range(len(encodings))
            if len(EncodingSpace.from_header(context.header, index).recon_matrix) == 3
        }
        self.navigator = []
        self.navigator_profiles = {}
        stated = _navigator_echo_scale(getattr(context, "header", None))
        self.echo_scale = 1.0 if stated is None else stated
        self.echo_scale_stated = stated is not None
        self.excitation = None
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
                self.excitation = self._excitation_phase(acquisition)
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
        space = int(getattr(acquisition, "encoding_space_ref", 0) or 0)
        if self.excitation is not None and space in self.slabs:
            if reverse:
                phase = self.excitation if phase is None else phase + self.excitation
            else:
                # A forward line takes the phase correct_lines gives a
                # reversed one, flipped back into its own order first.
                line = (torch.flip(line[0], [-1]), True)
                phase = self.excitation
        return tools.correct_lines([line], phase)[0].numpy()

    def _excitation_phase(self, acquisition):
        """Return the coefficients removing the phase of the excitation whose navigator was just read, against the first of its navigator's encoding space and slice; ``None`` for the first, or where no encoding space is 3D."""
        if not self.excitation_phase or not self.slabs:
            return None
        from bartorch import tools

        lines = self._lines(self.navigator)
        forward = [
            line
            for line, (_, _, reverse) in zip(lines, self.navigator, strict=True)
            if not reverse
        ]
        profile = sum(forward) / len(forward)
        location = (
            int(getattr(acquisition, "encoding_space_ref", 0) or 0),
            int(acquisition_label(acquisition, "slice", 0) or 0),
        )
        first = self.navigator_profiles.setdefault(location, profile)
        if first is profile:
            _log.info(
                "each excitation's phase is fitted to its navigator against the "
                "first of encoding space %d, slice %d, scaled by %.3f to the "
                "train's echo time%s, and removed from its readouts in a 3D "
                "encoding space",
                *location,
                self.echo_scale,
                ""
                if self.echo_scale_stated
                else " (the header states no NavigatorEchoTime and TrainEchoTime)",
            )
            return None
        phase = tools.estimate_epi_phase([first, profile, first])
        phase[0] = math.remainder(float(phase[0]), 2.0 * math.pi)
        return phase * self.echo_scale


def _navigator_echo_scale(header) -> float | None:
    """Return the header's ``TrainEchoTime`` over its ``NavigatorEchoTime``; ``None`` where either is missing or not positive."""
    if header is None:
        return None
    try:
        train = float(user_parameter(header, "TrainEchoTime", 0.0))
        navigator = float(user_parameter(header, "NavigatorEchoTime", 0.0))
    except (TypeError, ValueError):
        return None
    if train <= 0.0 or navigator <= 0.0:
        return None
    return train / navigator


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
    bands of a multiband unit share their virtual channels with the slices
    whose maps unfold them, and no later step holds every receive channel),
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

    A 3D slab is solved as one, the phase of each of its excitations removed
    from its navigator as its readouts arrive (:class:`EpiPhaseCorrection`). A
    slice or multiband unit whose lines come from several segments
    (``SEG``), the interleaved shots of a segmented scan, is solved with the phase each shot adds to the
    image, as MUSE solves it (Chen et al., NeuroImage 2013,
    doi:10.1016/j.neuroimage.2013.01.076): breathing and motion change the
    field and the phase of the image from shot to shot, and lines encoded as
    if they shared one phase alias each other. Each shot's phase relative to
    the first shot's is a sum of Legendre polynomials in plane of degree
    ``SHOT_PHASE_ORDER``, per partition, so per band of a multiband unit
    (:func:`bartorch.tools.shot_phase_basis`), fitted by Gauss-Newton against an image of the unit
    (:func:`fit_shot_phases`). The images tried are the full-k phase reference
    of its slice where the stream holds one of the unit's shape; for a unit
    read with the phase encode forward, its field pair's forward planes, read
    with the scan's train at its echo time, so the full-k reference, itself
    read in several shots before the scan, is fitted against the pair; and for
    at most ``SHOT_PHASE_SEARCH_SHOTS`` shots, the unit solved with the
    constant phase per shot, out of ``SHOT_PHASE_CONSTANTS``, that leaves the
    least data residual, which is how the field pair itself is fitted. A
    shot's own lines are not unfolded on their own: a multiband shot of a
    segmented scan reads one partition of the blipped-CAIPI lattice, its bands
    superimposed. From each image the fit and the joint solve, which encodes
    each shot's lines from the image times its phase
    (:func:`bartorch.tools.shot_encoding`), alternate until
    the data residual settles (``SHOT_PHASE_TOLERANCE``, at most
    ``SHOT_PHASE_ITERATIONS`` times); every template is tried, the constant
    search only where no template lowers the residual by more than
    ``SHOT_PHASE_CLEAR_GAIN`` against the solve as one, and the phases that
    leave the least residual are kept if they lower it at all, and the
    partial-Fourier or wavelet solve takes the phases into its encoding. A
    unit that no image improves is solved as one, and the log says so; one
    whose solve as one leaves a residual of at most ``SHOT_PHASE_FLOOR`` is
    solved as one at once. The fits, the joint solves and their residuals run
    on the reconstruction's device. A segmented single-band full-k reference
    is unfolded by its slice's calibration maps, and its own maps are drawn
    from its coil images with the shot phases removed
    (:func:`shot_corrected`).

    A multiband unit is solved as one volume of its bands, a partition each,
    so its in-plane and band aliases, which the blipped-CAIPI shift spreads
    apart, are unfolded together by every virtual channel; its maps are its
    bands' slices' maps stacked (:func:`band_maps`). Separating the bands
    first by a split-slice GRAPPA kernel on the acquired lines leaves a few
    local weights to cancel as many aliases as the joint solve unfolds pixel
    by pixel, and beyond twofold in plane it fails with six virtual channels.
    A multiband unit whose bands each lie on a calibrated slice is solved at
    ``EPI_WAVELET_MULTIBAND``, or at ``EPI_WAVELET_SLICE`` where it is read in
    several shots and its in-plane line step times its bands, per shot, is at
    most ``EPI_WAVELET_SEGMENTED_ACCELERATION`` (:meth:`_wavelet`), and Fermi
    filtered, unwarped and divided by its receive field band by band, each in its own plane, as a slice is: the
    bands lie several slices apart, and filtered or resampled across the
    stack a band takes in the others. One without that calibration is solved
    at ``EPI_WAVELET_SLAB`` and filtered and corrected as one volume of its
    bands.

    When PyHySCO is installed (bartorch's ``pyhysco`` extra, GPL-3.0), one
    susceptibility displacement field is estimated for the whole stack
    (:func:`bartorch.tools.correct_susceptibility`) from its field pair
    (``SET`` 2 forward, 3 reversed), reconstructed and kept but sent as no
    image of the series. The pair is an echo of the scan's kind: a
    gradient-echo pair at the scan's echo time loses and piles up signal
    where the scan does, so the field is fitted to the intensity the scan
    has, and a spin-echo scan's pair is refocused. The FLEET calibration's
    two polarities and the reversed reference volume (``SET`` 1) are read at
    a short echo or with other lines and give no field; the reversed
    reference volume is sent as no image. The field is
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
        solves them by conjugate gradients. ``None`` takes
        ``EPI_WAVELET_MULTIBAND`` for a multiband unit whose bands lie on
        calibrated slices (``EPI_WAVELET_SLICE`` for one read in shots enough
        to bring its acceleration per shot to
        ``EPI_WAVELET_SEGMENTED_ACCELERATION``), ``EPI_WAVELET_SLAB`` for any
        other unit of more than one partition accelerated beyond
        ``EPI_WAVELET_SLAB_ACCELERATION`` and ``EPI_WAVELET_SLICE`` for a
        slice or a slab accelerated no more.
    iterations : int, default=30
        Conjugate-gradient steps, or ``pics`` iterations with ``wavelet``.
    l2 : float, default=0.0
        Tikhonov weight ``lambda`` of the conjugate-gradient solve, on data
        scaled as ``pics`` scales them.
    virtual_coils : int | None, default=VIRTUAL_COILS
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`, raised to
        ``EPI_VIRTUAL_COILS_ACCELERATED`` where the header's acceleration per
        joint solve (``ACCELERATION_PER_SOLVE``) is at least
        ``EPI_ACCELERATED_SOLVE``. Compression stays first in the chain.
    kspace_filter, gradient_unwarp, intensity_correction : bool, default=True
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.
    shot_phase : bool, default=True
        Fit and encode the phase each shot of a segmented slice or multiband
        unit adds to the image; ``False`` solves every unit as one. A 3D slab
        is solved as one, each of its excitations' phase removed from its
        navigator (:class:`EpiPhaseCorrection`).
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
        shot_phase: bool = True,
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
        self.shot_phase = bool(shot_phase)
        self.shot_segments = None
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
        if self.virtual_coils is not None:
            self.virtual_coils = virtual_coils_for(
                self.virtual_coils,
                float(user_parameter(context.header, ACCELERATION_PER_SOLVE, 1.0)),
            )
            self.gadget(CompressChannels).n_virtual = self.virtual_coils
        self.calibrations = {}
        self.slice_calibrations = []
        self.own_maps = {}
        self.full_resolution_maps = {}
        self.field = None
        self.field_pair = {}
        self.pair_planes = []
        self.held_pairs = []
        self.releasing = False
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
        if self._awaits_slab_maps(context, data):
            self.held_pairs.append(data)
            return None
        image = self.image(averaged(buffer), buffer.image_shape, context, data)
        return buffer, data, image

    def _awaits_slab_maps(self, context, data) -> bool:
        """Return whether a plane of a field pair has no coil maps yet and waits for the slab it is played for.

        The planes of a 3D scan's field pair lie on the partitions of its slab,
        whose calibration is estimated with the slab's first volume, after the
        pair is read. A plane the stream holds no maps for is reconstructed
        once those maps are stored, unfolded by the slab's maps at its
        partition (:func:`slab_plane_maps`).
        """
        if self.releasing or int(data.counters.get("set", 0)) not in FIELD_PAIR_SETS:
            return False
        if data.data.kspace.ndim == 4 and data.data.kspace.shape[1] > 1:
            return False
        if data.ref is not None:
            return False
        stored = coil_maps(context, data, estimate=epi_nlinv_estimate, required=False)
        return stored is None and slab_plane_maps(context, data) is None

    def _release_pairs(self, context) -> None:
        """Reconstruct the held planes of the field pair now that the stream holds the slab's maps."""
        held, self.held_pairs = self.held_pairs, []
        self.releasing = True
        try:
            for data in held:
                self.recon(context, "", data)
        finally:
            self.releasing = False

    def image(self, kspace, shape, context, data):
        """Return the complex image of ``(coils, [partitions,] phase encodes, readout)`` k-space, solved as one volume, cropped to ``shape`` and Fermi filtered, each band of a multiband unit in its plane."""
        import torch

        volume = kspace if kspace.ndim == 4 else kspace[:, None]
        volume = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.complex64))
        in_plane = self._band_positions(data, volume.shape[1]) is not None
        slab = volume.shape[1] > 1 and not in_plane
        shots = (
            shot_masks(data.data, tuple(volume.shape[1:]))
            if self.shot_phase and not slab
            else None
        )
        maps = (
            self._maps(context, data, volume)
            if shots is None
            else self._maps(context, data, volume, segmented=True)
        )
        location = int(data.counters.get("slice", 0))
        reference = self.phase_references.get(location)
        templates = None
        if shots is not None:
            templates = [("its full-k phase reference", reference)]
            if int(data.counters.get("set", 0)) not in REVERSED_SETS:
                templates.append(
                    (
                        "its field pair's forward planes",
                        self._pair_template(data, volume),
                    )
                )
        image = self._solved(
            volume, maps, context, reference, in_plane, shots, templates
        )
        unit_set = int(data.counters.get("set", 0))
        if (
            shots is not None
            and unit_set == FULL_K_REFERENCE_SET
            and volume.shape[1] == 1
            and location not in self.full_resolution_maps
            and fully_encoded(volume[:, 0])
        ):
            image = self._drawn_after_shots(volume, location, maps.dtype, context)
        if unit_set == FULL_K_REFERENCE_SET:
            self.phase_references[location] = image
        if unit_set == FIELD_PAIR_SETS[0] and volume.shape[1] == 1:
            self.pair_planes.append((_along_slice(data.data.reference), image))
        image = image.numpy()
        if kspace.ndim == 3:
            image = image[0]
        return self._filtered(
            np.array(center_crop(image, shape[-image.ndim :])),
            (-2, -1) if in_plane else None,
        )

    def _drawn_after_shots(self, volume, location, dtype, context=None):
        """Draw a segmented full-k reference's maps from its k-space with its shot phases removed, and return its image unfolded by them.

        Each coil's image is solved with the phases fitted to the reference
        (:func:`shot_corrected`), so the coil images the maps are drawn from
        (:func:`fully_encoded_maps`) hold no alias of one shot's phase
        against another's; drawn from the k-space as read, every map would.
        """
        corrected = volume
        if self.shot_segments is not None:
            import torch

            device = getattr(context, "device", None)
            on = torch.device(device) if device is not None else volume.device
            masks, phases = self.shot_segments
            corrected = shot_corrected(volume.to(on), masks.to(on), phases.to(on)).cpu()
        drawn = fully_encoded_maps(corrected[:, 0])[:, None].to(dtype)
        self.full_resolution_maps[location] = drawn
        _log.info(
            "coil maps of slice %d drawn from its fully encoded phase reference%s",
            location,
            "" if self.shot_segments is None else ", its shot phases removed",
        )
        return sense_combined(corrected, drawn)

    def _pair_template(self, data, volume):
        """Return the complex images of the forward field pair at each plane of a unit, ``(z, y, x)``; ``None`` where the stream holds none for some plane.

        The forward pair is read with the scan's train at its echo time, so
        its planes are distorted as the scan and its full-k reference are,
        and its shot phases are fitted first, by the constant search: they are
        the image a full-k reference of several shots, played before the scan,
        has its shot phases fitted against (:meth:`_shot_phases`).
        """
        import torch

        buffer = data.data
        if buffer is None or not self.pair_planes:
            return None
        planes = []
        for position in plane_positions(
            buffer.reference, buffer.space, volume.shape[1]
        ):
            found = [
                image
                for along, image in self.pair_planes
                if abs(along - position) <= BAND_TOLERANCE_MM
            ]
            if not found:
                return None
            planes.append(found[-1].reshape(1, *volume.shape[2:]))
        return torch.cat(planes).to(torch.complex64)

    def _band_positions(self, data, count):
        """Return the position (mm) of each of a unit's ``count`` planes where they are the bands of a multiband unit; ``None`` for a slice or a slab.

        The planes (:func:`plane_positions`) are bands where each lies within
        ``BAND_TOLERANCE_MM`` of a slice the scan calibrated on its own.
        """
        buffer = data.data
        if buffer is None or count < 2:
            return None
        positions = plane_positions(buffer.reference, buffer.space, count)
        calibrated = [along for _, along, _ in self.slice_calibrations]
        if all(
            any(abs(along - position) <= BAND_TOLERANCE_MM for along in calibrated)
            for position in positions
        ):
            return positions
        return None

    def _calibrate(self, context):
        """Estimate the maps of each held calibration slice on its own, and keep its position and k-space: the position places the bands of a multiband unit (:meth:`_band_positions`), the k-space trains :func:`full_resolution_maps`."""
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

    def _wavelet(self, volume, bands=False, shots=1):
        """Return the wavelet ``lambda`` of ``(coils, z, y, x)`` k-space, whose partitions are the ``bands`` of a multiband unit or a slab's, read in ``shots`` segments.

        A multiband unit read in several shots whose in-plane line step times
        its bands, divided by its shots, is at most
        ``EPI_WAVELET_SEGMENTED_ACCELERATION`` takes ``EPI_WAVELET_SLICE``;
        any other multiband unit ``EPI_WAVELET_MULTIBAND``. A slab
        accelerated at most ``EPI_WAVELET_SLAB_ACCELERATION`` (R3 x 2) takes
        ``EPI_WAVELET_SLICE``, and a slab accelerated further
        ``EPI_WAVELET_SLAB``.
        """
        if not self.automatic_wavelet:
            return self.wavelet
        if volume.shape[1] == 1:
            return EPI_WAVELET_SLICE
        if not bands:
            if _acceleration(volume) <= EPI_WAVELET_SLAB_ACCELERATION:
                return EPI_WAVELET_SLICE
            return EPI_WAVELET_SLAB
        if shots > 1:
            per_shot = _line_step(volume) * volume.shape[1] / shots
            if per_shot <= EPI_WAVELET_SEGMENTED_ACCELERATION:
                return EPI_WAVELET_SLICE
        return EPI_WAVELET_MULTIBAND

    def _solved(
        self,
        volume,
        maps,
        context,
        reference=None,
        bands=False,
        shots=None,
        templates=None,
    ):
        """Return the image of ``(coils, z, y, x)`` k-space by the solve its sampling calls for, a partial-Fourier unit solved for a real image times its phase; ``bands`` marks the partitions as a multiband unit's (:meth:`_wavelet`), ``shots`` holds the ``(shots, z, y)`` lines of each segment of a segmented unit and ``templates`` the ``(name, image)`` its shot phases are fitted against, the full-k ``reference`` alone where it is not given (:meth:`_shot_phases`)."""
        sampled = volume.abs().sum(dim=0) > 0
        partial = partial_fourier_axes(sampled.numpy())
        fills = _fills_acquired_range(sampled.numpy()) and not partial
        wavelet = self._wavelet(
            volume, bands, 1 if shots is None else int(shots.shape[0])
        )
        l2 = 0.0 if wavelet > 0 and not fills else self.l2
        segments, start = None, None
        if shots is not None:
            if templates is None:
                templates = [("its full-k phase reference", reference)]
            segments, start = self._shot_phases(
                volume, maps, sampled, shots, templates, context, l2
            )
        self.shot_segments = segments
        if fills:
            return sense_combined(volume, maps) if segments is None else start
        if start is None:
            start = cg_sense(
                volume, maps, sampled, self.iterations, context.device, l2=l2
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
                    shots=segments,
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
            shots=segments,
        )

    def _shot_phases(self, volume, maps, sampled, masks, templates, context, l2):
        """Return the ``(masks, phases)`` of a segmented unit and its conjugate-gradient image, or ``None`` and its image solved as one.

        The phase each shot adds to the image, relative to the first shot's,
        is fitted as a smooth function (:func:`fit_shot_phases`) against an
        image of the unit, from each start in turn: each of ``templates``,
        ``(name, image)``, whose image has the unit's shape (its slice's full-k
        phase reference, then its field pair's forward planes,
        :meth:`_pair_template`), then, for at most ``SHOT_PHASE_SEARCH_SHOTS``
        shots and unless a template lowered the residual by more than
        ``SHOT_PHASE_CLEAR_GAIN`` against the solve that ignores the phases,
        the unit solved with the constant phase per shot, out of
        ``SHOT_PHASE_CONSTANTS``, that leaves the least data residual
        (:func:`shot_constants`). From each start, the fit and the joint solve
        with the phases (:func:`bartorch.tools.shot_encoding`)
        alternate until an alternation lowers the data residual by less than
        ``SHOT_PHASE_TOLERANCE``, at most ``SHOT_PHASE_ITERATIONS`` times. The
        phases of the start that leaves the least residual are kept where it
        is below that solve's. A unit whose solve as one leaves a residual
        of at most ``SHOT_PHASE_FLOOR``, or that no start improves, is solved
        as one. The fits and residuals run on ``context.device``, as the
        solves do; the phases and the image are returned on the host.
        """
        import torch
        from bartorch import tools

        begin = time.perf_counter()
        device = context.device
        on = torch.device(device) if device is not None else volume.device
        placed = (volume.to(on), maps.to(on), masks.to(on))
        merged = cg_sense(volume, maps, sampled, self.iterations, device, l2=l2)
        count = int(masks.shape[0])
        flat = torch.zeros((count, *volume.shape[1:]), device=on)
        before = tools.shot_residual(*placed, merged.to(on), flat)
        if before <= SHOT_PHASE_FLOOR:
            return None, merged
        basis = tools.shot_phase_basis(tuple(volume.shape[1:]), SHOT_PHASE_ORDER).to(on)

        def joint(phases):
            return cg_sense(
                volume,
                maps,
                sampled,
                self.iterations,
                device,
                l2=l2,
                shots=(masks, torch.exp(1j * phases).cpu()),
            ).to(on)

        starts = [
            (name, image)
            for name, image in templates
            if image is not None and tuple(image.shape) == tuple(volume.shape[1:])
        ]
        if count <= SHOT_PHASE_SEARCH_SHOTS:
            starts.append(("a search of constant shot phases", None))
        best = None
        for against, template in starts:
            if (
                template is None
                and best is not None
                and best[0] < (1.0 - SHOT_PHASE_CLEAR_GAIN) * before
            ):
                break
            if template is not None:
                image, phases = template.to(on), flat
                residual = math.inf
            else:
                phases = shot_constants(volume, maps, sampled, masks, device).to(on)
                image = joint(phases)
                residual = tools.shot_residual(*placed, image, phases)
            for _ in range(SHOT_PHASE_ITERATIONS):
                fitted = fit_shot_phases(*placed, image, phases, basis)
                solved = joint(fitted)
                lower = tools.shot_residual(*placed, solved, fitted)
                if lower >= residual:
                    break
                image, phases, previous, residual = solved, fitted, residual, lower
                if previous - residual < SHOT_PHASE_TOLERANCE * previous:
                    break
            _log.info(
                "shot phases of a unit of %d shots fitted against %s: data "
                "residual %.3f, %.3f solved as one",
                count,
                against,
                residual,
                before,
            )
            if best is None or residual < best[0]:
                best = (residual, phases, image, against)
        seconds = time.perf_counter() - begin
        if best is None or not best[0] < before:
            _log.info(
                "a unit of %d shots is solved as one%s (%.2f s)",
                count,
                ""
                if starts
                else (
                    f": it has no reference image, and the constant phases of "
                    f"more than {SHOT_PHASE_SEARCH_SHOTS} shots are not searched"
                ),
                seconds,
            )
            return None, merged
        residual, phases, image, against = best
        phases, image = phases.cpu(), image.cpu()
        _log.info(
            "shot phases of a unit of %d shots kept, fitted against %s: data "
            "residual %.3f, %.3f solved as one, in %.2f s",
            count,
            against,
            residual,
            before,
            seconds,
        )
        return (masks, torch.exp(1j * phases).to(torch.complex64)), image

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

    def _maps(self, context, data, volume, segmented=False):
        """Return the unit's coil maps shaped as ``volume``: its slice's, its bands' slices', else fitted once to it and kept.

        From a single-band slice's full-k phase reference on, its maps are
        drawn from the reference: by :func:`fully_encoded_maps` where the
        reference holds every line, else by :func:`full_resolution_maps`. A
        ``segmented`` reference that holds every line is unfolded by its
        slice's calibration maps instead, and its maps are drawn once its shot
        phases are removed (:meth:`_drawn_after_shots`). The bands of a
        multiband unit keep their slices' calibration maps.
        """
        try:
            maps = coil_maps(
                context, _polarity_of(data, volume), estimate=epi_nlinv_estimate
            ).reshape(volume.shape)
        except MissingCalibration as missing:
            bands = band_maps(context, data, volume.shape)
            if bands is not None:
                return bands
            plane = slab_plane_maps(context, data)
            if plane is not None and plane.shape == volume.shape:
                return plane
            key = (int(data.counters.get("slice", 0)), int(data.counters.get("set", 0)))
            if key not in self.own_maps:
                self._self_calibrated(context, data, missing)
                self.own_maps[key] = epi_nlinv_estimate(volume)
            return self.own_maps[key]
        if volume.shape[1] != 1:
            return maps
        location = int(data.counters.get("slice", 0))
        if location in self.full_resolution_maps:
            return self.full_resolution_maps[location]
        if int(data.counters.get("set", 0)) != FULL_K_REFERENCE_SET:
            return maps
        if fully_encoded(volume[:, 0]):
            if segmented:
                return maps
            drawn = fully_encoded_maps(volume[:, 0])[:, None].to(maps.dtype)
            self.full_resolution_maps[location] = drawn
            _log.info(
                "coil maps of slice %d drawn from its fully encoded phase reference",
                location,
            )
            return drawn
        calibration = self._slice_calibration(data)
        if calibration is None:
            return maps
        start = time.perf_counter()
        import torch

        refined = full_resolution_maps(
            volume[:, 0],
            torch.from_numpy(np.ascontiguousarray(calibration, dtype=np.complex64)),
            maps[:, 0],
            context.device,
        )[:, None]
        self.full_resolution_maps[location] = refined
        _log.info(
            "coil maps of slice %d drawn at full resolution from its phase "
            "reference by SPIRiT in %.2f s",
            location,
            time.perf_counter() - start,
        )
        return refined

    def _slice_calibration(self, data):
        """Return the ``(coils, y, x)`` calibration k-space of the unit's slice, set 0's where there are several; ``None`` where there is none."""
        if data.data is None:
            return None
        position = _along_slice(data.data.reference)
        found = [
            (entry_set != 0, values)
            for entry_set, along, values in self.slice_calibrations
            if abs(along - position) <= BAND_TOLERANCE_MM
        ]
        return min(found, key=lambda entry: entry[0])[1] if found else None

    def recon(self, context: ReconContext, branch: str, data: ReconData):
        del branch
        unit = self.unit_image(context, data)
        if unit is None:
            return None
        buffer, data, image = unit
        location = int(data.counters.get("slice", 0))
        polarity = int(data.counters.get("set", 0))
        if self.held_pairs and polarity not in FIELD_PAIR_SETS:
            self._release_pairs(context)
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
        """Estimate the stack's field from its field pair and keep it for the exam."""
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
        self._estimate_field(pairs, "its field pair")
        if self.field is not None:
            self.field = self.field._replace(train=train)
            context.exam.set(SUSCEPTIBILITY_FIELD, self.field)

    def _results(self, context, data, image):
        """Return the image's results, its geometry its own unit's whatever unit closed, the bands of a multiband unit each corrected in its plane."""
        corrections = self.corrections
        history = [
            "PULSERVER",
            "PYTHON",
            "PICS" if self.automatic_wavelet or self.wavelet > 0 else "SENSE",
        ]
        planes = self._band_positions(data, image.shape[0]) if image.ndim == 3 else None
        if corrections is not None:
            image = corrections.unwarped(image, context, data, planes)
            history += corrections.applied()
        results = [
            ReconResult(
                np.abs(image),
                reference=data.data.reference,
                attributes={"ImageProcessingHistory": history},
            )
        ]
        if corrections is not None:
            results = corrections.intensity_corrected(results, data, planes is not None)
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
            reason = "no field pair (sets 2 and 3) came before it"
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


#: Wavelet ``lambda`` of an undersampled 3D EPI slab accelerated beyond
#: ``EPI_WAVELET_SLAB_ACCELERATION``, 2D EPI multiband unit and 2D EPI slice,
#: on data scaled as ``pics`` scales them. A slice's prior is weak enough to
#: keep the cortex and the deep grey nuclei. A multiband unit unfolds its
#: bands and its in-plane aliases together: in a noisy 7T simulation of a
#: 2 mm stack at MB2 and R 2 to 4 with six virtual channels, its bands are
#: closest to a fully sampled gradient echo near this weight, a slice's
#: leaving R 4 aliased.
EPI_WAVELET_SLAB = 0.03
EPI_WAVELET_MULTIBAND = 0.03
EPI_WAVELET_SLICE = 0.003

#: Largest acceleration of a 3D EPI slab, the encoded lines of its partition
#: and phase-encode extent over those read, solved at ``EPI_WAVELET_SLICE``
#: instead of ``EPI_WAVELET_SLAB``. In the noisy 7T simulation of a 2 mm slab
#: of 16 partitions in the susceptibility field, unwarped, against the fully
#: sampled gradient echo: R2 x 2 in 3 shots with six virtual channels came
#: out at 0.106 brain NRMSE at the slice weight and by conjugate gradients
#: alike, and R3 x 2 with twelve virtual channels at 0.176 at the slice
#: weight, 0.188 at the slab weight and 0.185 by conjugate gradients (0.129,
#: and 0.137 by conjugate gradients, in 2 shots).
EPI_WAVELET_SLAB_ACCELERATION = 6.0

#: Largest in-plane line step times bands, divided by its shots, of a
#: segmented multiband unit solved at ``EPI_WAVELET_SLICE`` instead of
#: ``EPI_WAVELET_MULTIBAND``. A segmented unit's shorter echo train blurs it
#: less, and the multiband weight's smoothing is then what shows: in the 7T
#: simulation, R2 x MB2 in 3 shots (4 / 3 per shot) came out at the GRE's
#: sharpness at the slice weight for a brain NRMSE of 0.073 against 0.070,
#: and in 2 shots (2 per shot) 0.078 against 0.077, while a single-shot
#: R2 x MB2 unit (4) lost at it (0.139 against 0.136) and R3 x MB2 (6) needs
#: the multiband weight against its aliases.
EPI_WAVELET_SEGMENTED_ACCELERATION = 2.0

#: Acceleration per joint solve (in-plane line step times partition step times
#: bands, divided by shots) from which a stream is compressed to
#: ``EPI_VIRTUAL_COILS_ACCELERATED`` virtual channels instead of
#: ``virtual_coils``: a solve that unfolds this much needs more of the array's
#: sensitivity variation than the first six compressed channels carry.
EPI_ACCELERATED_SOLVE = 6.0
EPI_VIRTUAL_COILS_ACCELERATED = 12

#: Sobolev weight of the EPI coil maps on their calibration grid. Below
#: BART's, the maps follow the steep sensitivity next to the coil elements:
#: along the scalp of a close-fitting array, where the scalp is far brighter
#: than the brain and smoother maps fold it into the brain at R=3, and across
#: the end partitions of a slab whose signal reaches them and wraps onto each
#: other.
EPI_SOBOLEV = 1.0


#: Kernel extent, consistency weight and conjugate-gradient steps of the
#: SPIRiT reconstruction :func:`full_resolution_maps` draws coil maps from.
SPIRIT_KERNEL = 5
SPIRIT_WEIGHT = 1.0
SPIRIT_ITERATIONS = 100

#: Root-sum-of-squares, as a fraction of its 99th percentile, at which the
#: maps of :func:`full_resolution_maps` weigh the reference's coil images and
#: the calibration's maps equally.
SPIRIT_MAPS_FLOOR = 0.05


def full_resolution_maps(kspace, calibration, maps, device):
    """Return coil maps of a slice at the resolution of its fully encoded k-space.

    ``kspace`` is reconstructed coil by coil by SPIRiT, its kernel fitted to
    the slice's calibration; its coil images divided by their
    root-sum-of-squares are the maps where there is signal. Elsewhere the
    maps fall back to ``maps``, given the phase of the coil images. Maps the
    calibration's few central lines resolve are too smooth next to the
    elements of a close-fitting array, where the steep sensitivity of the
    bright scalp folds it into the brain at ``R=3``.

    Parameters
    ----------
    kspace : torch.Tensor
        ``(coils, y, x)`` centred k-space of a slice, zero where not sampled,
        its lines spread over the whole of k-space.
    calibration : torch.Tensor
        ``(coils, y, x)`` calibration k-space of the slice, zero outside its
        central lines.
    maps : torch.Tensor
        ``(coils, y, x)`` maps of the slice from its calibration.
    device : str or torch.device
        Device of the conjugate-gradient solve.

    Returns
    -------
    torch.Tensor
        ``(coils, y, x)`` maps, of unit root-sum-of-squares.
    """
    import bartorch
    import torch
    from bartorch import linop, optim, tools

    lines = torch.nonzero(calibration.abs().sum(dim=(0, 2)) > 0)[:, 0]
    region = calibration[:, int(lines[0]) : int(lines[-1]) + 1].contiguous()
    kernel = tools.spirit_kernel(region.to(torch.complex64), SPIRIT_KERNEL)
    grid = tuple(kspace.shape[1:])
    shape = tuple(kspace.shape)
    sampling = linop.Diagonal(
        (kspace.abs().sum(dim=0, keepdim=True) > 0).to(torch.complex64), shape
    )
    solve = optim.CG(
        terms=optim.Tikhonov(
            SPIRIT_WEIGHT, operator=linop.spirit_consistency(kernel, grid)
        ),
        maxiter=SPIRIT_ITERATIONS,
        device=device,
    )
    filled = solve(kspace.to(torch.complex64), sampling).reshape(shape).cpu()
    images = bartorch.fft(filled, (-1, -2), inverse=True, unitary=True).reshape(shape)
    rss = images.abs().square().sum(dim=0).sqrt()
    weight = rss / (rss + SPIRIT_MAPS_FLOOR * torch.quantile(rss.flatten(), 0.99))
    drawn = images / rss.clamp_min(1e-12)
    phase = torch.exp(1j * torch.angle((maps.conj() * drawn).sum(dim=0)))
    blended = weight * drawn + (1 - weight) * maps * phase
    return (blended / blended.abs().square().sum(dim=0).sqrt().clamp_min(1e-12)).to(
        maps.dtype
    )


#: Root-sum-of-squares, as a fraction of its 99th percentile, at which the
#: maps of :func:`fully_encoded_maps` weigh a pixel's own coil images and
#: the NLINV maps equally.
ENCODED_MAPS_FLOOR = 0.02


def fully_encoded(kspace):
    """Return whether ``(coils, y, x)`` k-space holds every phase-encode line of its grid."""
    return bool((kspace.abs().sum(dim=(0, 2)) > 0).all())


def fully_encoded_maps(kspace):
    """Return coil maps of a slice drawn from k-space that encodes it whole.

    The coil images divided by their root-sum-of-squares are the maps where
    there is signal. A reference read with the echo train and at the echo
    time of the imaging holds, pixel by pixel, the sensitivity the imaging
    signal of that pixel is received with, also where the field has piled
    the signal of several places, each at its own phase, onto one pixel:
    there no map smooth in space, and none fitted at another echo time,
    unfolds the imaging. Where a pixel has little signal its maps are those
    NLINV fits to the central lines of the same k-space
    (:func:`epi_nlinv_estimate`), smooth into the air around the head.

    Parameters
    ----------
    kspace : torch.Tensor
        ``(coils, y, x)`` centred k-space of the slice, every phase-encode
        line sampled.

    Returns
    -------
    torch.Tensor
        ``(coils, y, x)`` maps, complex64, of unit root-sum-of-squares.
    """
    import bartorch
    import torch

    shape = tuple(kspace.shape)
    axes = (-1, -2)
    images = bartorch.fft(
        kspace.to(torch.complex64), axes, inverse=True, unitary=True
    ).reshape(shape)
    rss = images.abs().square().sum(dim=0).sqrt()
    floor = ENCODED_MAPS_FLOOR * torch.quantile(rss.flatten(), 0.99)
    weight = rss.square() / (rss.square() + floor.square()).clamp_min(1e-30)
    drawn = images / rss.clamp_min(1e-12)
    nearby = torch.as_tensor(epi_nlinv_estimate(kspace.to(torch.complex64)[:, None]))
    nearby = nearby.reshape(shape).to(images.device, torch.complex64)
    phase = torch.exp(1j * torch.angle((nearby.conj() * drawn).sum(dim=0)))
    blended = weight * drawn + (1 - weight) * nearby * phase
    return (blended / blended.abs().square().sum(dim=0).sqrt().clamp_min(1e-12)).to(
        torch.complex64
    )


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


def _polarity_of(data, volume):
    """Return ``data``, or for a reversed volume of a slab a view of it counted in the slab calibration's reversed set.

    A slab's FLEET calibration stores maps for each phase-encode polarity,
    set 0 forward and set 1 reversed (:func:`coil_maps`). The field pair's
    reversed volume and the reversed full-k reference are distorted as the
    calibration's reversed lines are, so they are unfolded by those maps.
    """
    polarity = int(data.counters.get("set", 0))
    if volume.shape[1] < 2 or data.ref is not None or polarity not in REVERSED_SETS:
        return data
    return dataclasses.replace(data, counters={**data.counters, "set": 1})


def slab_plane_maps(context, data):
    """Return the coil maps of a single-plane unit drawn from a slab's maps at its partition; ``None`` where the stream holds no slab through it.

    A slab's maps ``(coils, partitions, y, x)`` stored in the stream serve a
    plane parallel to the slab, on the same in-plane grid, lying within
    ``PARTITION_TOLERANCE`` of a partition's centre (:func:`plane_positions`).
    A plane of the reversed field-pair set takes the maps of the slab's
    reversed calibration (set 1) where the stream holds them, every other
    plane those of set 0.

    Returns
    -------
    torch.Tensor or None
        ``(coils, 1, y, x)`` complex64 maps on the host.
    """
    import torch

    buffer = data.data
    if buffer is None:
        return None
    reversed_set = FIELD_PAIR_SETS[1]
    wanted = 1 if int(data.counters.get("set", 0)) == reversed_set else 0
    along = _along_slice(buffer.reference)
    normal = np.asarray(getattr(buffer.reference, "slice_dir", (0, 0, 1)), float)
    plane = tuple(buffer.image_shape[-2:])
    found = {}
    for key, stored in context.coil_maps.items():
        maps = np.asarray(getattr(stored, "maps", None))
        geometry = getattr(stored, "geometry", {})
        extent = geometry.get("extent")
        if maps.ndim != 4 or maps.shape[1] < 2 or extent is None:
            continue
        if tuple(maps.shape[-2:]) != plane:
            continue
        slab_normal = np.asarray(geometry.get("slice_dir") or (0, 0, 1), float)
        if abs(abs(float(slab_normal @ normal)) - 1.0) > 1e-3:
            continue
        centre = float(np.asarray(geometry.get("position") or (0, 0, 0)) @ slab_normal)
        depth = float(extent[0]) / maps.shape[1]
        offset = (along * float(slab_normal @ normal) - centre) / depth
        index = round(offset) + maps.shape[1] // 2
        if abs(offset - round(offset)) > PARTITION_TOLERANCE or not (
            0 <= index < maps.shape[1]
        ):
            continue
        key_set = key[1] if isinstance(key, tuple) else 0
        found[key_set] = maps[:, index : index + 1]
    chosen = found.get(wanted, found.get(0))
    return None if chosen is None else torch.from_numpy(np.ascontiguousarray(chosen))


#: Fraction of a partition within which a plane lies on its centre.
PARTITION_TOLERANCE = 0.1


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


#: Distance (mm) within which a stored slice's position is a band's.
BAND_TOLERANCE_MM = 0.01


def _along_slice(acquisition) -> float:
    """Return an acquisition's position along its slice direction (mm)."""
    position = np.asarray(getattr(acquisition, "position", (0, 0, 0)), float)
    normal = np.asarray(getattr(acquisition, "slice_dir", (0, 0, 1)), float)
    return float(position @ normal)


def virtual_coils_for(virtual_coils: int, acceleration: float) -> int:
    """Return the virtual channels a stream is compressed to at an acceleration per joint solve.

    Parameters
    ----------
    virtual_coils : int
        Channels kept below ``EPI_ACCELERATED_SOLVE``.
    acceleration : float
        In-plane line step times partition step times bands, divided by the
        shots of one imaging unit.

    Returns
    -------
    int
        ``EPI_VIRTUAL_COILS_ACCELERATED`` where ``acceleration`` is at least
        ``EPI_ACCELERATED_SOLVE`` and more than ``virtual_coils``,
        ``virtual_coils`` otherwise.
    """
    if acceleration >= EPI_ACCELERATED_SOLVE:
        return max(virtual_coils, EPI_VIRTUAL_COILS_ACCELERATED)
    return virtual_coils


def _acceleration(volume) -> float:
    """Return the lines of the partition and phase-encode extent ``(coils, z, y, x)`` k-space acquired, over those it holds; 1 where it holds none."""
    magnitude = volume.abs().cpu().numpy() if hasattr(volume, "cpu") else np.abs(volume)
    read = magnitude.sum(axis=(0, 3)) > 0
    if not read.any():
        return 1.0
    extent = [np.flatnonzero(read.any(axis=1 - axis)) for axis in (0, 1)]
    box = np.prod([found[-1] - found[0] + 1 for found in extent])
    return float(box / read.sum())


def _line_step(volume) -> int:
    """Return the most common step between the phase-encode lines ``(coils, z, y, x)`` k-space holds, on any partition; 1 where it holds fewer than two."""
    magnitude = volume.abs().cpu().numpy() if hasattr(volume, "cpu") else np.abs(volume)
    acquired = np.flatnonzero(magnitude.sum(axis=(0, 1, 3)) > 0)
    if acquired.size < 2:
        return 1
    return int(np.bincount(np.diff(acquired)).argmax())


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
    from bartorch import apps

    mask = sampled.cpu().numpy()
    kept = np.ones(mask.shape, dtype=bool)
    for axis in range(mask.ndim):
        lines = apps.partial_fourier_range(mask, axis)
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


def cg_sense(kspace, maps, sampled, iterations, device, *, l2=0.0, shots=None):
    """Return the image ``(z, y, x)`` minimising ``|P F S x - y|^2 + l2 |x|^2``, ``l2`` at least ``CG_MIN_L2``, by conjugate gradients from zero.

    The encoding is :class:`bartorch.linop.CartesianSense` of ``maps`` over
    the ``sampled`` lines, on ``device``, with ``shots``, the ``(masks,
    phases)`` of a segmented unit, each shot's phase in it
    (:func:`bartorch.tools.shot_encoding`); stopping after
    ``iterations`` steps regularizes the solve. The data are scaled as
    ``pics`` scales them.
    """
    import bartorch
    import torch
    from bartorch import linop, optim, tools

    grid = tuple(kspace.shape[2:] if kspace.shape[1] == 1 else kspace.shape[1:])
    pattern = sampled.reshape(grid).to(torch.complex64)
    measured = kspace * sampled
    encoding = linop.CartesianSense(
        maps.reshape(maps.shape[0], *grid), grid, pattern=pattern, device=device
    )
    if shots is not None:
        encoding = tools.shot_encoding(encoding, *shots)
    shifted = bartorch.fftmod(measured, axes=(-1, -2, -3), inverse=True)
    scaling = optim.data_scaling(shifted.reshape(kspace.shape[0], *grid), A=encoding)
    scaling = scaling or 1.0
    data = (measured * (1.0 / scaling)).reshape(kspace.shape[0], *grid)
    solve = optim.CG(lambda_=max(float(l2), CG_MIN_L2), maxiter=int(iterations))
    return (scaling * solve(data, encoding)).reshape(kspace.shape[1:])


#: Degree of the Legendre polynomials in plane, per partition, that a shot's
#: phase is fitted on (:func:`bartorch.tools.shot_phase_basis`). The fit error turns into frame-
#: to-frame magnitude change: in a 7T simulation of R2 x MB2 in 3 shots, 20
#: frames each with its own shot phase (0.04 to 0.2 rad across the head plus
#: blobs of 0.06 to 0.12 rad), brain tSNR was 73, 88, 93, 95 and 97 at degrees
#: 2, 3, 4, 6 and 8 against 99 with the true phases and 107 without shot
#: phase, single-volume NRMSE unchanged (0.069); degree 6 costs 3.4 s per
#: volume on an A40 against 3.1 s at degree 3 (21 s against 11 s on 4 CPU
#: threads).
SHOT_PHASE_ORDER = 6

#: Most alternations of the shot-phase fit and the joint solve of a segmented
#: unit, and the fraction by which an alternation must lower its data
#: residual for the next to run. In the 7T simulation an imaging unit fitted
#: against its full-k reference settles after one; the 6-shot full-k reference
#: of an MB2 x R2 x 3-shot scan, fitted against its field pair, came within
#: 0.03 of the reference without shot phase after six, against 0.06 after one.
SHOT_PHASE_ITERATIONS = 8
SHOT_PHASE_TOLERANCE = 0.02

#: Gauss-Newton steps of each shot-phase fit, and their Levenberg damping, a
#: fraction of the mean curvature.
SHOT_PHASE_STEPS = 2
SHOT_PHASE_DAMPING = 1e-3

#: Phase basis functions carried through the forward and adjoint model at
#: once while a shot's Gauss-Newton curvature is built. Each holds the coils
#: times the volume, so the peak memory of the fit grows with it.
SHOT_PHASE_TERMS_AT_ONCE = 28

#: Constant phases, evenly spaced over a turn, tried for each shot of a
#: segmented unit with no reference image, and the conjugate-gradient steps of
#: each trial solve.
SHOT_PHASE_CONSTANTS = 8
SHOT_PHASE_SEARCH_STEPS = 8

#: Most shots of a unit with no reference image whose constant phases are
#: searched; a unit of more is solved as one. The search costs a trial solve
#: per constant and shot, and in a noisy 7T simulation it found the phases of
#: a 2-shot unit but not those of the 4-shot full-k reference of an MB2 x R2
#: scan, each of whose shots reads one partition of the CAIPI lattice.
SHOT_PHASE_SEARCH_SHOTS = 3

#: Conjugate-gradient steps of each coil's image with the shot phases removed
#: (:func:`shot_corrected`), which a segmented full-k reference's maps are
#: drawn from.
SHOT_PHASE_COIL_STEPS = 20

#: Fraction by which the shot phases fitted against a template must lower a
#: segmented unit's data residual, against its solve as one, for the constant
#: search not to be tried. The start that leaves the least residual is kept,
#: if it lowers the residual at all: a shot phase of a few
#: tenths of a radian lowers the residual of a noisy 7T unit by about 1 % and
#: still aliases its image visibly, and phases fitted against an image that
#: does not describe the unit lower it as little.
SHOT_PHASE_CLEAR_GAIN = 0.05

#: Data residual of a segmented unit solved as one, relative to its k-space,
#: below which its shots agree and no shot phase is fitted.
SHOT_PHASE_FLOOR = 1e-3


def shot_masks(buffer, shape):
    """Return the ``(shots, z, y)`` lines each segment of a unit placed; ``None`` where fewer than two segments placed any.

    A line's segment is its acquisition's ``segment`` counter (``SEG``), and
    the shots are in increasing segment order, so the first is the reference
    every other shot's phase is fitted against. ``shape`` is the unit's
    ``(z, y, x)``; a unit without a partition axis is one partition.
    """
    import torch

    headers = getattr(buffer, "headers", None) or []
    if not headers or not hasattr(buffer, "axes"):
        return None
    axes = tuple(buffer.axes[1:-1])
    lines = {}
    for header in headers:
        where = dict(zip(axes, buffer.position(header), strict=True))
        segment = int(acquisition_label(header, "segment", 0) or 0)
        lines.setdefault(segment, []).append(
            (where.get("partition", 0), where.get("phase_encode", 0))
        )
    if len(lines) < 2:
        return None
    masks = np.zeros((len(lines), *shape[:2]), dtype=bool)
    for index, segment in enumerate(sorted(lines)):
        for partition, line in lines[segment]:
            masks[index, partition, line] = True
    return torch.from_numpy(masks)


def shot_corrected(kspace, masks, phases, iterations=SHOT_PHASE_COIL_STEPS):
    """Return ``(coils, z, y, x)`` k-space with the phase of each shot removed, coil by coil.

    Each coil's image ``u`` minimises ``sum_s |M_s F(c_s u) - y_s|^2`` over
    the shots' lines ``masks`` and unit-magnitude ``phases`` ``c_s``, by
    ``iterations`` conjugate-gradient steps from the coil image as read; the
    k-space returned is that of ``u``.
    """
    import torch
    from bartorch import linop, optim

    shape = tuple(kspace.shape)
    axes = (-1, -2, -3) if shape[1] > 1 else (-1, -2)
    transform = linop.FFT(shape, axes)
    shots = [
        linop.Diagonal(mask.to(kspace.dtype)[None, ..., None].contiguous(), shape)
        @ transform
        @ linop.Diagonal(phase[None].to(kspace.dtype).contiguous(), shape)
        for mask, phase in zip(masks, phases, strict=True)
    ]
    data = torch.stack([kspace * mask.to(kspace.dtype)[..., None] for mask in masks])
    images = transform.H(kspace.contiguous())
    solved = optim.CG(maxiter=int(iterations))(
        data.contiguous(), linop.stack(shots), x0=images
    )
    return transform(solved.reshape(shape).contiguous()).reshape(shape)


def shot_constants(kspace, maps, sampled, masks, device):
    """Return ``(shots, z, y, x)`` constant phases (rad), the first shot's zero: each later shot's, in turn, the one of ``SHOT_PHASE_CONSTANTS`` whose solve (``SHOT_PHASE_SEARCH_STEPS`` steps) leaves the least data residual, the solves and residuals on ``device``."""
    import torch
    from bartorch import tools

    on = torch.device(device) if device is not None else kspace.device
    placed = (kspace.to(on), maps.to(on), masks.to(on))
    phases = torch.zeros((masks.shape[0], *kspace.shape[1:]))
    for shot in range(1, masks.shape[0]):
        best = None
        for constant in np.linspace(
            -math.pi, math.pi, SHOT_PHASE_CONSTANTS, endpoint=False
        ):
            trial = phases.clone()
            trial[shot] = float(constant)
            image = cg_sense(
                kspace,
                maps,
                sampled,
                SHOT_PHASE_SEARCH_STEPS,
                device,
                shots=(masks, torch.exp(1j * trial)),
            )
            residual = tools.shot_residual(*placed, image.to(on), trial.to(on))
            if best is None or residual < best[0]:
                best = (residual, float(constant))
        phases[shot] = best[1]
    return phases


def fit_shot_phases(kspace, maps, masks, image, phases, basis):
    """Return the ``(shots, z, y, x)`` phases (rad) each shot adds to ``image``, relative to the first shot's.

    :func:`bartorch.tools.estimate_shot_phases` with ``SHOT_PHASE_STEPS``
    Gauss-Newton steps, ``SHOT_PHASE_DAMPING`` and
    ``SHOT_PHASE_TERMS_AT_ONCE``; ``basis`` is
    :func:`bartorch.tools.shot_phase_basis`.
    """
    from bartorch import tools

    return tools.estimate_shot_phases(
        kspace,
        maps,
        masks,
        image,
        phases,
        basis,
        steps=SHOT_PHASE_STEPS,
        damping=SHOT_PHASE_DAMPING,
        terms_at_once=SHOT_PHASE_TERMS_AT_ONCE,
    )


def susceptibility_field(image, reverse, voxel_size):
    """Return the displacement field of ``image`` against its phase-encode-reversed ``reverse``.

    A slice is ``(phase encode, readout)`` and a volume ``(partition, phase
    encode, readout)``, with ``voxel_size`` in mm along the same axes. The
    field is PyHySCO's (:func:`bartorch.tools.correct_susceptibility`), in mm
    on the cell edges of the phase encode, in the axes of ``image``: one row
    (a volume: one phase-encode line) longer. A volume is corrected as one,
    its phase encode first, at the smoothness weight
    ``SLAB_FIELD_SMOOTHNESS``, and a slice in its plane, at
    ``SLICE_FIELD_SMOOTHNESS``.

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
        smoothness = {"alpha": SLICE_FIELD_SMOOTHNESS}
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
#: :func:`bartorch.tools.correct_susceptibility`. In a noisy 7T simulation of
#: a 2 mm stack (R 2 to 4, partial Fourier 6/8), the unwarped scan is closest
#: to a fully sampled gradient echo in the same field near this weight, with
#: a gradient-echo or a spin-echo pair. Much looser the field takes up the
#: ventricles and the noise of the pair, and its steps draw dark lines in the
#: corrected image; much stiffer it recovers less of the displacement near
#: the sinuses. R 2 alone, with twice R 4's displacement, prefers a third of it.
SLAB_FIELD_SMOOTHNESS = 3000.0

#: PyHySCO smoothness weight of a slice's field, on the same scale. A slice
#: has no neighbours to constrain its field, and at PyHySCO's own weight the
#: field of a 3T spin-echo pair takes up the anatomy of the pair's images.
SLICE_FIELD_SMOOTHNESS = 3000.0


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


#: ``SET`` of the forward and of the reversed volume of a field pair.
FIELD_PAIR_SETS = (2, 3)

#: Exam key of the last field pair's :class:`PlaneField`.
SUSCEPTIBILITY_FIELD = "susceptibility_field"

#: ``SET`` of the full-k phase reference volume.
FULL_K_REFERENCE_SET = 4

#: ``SET`` of the volumes read with the phase encode reversed: the FLEET
#: calibration's second polarity, the field pair's and the reference's. They
#: are distorted the other way, and no forward image is their template.
REVERSED_SETS = (1, FIELD_PAIR_SETS[1], FULL_K_REFERENCE_SET + 1)

#: Fraction of a partial-Fourier image's energy in quadrature to its smooth
#: phase above which that phase does not describe the image. A spin echo
#: keeps about 0.001 there, a gradient echo at 7T near air-tissue interfaces
#: 0.015 or more.
SMOOTH_PHASE_RESIDUAL = 0.01

#: The package PyHySCO installs, which the susceptibility correction imports.
PYHYSCO_PACKAGE = "EPI_MRI"

PLUGIN = EpiRecon()
