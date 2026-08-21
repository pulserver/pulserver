"""What happens to a readout between the scanner and its place in k-space.

Gadgetron puts these steps in the chain ahead of the buffer -- noise adjustment,
coil reduction, the EPI corrections -- because each one is per-acquisition work
that has nothing to do with the reconstruction that follows it, and because a
readout is only worth placing once it has had them. Pulserver keeps that idea
and the names: a :class:`Gadget` is one such step, a plugin lists the ones it
wants as its ``chain``, and :meth:`pulserver.recon.ReconPlugin.receive` runs
them in order as each acquisition lands.

A gadget is stateful on purpose -- a noise covariance, a coil basis, a phase
fit are all learned from acquisitions that came earlier -- so one belongs to
one stream, and :meth:`Gadget.startup` is where that state is set up.

Returning ``None`` consumes the acquisition: a noise scan and a navigator line
measure the receiver and the readout rather than the object, so nothing of them
reaches a buffer, and what they leave behind is the correction every later line
is put through.
"""

from __future__ import annotations

__all__ = [
    "CoilCompression",
    "EpiPhaseCorrection",
    "Gadget",
    "NoiseAdjust",
    "RampSampling",
]

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ._buffers import EncodingSpace
from ._mrd.metadata import has_acquisition_flag
from .preprocessing import (
    coil_compress,
    correct_lines,
    epi_ramp_operator,
    estimate_epi_phase,
    noise_prewhiten,
)


class Gadget(ABC):
    """One per-acquisition step of a plugin's chain.

    Subclass it, keep whatever the step learns on ``self``, and return the
    readout the next step should see. Returning ``None`` consumes the
    acquisition: it never reaches a buffer and never reaches
    :meth:`~pulserver.recon.ReconPlugin.recon`.

    Attributes
    ----------
    context : ReconContext
        The scan context, from :meth:`startup`.
    """

    def startup(self, context: Any) -> None:
        """Prepare for one stream, before its first acquisition.

        The default keeps the context, which is what a gadget sharing an
        artifact through ``context.exam`` needs. Override to add state, and
        call ``super().startup(context)`` to keep it.
        """
        self.context = context

    @abstractmethod
    def __call__(self, acquisition: Any, data: Any) -> Any:
        """Return this readout as the next step should see it, or ``None``.

        Parameters
        ----------
        acquisition
            The acquisition, for its flags and counters.
        data
            The readout so far, ``(coils, samples)`` -- what the acquisition
            carried, or what the step before this one returned.

        Returns
        -------
        ndarray or None
            The readout, or ``None`` to consume the acquisition.
        """


class NoiseAdjust(Gadget):
    """Whiten every readout against the scan's own noise measurement.

    Gadgetron's ``NoiseAdjustGadget``. A noise scan is a measurement of the
    receiver rather than of the object, so it is consumed: what it leaves
    behind is the covariance every readout that follows is decorrelated
    against. A scan whose scanner sends none passes through untouched.
    """

    def startup(self, context: Any) -> None:
        """Start the stream with no noise measured yet."""
        super().startup(context)
        self.noise: Any = None

    def __call__(self, acquisition: Any, data: Any) -> Any:
        """Consume a noise scan, or whiten an imaging readout."""
        if has_acquisition_flag(acquisition, "ACQ_IS_NOISE_MEASUREMENT"):
            self.noise = (
                data if self.noise is None else np.concatenate([self.noise, data], -1)
            )
            return None
        if self.noise is None:
            return data
        return noise_prewhiten(data, self.noise, coil_axis=0)


class CoilCompression(Gadget):
    """Project every readout onto the array's principal channels.

    Gadgetron's ``PCACoilGadget``. The basis is learned once, from the
    calibration the scan acquired for it, and left in ``context.exam`` under
    :attr:`key` -- so a prescan that arrives as a stream of its own still
    compresses the imaging that follows it onto the same channels. Until it
    exists, readouts pass through at full channel count.

    The acquisitions the basis is learned *from* are not compressed on the way
    in, or there would be nothing to learn it from: ``learn_from`` names them,
    and :meth:`learn` is what the calibration branch calls once its buffer is
    filled.

    Parameters
    ----------
    virtual_coils
        Channels to keep. A scan with fewer physical channels keeps them all.
    key
        Where the basis is left in the exam cache.
    learn_from
        Flag marking the acquisitions the basis is learned from, which are
        therefore passed through uncompressed.
    """

    def __init__(
        self,
        virtual_coils: int,
        *,
        key: str = "coil_basis",
        learn_from: Any = "ACQ_IS_PARALLEL_CALIBRATION",
    ) -> None:
        self.virtual_coils = int(virtual_coils)
        self.key = key
        self.learn_from = learn_from

    @property
    def basis(self) -> Any:
        """The compression basis, or ``None`` before one is learned."""
        return self.context.exam.get(self.key)

    def learn(self, kspace: Any, mask: Any = None) -> Any:
        """Learn the basis from filled calibration k-space, and share it.

        Parameters
        ----------
        kspace
            The calibration buffer, coils first.
        mask
            Where it actually landed. ``None`` uses every position.

        Returns
        -------
        ndarray
            The basis, ``(virtual_coils, coils)``, also left in the exam cache.
        """
        lines = kspace if mask is None else kspace[:, mask.any(axis=-1)]
        _, basis = coil_compress(lines.reshape(kspace.shape[0], -1), self.virtual_coils)
        self.context.exam.set(self.key, basis)
        return basis

    def __call__(self, acquisition: Any, data: Any) -> Any:
        """Compress the readout, once there is a basis and it is not its source."""
        if has_acquisition_flag(acquisition, self.learn_from):
            return data
        basis = self.basis
        return data if basis is None else basis @ data


class EpiPhaseCorrection(Gadget):
    """Undo the odd/even phase an echo-planar train leaves on reversed lines.

    Gadgetron's ``EPICorrGadget``. A reversed line carries a phase its forward
    neighbours do not, and leaving it there puts a ghost at half the field of
    view. The blip-nulled navigator triplet the sequence plays is a measurement
    of the readout rather than of the object, so it is consumed; the fit it
    produces is what every reversed line that follows is flipped and
    demodulated by.

    Parameters
    ----------
    order
        Order of the fitted phase. One is the gradient-delay ramp every product
        reconstruction corrects; raising it picks up what an eddy current
        leaves beyond a ramp.
    """

    def __init__(self, *, order: int = 1) -> None:
        self.order = int(order)

    def startup(self, context: Any) -> None:
        """Start the stream with no navigator and no fit."""
        super().startup(context)
        self.navigator: list[Any] = []
        self.phase: Any = None

    def __call__(self, acquisition: Any, data: Any) -> Any:
        """Collect a navigator line, or correct an imaging readout."""
        backwards = has_acquisition_flag(acquisition, "ACQ_IS_REVERSE")
        if has_acquisition_flag(acquisition, "ACQ_IS_PHASECORR_DATA"):
            self.navigator.append(data[..., :: -1 if backwards else 1])
            if len(self.navigator) == 3:
                self.phase = estimate_epi_phase(
                    self.navigator, polynomial_order=self.order
                )
                self.navigator = []
            return None
        (corrected,) = correct_lines([(data, backwards)], self.phase)
        return corrected


class RampSampling(Gadget):
    """Resample a ramp-sampled readout onto the grid it belongs on.

    Gadgetron's ``EPIReconXGadget``. A train worth playing samples across its
    read ramps rather than waiting for the plateau, so k does not advance at a
    constant rate along a readout and the samples are not on the grid. Where
    they fell is what the acquisition's trajectory says -- a client attaches
    one exactly when the gradient was still moving under the ADC -- and the
    change of basis onto the grid is exact while the samples outnumber the
    pixels they determine.

    It is normalised onto the readout's own extent, so the units the client
    wrote do not matter. An acquisition carrying no trajectory was sampled
    uniformly, which is what a train that waits for its plateau is, and passes
    through.
    """

    def startup(self, context: Any) -> None:
        """Read the encoding space the readouts are resampled onto.

        The space rather than the buffer: a buffer would be allocated at the
        header's channel count, before the first compressed readout reached it.
        """
        super().startup(context)
        self.space = EncodingSpace.from_header(context.header)
        self.operator: Any = None

    def __call__(self, acquisition: Any, data: Any) -> Any:
        """Regrid the readout, if the gradient was still moving under it."""
        trajectory = getattr(acquisition, "traj", None)
        samples = data.shape[-1]
        if trajectory is None or np.size(trajectory) < samples:
            return data
        if self.operator is None or self.operator.shape[1] != samples:
            self.operator = self._change_of_basis(trajectory, samples)
        return data @ self.operator.T

    def _change_of_basis(self, trajectory: Any, samples: int) -> Any:
        """The resampling one readout length needs, built once and reused."""
        taken = np.asarray(trajectory).reshape(samples, -1)[:, 0]
        # k is zero at the echo and a truncated readout still ends where a full
        # one would, so the largest |k| it reaches is half the full sweep --
        # which normalises it without assuming this readout swept all of it.
        taken = taken / (2.0 * np.abs(taken).max())
        # The grid is the whole encoded readout, not the part this one sampled:
        # a partial echo resamples onto the same pitch as a full one and is
        # right-aligned in it, exactly as the buffer places it.
        readout = self.space.readout
        grid = (np.arange(readout) - readout // 2) / readout
        return epi_ramp_operator(
            taken, grid[readout - samples :], self.space.recon_matrix[-1]
        )
