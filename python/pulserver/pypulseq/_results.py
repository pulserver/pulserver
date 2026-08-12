"""What the analysis methods return when they are not being PyPulseq.

Every analysis method on :class:`~pulserver.pypulseq.Sequence` takes a
keyword-only ``compat``. Left alone it is ``True`` and the method returns
exactly what upstream PyPulseq returns, down to upstream's omissions -- that is
what makes an unchanged PyPulseq script keep its unchanged PyPulseq meaning.
Passing ``compat=False`` returns one object from this module instead, carrying
the same information under names plus everything upstream's tuple has nowhere
to put.

The flag exists rather than a longer tuple because a tuple return is unpacked
positionally: ``a, b, c = seq.waveforms_and_times()`` breaks the moment a
fourth element appears, and it breaks at the call site of whoever wrote the
line, not here. So these types deliberately **do not** support unpacking --
they have no ``__iter__`` and no ``__getitem__``, and adding either would
reintroduce exactly the fragility the flag was added to remove.

Fields hold what the sequence says. Anything that costs a separate computation
-- the echo centres, which need the k-space trajectory -- is a
:func:`~functools.cached_property` that does the work on first read and not
before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np

__all__ = [
    "AdcTimes",
    "BTensor",
    "DiffusionTable",
    "GradientSpectrum",
    "KSpace",
    "Pns",
    "RfPower",
    "RfTimes",
    "SoftDelay",
    "Waveforms",
    "WaveformsAndTimes",
]

#: Pulseq's RF use tags, in the order of the file format's ``e/r/i/s/p/o``
#: trailing character. ``"undefined"`` is what a row without one reads as.
RF_USES = (
    "excitation",
    "refocusing",
    "inversion",
    "saturation",
    "preparation",
    "other",
    "undefined",
)


@dataclass(frozen=True)
class Waveforms:
    """The decompressed gradient waveforms, and the RF if it was asked for.

    Each channel is a ``(2, n)`` array of times in seconds over amplitudes in
    Hz/m, which is upstream's layout. ``rf`` is complex and is ``None`` unless
    ``append_RF`` was set.
    """

    gx: np.ndarray
    gy: np.ndarray
    gz: np.ndarray
    rf: np.ndarray | None = None

    @property
    def channels(self) -> list[np.ndarray]:
        """The gradients as upstream returns them: a list, RF appended if present."""
        channels = [self.gx, self.gy, self.gz]
        if self.rf is not None:
            channels.append(self.rf)
        return channels

    @property
    def duration(self) -> float:
        """The last time any channel carries, in seconds."""
        ends = [channel[0, -1] for channel in self.channels if channel.shape[1]]
        return float(max(ends)) if ends else 0.0


@dataclass(frozen=True)
class RfTimes:
    """Every RF pulse, in play order, with its use tag.

    Upstream buckets RF into excitation and refocusing and drops the other
    five uses on the floor -- an inversion pulse simply does not appear in
    ``waveforms_and_times``. This is one flat table instead, so nothing is
    dropped and a caller who wants upstream's buckets asks for them by name.

    ``freq_offset`` and ``phase_offset`` include the ppm terms, and the phase
    includes the ``2*pi*f*t_centre`` term that carries it to the pulse's
    centre -- the same convention upstream and MATLAB both use.
    """

    t: np.ndarray
    freq_offset: np.ndarray
    phase_offset: np.ndarray
    use: tuple[str, ...]
    block: np.ndarray

    def of(self, *uses: str) -> RfTimes:
        """The pulses carrying any of ``uses``, as another :class:`RfTimes`.

        More than one is accepted because upstream's buckets are not one tag
        each: a pulse whose row carries no use tag reads back as
        ``"undefined"`` and upstream counts it as an excitation, so
        reproducing upstream means asking for both.
        """
        unknown = set(uses) - set(RF_USES)
        if unknown:
            raise ValueError(f"unknown RF use(s) {sorted(unknown)}; expected {RF_USES}")
        wanted = set(uses)
        keep = np.array([tag in wanted for tag in self.use], dtype=bool)
        return RfTimes(
            t=self.t[keep],
            freq_offset=self.freq_offset[keep],
            phase_offset=self.phase_offset[keep],
            use=tuple(np.asarray(self.use, dtype=object)[keep].tolist()),
            block=self.block[keep],
        )

    @property
    def tfp(self) -> np.ndarray:
        """``(3, n)`` of time over frequency over phase -- upstream's packing."""
        return np.vstack((self.t, self.freq_offset, self.phase_offset))

    def __len__(self) -> int:
        return int(self.t.size)


@dataclass(frozen=True)
class AdcTimes:
    """Every ADC sample time, and the offsets that apply to it.

    Two granularities, because upstream and MATLAB disagree about which one
    ``fp_adc`` means and both are worth having:

    - **Per ADC event** -- ``freq_offset``, ``phase_offset``, ``block``,
      ``num_samples``. This is upstream's ``fp_adc``, which is the raw offsets
      off the event with no ppm term folded in.
    - **Per sample** -- ``t``, ``phase_modulation``, ``sample_phase``. The
      last two are MATLAB's ``pm_adc`` and its ``fp_adc`` second row, the phase
      each individual sample is acquired with, ppm and phase modulation and the
      accumulated ``2*pi*f*t`` all included. That is the number a simulation or
      a demodulator actually wants.
    """

    t: np.ndarray
    freq_offset: np.ndarray
    phase_offset: np.ndarray
    phase_modulation: np.ndarray
    sample_phase: np.ndarray
    block: np.ndarray
    num_samples: np.ndarray
    _echoes: Callable[[], tuple[np.ndarray, np.ndarray]] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def fp(self) -> np.ndarray:
        """``(n_events, 2)`` of frequency over phase -- upstream's ``fp_adc``."""
        return np.stack((self.freq_offset, self.phase_offset), axis=-1)

    @cached_property
    def echo_center_index(self) -> np.ndarray:
        """The sample index nearest k-space zero, one per readout.

        Read on first access, not built with the rest: it needs the k-space
        trajectory, and most callers of ``waveforms_and_times`` do not want to
        pay for one.
        """
        return self._echo()[0]

    @cached_property
    def echo_center_time(self) -> np.ndarray:
        """When each readout reaches k-space zero, in seconds."""
        return self._echo()[1]

    def _echo(self) -> tuple[np.ndarray, np.ndarray]:
        if self._echoes is None:
            raise NotImplementedError(
                "echo centres were not derivable for this result -- they need the "
                "k-space trajectory, which is unavailable here."
            )
        return self._echoes()

    def __len__(self) -> int:
        return int(self.t.size)


@dataclass(frozen=True)
class WaveformsAndTimes:
    """Everything :meth:`Sequence.waveforms_and_times` knows.

    Upstream's five-tuple is ``(wave_data, tfp_excitation, tfp_refocusing,
    t_adc, fp_adc)``; the same numbers are ``waveforms.channels``,
    ``rf.of("excitation").tfp``, ``rf.of("refocusing").tfp``, ``adc.t`` and
    ``adc.fp``. What upstream cannot express is the other five RF uses, the
    per-sample ADC phase, and the echo centres.
    """

    waveforms: Waveforms
    rf: RfTimes
    adc: AdcTimes


@dataclass(frozen=True)
class KSpace:
    """Where the sequence goes in k-space.

    ``k_traj`` is upstream's dense trajectory, on the gradient raster and
    computed by upstream. ``k_traj_breakpoints`` is the C core's own answer on
    the gradient *breakpoint* grid: the same curve in five to ten times fewer
    points, with ``t_breakpoints`` as its time base. Upstream's tuple has
    nowhere to put the latter, which is the reason this class exists.
    """

    k_traj_adc: np.ndarray
    k_traj: np.ndarray
    t_excitation: np.ndarray
    t_refocusing: np.ndarray
    t_adc: np.ndarray
    k_traj_breakpoints: np.ndarray
    t_breakpoints: np.ndarray
    k_center: tuple
    readout_center_sample: np.ndarray


@dataclass(frozen=True)
class Pns:
    """Peripheral-nerve-stimulation prediction, as a fraction of threshold.

    ``total`` and ``components`` are upstream's, in percent of the limit.
    ``ok`` is upstream's boolean verdict.
    """

    ok: bool
    total: np.ndarray
    components: np.ndarray
    t: np.ndarray


@dataclass(frozen=True)
class GradientSpectrum:
    """The gradient waveform's spectrum, and any acoustic resonance verdict.

    ``resonance_lines`` is the mechanical-resonance analysis, which upstream
    has no equivalent of. It used to be smuggled out as a fifth tuple element
    when ``resonance_lines=True`` was passed; it is a field here, which is what
    the ``compat`` flag is for.
    """

    spectrograms: list
    frequencies: np.ndarray
    times: np.ndarray
    spectrograms_rss: np.ndarray
    resonance_lines: object | None = None


@dataclass(frozen=True)
class BTensor:
    """The diffusion b-tensor per excitation, in both communities' units.

    MATLAB's ``calcMomentsBtensor`` returns ``B`` in **s/m^2** and leaves the
    step to a diffusion pipeline to the caller. Both live here:

    - ``B``, ``m1``, ``m2``, ``m3`` are MATLAB's, unchanged, so a script
      ported from it reads the same numbers. ``B`` is s/m^2 and ``m_n`` is
      rad/m times seconds to the n-th.
    - ``b_values``, ``b_vectors`` and ``b_tensors`` are **s/mm^2**, which is
      what every diffusion tool means by a b-value.

    ``b_tensors`` is directly what DIPY's ``gradient_table(..., btens=)``
    takes as an ``(N, 3, 3)`` array: the full tensor, whose trace is that
    shot's b-value, *not* normalised to unit trace. For MRtrix3's ``-grad``
    table, ``numpy.column_stack((b_vectors, b_values))`` is the ``(N, 4)``
    ``[x y z b]`` it reads; for ``-fslgrad``, ``b_vectors.T`` and
    ``b_values``. All of it is in the sequence's physical frame, rotation
    extensions resolved, which is the frame the scanner plays and the one a
    scanner-space gradient table is expected in.

    ``b_delta`` is the normalised anisotropy that distinguishes b-tensor
    encodings -- 1 for linear, 0 for spherical, -0.5 for planar -- computed
    from the eigenvalues rather than assumed from the sequence's name.

    **The prescription is not in the sequence, so the tensor comes in three
    parts.** A ``ROTATIONS`` extension is part of the waveform and is already
    resolved in ``B``; the FOV rotation set on the console is not, and
    ``NOROT`` is how a block opts out of it -- which is exactly what a product
    diffusion preparation does, so that the b-matrix does not swing with the
    prescribed slice orientation. The imaging gradients of the same shot are
    not ``NOROT``, though, and their cross terms with the diffusion lobes are
    real, so no single frame describes the shot. Writing @f$R@f$ for the
    prescription::

        B(R) = b_fixed + b_cross @ R.T + R @ b_cross.T + R @ b_rotatable @ R.T

    which :meth:`compose` evaluates. ``B`` is that at ``R = I``, which is what
    MATLAB returns. An all-``NOROT`` shot has ``b_rotatable == 0`` and does not
    move with the prescription at all.

    ``b_tensor_table`` is the distinct tensors in first-seen order and
    ``table_index`` says which row each shot used -- a diffusion scan plays a
    few dozen directions over many thousands of shots, and the short list is
    what a reconstruction and an MRD header want.
    """

    B: np.ndarray
    m1: np.ndarray
    m2: np.ndarray
    m3: np.ndarray
    excitation_times: np.ndarray
    echo_times: np.ndarray
    b_values: np.ndarray
    b_vectors: np.ndarray
    b_tensors: np.ndarray
    eigenvalues: np.ndarray
    b_delta: np.ndarray
    b_fixed: np.ndarray
    b_rotatable: np.ndarray
    b_cross: np.ndarray
    b_tensor_table: np.ndarray
    table_index: np.ndarray

    def compose(self, rotation: np.ndarray) -> np.ndarray:
        """``B`` under a prescription rotation, ``(N, 3, 3)`` in s/m^2.

        Parameters
        ----------
        rotation : np.ndarray
            The console's FOV rotation as a ``(3, 3)`` matrix taking logical
            axes to physical ones.
        """
        R = np.asarray(rotation, dtype=float).reshape(3, 3)
        cross = self.b_cross @ R.T
        return self.b_fixed + cross + np.swapaxes(cross, -1, -2) + R @ self.b_rotatable @ R.T

    @classmethod
    def of(
        cls,
        *,
        B: np.ndarray,
        m1: np.ndarray,
        m2: np.ndarray,
        m3: np.ndarray,
        excitation_times: np.ndarray,
        echo_times: np.ndarray,
        b_fixed: np.ndarray | None = None,
        b_rotatable: np.ndarray | None = None,
        b_cross: np.ndarray | None = None,
        b_tensor_table: np.ndarray | None = None,
        table_index: np.ndarray | None = None,
    ) -> BTensor:
        """Derive the diffusion-facing fields from the tensors."""
        # s/m^2 to s/mm^2. The tensor is symmetric by construction, so eigh
        # is the right decomposition and its eigenvalues come out ascending.
        tensors = np.asarray(B, dtype=float) * 1e-6
        values, vectors = np.linalg.eigh(tensors) if tensors.size else (
            np.zeros((0, 3)),
            np.zeros((0, 3, 3)),
        )
        traces = np.trace(tensors, axis1=-2, axis2=-1)

        # The principal direction, with a fixed sign: a b-vector and its
        # negative describe the same measurement, and a table that flips
        # between them for numerical reasons is a table nobody can diff.
        principal = vectors[..., -1] if tensors.size else np.zeros((0, 3))
        leading = np.argmax(np.abs(principal), axis=-1) if tensors.size else np.zeros(0, dtype=int)
        if tensors.size:
            signs = np.sign(np.take_along_axis(principal, leading[:, None], axis=-1))
            signs[signs == 0] = 1.0
            principal = principal * signs

        with np.errstate(invalid="ignore", divide="ignore"):
            # Topgaard's b_delta: the eigenvalue furthest from the mean is the
            # axial one, and the other two average to the radial one.
            mean = traces / 3.0
            axial = np.take_along_axis(
                values, np.argmax(np.abs(values - mean[:, None]), axis=-1)[:, None], axis=-1
            ).squeeze(-1) if tensors.size else np.zeros(0)
            radial = (traces - axial) / 2.0
            delta = np.where(traces > 0, (axial - radial) / np.where(traces > 0, traces, 1.0), 0.0)

        return cls(
            B=np.asarray(B, dtype=float),
            m1=np.asarray(m1, dtype=float),
            m2=np.asarray(m2, dtype=float),
            m3=np.asarray(m3, dtype=float),
            excitation_times=np.asarray(excitation_times, dtype=float),
            echo_times=np.asarray(echo_times, dtype=float),
            b_values=traces,
            b_vectors=principal,
            b_tensors=tensors,
            eigenvalues=values,
            b_delta=delta,
            b_fixed=_or_zeros(b_fixed, B),
            b_rotatable=_or_zeros(b_rotatable, B),
            b_cross=_or_zeros(b_cross, B),
            b_tensor_table=_or_zeros(b_tensor_table, B),
            table_index=(
                np.arange(len(np.asarray(B)))
                if table_index is None
                else np.asarray(table_index, dtype=int)
            ),
        )


def _or_zeros(value: np.ndarray | None, like: np.ndarray) -> np.ndarray:
    """@p value as float64, or a zeroed array shaped like @p like."""
    if value is None:
        return np.zeros_like(np.asarray(like, dtype=float))
    return np.asarray(value, dtype=float)


@dataclass(frozen=True)
class DiffusionTable:
    """A scan's distinct diffusion encodings, and the counter that selects one.

    The short, consumer-facing form of :class:`BTensor`: one row per *distinct*
    encoding rather than one per shot, since a diffusion scan plays a few dozen
    directions over many thousands of shots and every tool downstream indexes
    by volume.

    ``axis`` names the MRD counter whose value is the row index -- ``"SET"``,
    ``"ECO"``, whichever the design's :class:`~pulserver.ScanLoop` declared for
    its diffusion dimension. It is carried rather than guessed because nothing
    in the tensors themselves says which counter moved with them.

    Units are the diffusion community's throughout: ``b_tensors`` is
    ``(N, 3, 3)`` in **s/mm^2** with trace equal to the b-value, which is
    exactly what DIPY's ``gradient_table(..., btens=)`` takes -- the full
    tensor, not normalised to unit trace. MRtrix3's ``-grad`` table is
    ``numpy.column_stack((b_vectors, b_values))``; for ``-fslgrad`` it is
    ``b_vectors.T`` and ``b_values``.

    Two producers build this and they must agree: the design side, from
    :meth:`BTensor.diffusion_table`, and the reconstruction side, from the MRD
    header the scanner wrote. A second type for the second producer is the
    mistake this one exists to avoid.
    """

    b_tensors: np.ndarray
    b_values: np.ndarray
    b_vectors: np.ndarray
    b_delta: np.ndarray
    axis: str

    @classmethod
    def of(cls, tensors: np.ndarray, *, axis: str) -> DiffusionTable:
        """Derive the b-values, b-vectors and shape from ``(N, 3, 3)`` s/m^2."""
        derived = BTensor.of(
            B=np.asarray(tensors, dtype=float).reshape(-1, 3, 3),
            m1=np.zeros((0, 3)),
            m2=np.zeros((0, 3)),
            m3=np.zeros((0, 3)),
            excitation_times=np.zeros(0),
            echo_times=np.zeros(0),
        )
        return cls(
            b_tensors=derived.b_tensors,
            b_values=derived.b_values,
            b_vectors=derived.b_vectors,
            b_delta=derived.b_delta,
            axis=str(axis),
        )

    @classmethod
    def from_definitions(
        cls, definitions: dict, *, rotation: np.ndarray | None = None
    ) -> DiffusionTable:
        """Read back what ``write_diffusion_definitions`` wrote.

        The one parser, used by both readers: the design side, from a
        sequence's ``[DEFINITIONS]``, and the reconstruction side, from the MRD
        header the scanner copied them into. Values may be lists of numbers or
        the whitespace-joined strings an ISMRMRD ``userParameterString`` holds
        -- the wire form is a detail of how they travelled, not of what they
        mean.

        Parameters
        ----------
        definitions : dict
            Keyed by ``bTensorFixed`` / ``bTensorRotatable`` / ``bTensorCross``
            / ``bTensorAxis``. Only ``bTensorFixed`` and ``bTensorAxis`` are
            required; a part that was identically zero is not written.
        rotation : np.ndarray, optional
            The console's FOV rotation as ``(3, 3)``. Without it the table is
            in the frame the ``.seq`` describes, which is the right answer only
            for a preparation played entirely under ``NOROT``.
        """

        def rows(key):
            value = definitions.get(key)
            if value is None:
                return None
            if isinstance(value, str):
                value = value.split()
            return np.asarray([float(v) for v in value], dtype=float).reshape(-1, 3, 3)

        fixed = rows("bTensorFixed")
        if fixed is None or "bTensorAxis" not in definitions:
            raise ValueError(
                "DiffusionTable.from_definitions(): no bTensorFixed/bTensorAxis entry, so "
                "this sequence carries no diffusion gradient table. Call "
                "Sequence.write_diffusion_definitions() on the design side."
            )

        rotatable = rows("bTensorRotatable")
        cross = rows("bTensorCross")
        R = np.eye(3) if rotation is None else np.asarray(rotation, float).reshape(3, 3)

        tensors = np.array(fixed)
        if rotatable is not None:
            tensors = tensors + R @ rotatable @ R.T
        if cross is not None:
            turned = cross @ R.T
            tensors = tensors + turned + np.swapaxes(turned, -1, -2)

        axis = definitions["bTensorAxis"]
        return cls.of(tensors, axis=axis if isinstance(axis, str) else str(axis))

    def mrtrix_table(self) -> np.ndarray:
        """``(N, 4)`` of ``[x y z b]``, which is MRtrix3's ``-grad``."""
        return np.column_stack((self.b_vectors, self.b_values))


@dataclass(frozen=True)
class RfPower:
    """RF energy and power over a window, in Pulseq's relative units.

    Amplitudes are in Hz, so ``mean_power`` and ``peak_power`` are Hz^2 and
    ``total_energy`` is Hz^2 s. Dividing by ``gamma`` (and ``gamma**2``) gives
    Tesla and mT^2 s; **none of them is SAR**, which needs the electric field
    and so depends on the transmit coil and on the subject.

    ``duration`` is the window actually walked, and ``window_duration`` is the
    sliding window ``mean_power`` and ``rf_rms`` were maximised over, or
    ``None`` when they are averages over the whole of it. ``total_energy`` is
    always the whole window's, never one slice of it.
    """

    mean_power: float
    peak_power: float
    rf_rms: float
    total_energy: float
    duration: float
    window_duration: float | None = None


@dataclass(frozen=True)
class SoftDelay:
    """One named soft delay: what it is set to, and what it may be set to.

    MATLAB's ``getDefaultSoftDelayValues`` returns the values in a struct and
    the limits in a parallel cell array. Here they travel together, and
    ``__float__`` makes the object usable wherever the value alone was wanted
    --- ``seq.apply_soft_delay(TE=float(defaults["TE"]))``.

    The limits are where the block duration would reach zero, which is the
    only bound the sequence itself states. A value inside them can still be
    impossible for a different reason -- events that no longer fit the block --
    and neither this nor MATLAB catches that.
    """

    hint: str
    numeric_id: int
    value: float
    minimum: float
    maximum: float

    def __float__(self) -> float:
        return float(self.value)
