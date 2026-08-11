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
    "GradientSpectrum",
    "KSpace",
    "Pns",
    "RfTimes",
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
