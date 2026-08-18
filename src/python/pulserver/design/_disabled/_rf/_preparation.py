"""Magnetization-preparation and special-purpose RF pulse factories."""

from __future__ import annotations

import copy
import math

import numpy as np
import pypulseq as pp

from ...pypulseq._make_label import make_label
from ...pypulseq._opts import default_system
from .._gradients import make_spoiler
from .._system import round_to_raster
from ._base import RfModule, RfPulse
from ._excitation import make_frequency_selective_pulse, make_hard_pulse, make_refocusing_pulse

FAT_SHIFT_PPM = -3.45


#: FOV-transform exemptions every magnetization preparation carries: the
#: pulse takes its frequency offset and orientation from its own design, so
#: the interpreter must not reposition or rotate it with the imaging FOV.
#: Declared as module flags at construction, which scopes them to the module
#: and resets them on its last block.
FOV_EXEMPT_FLAGS = ("NOPOS", "NOROT")


def _labels(value: int) -> tuple:
    return tuple(make_label(label=name, type="SET", value=value) for name in FOV_EXEMPT_FLAGS)


def fat_shift_hz(b0: float, gamma: float = 42.576e6) -> float:
    """Return the main fat resonance relative to water, in Hz."""
    return FAT_SHIFT_PPM * 1e-6 * b0 * gamma


class FatSaturationPulse(RfModule):
    def __init__(self, system, rf, spoiler: tuple, labels_on: tuple, labels_off: tuple, duration: float):
        self.rf = rf
        self.spoiler = spoiler
        self.labels_on = labels_on
        self.labels_off = labels_off
        self.duration = duration
        super().__init__(system, ((rf,), spoiler), flags=dict.fromkeys(FOV_EXEMPT_FLAGS, 1))


def make_fat_saturation_pulse(
    *,
    freq_offset: float | None = None,
    b0: float | None = None,
    flip_angle: float = np.deg2rad(110.0),
    bandwidth: float = 250.0,
    time_bw_product: float = 2.0,
    voxel_size: float,
    spoil_cycles: float = 4.0,
    system: pp.Opts | None = None,
) -> FatSaturationPulse:
    """Create a fat-saturation module: spectral pulse plus 3-axis spoiler.

    Saturates the fat resonance and immediately dephases it, so the imaging
    excitation that follows sees only water. The flip angle exceeds 90 degrees
    (110 by default) to leave fat slightly past the transverse plane, which is
    more robust to B1 shortfall than an exact 90.

    The two blocks are bracketed by ``NOPOS``/``NOROT`` labels: the module is
    frequency-selective, not spatially selective, so it must be excluded from
    the imaging FOV position and rotation transform.

    Give either ``freq_offset`` directly or ``b0``, from which the -3.45 ppm
    fat shift is computed.

    Parameters
    ----------
    freq_offset : float or None, optional
        Fat offset relative to water (Hz). Mutually exclusive with ``b0``.
    b0 : float or None, optional
        Main field (T), used to derive the fat offset.
    flip_angle : float, optional
        Saturation flip angle (rad); 110 degrees by default.
    bandwidth : float, optional
        Spectral passband (Hz).
    time_bw_product : float, optional
        Time-bandwidth product.
    voxel_size : float
        Reference length (m) the spoiler dephases across.
    spoil_cycles : float, optional
        Dephasing cycles per ``voxel_size``.
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    FatSaturationPulse
        Two-block module: ``(rf, labels on)`` then ``(spoiler, labels off)``;
        total length in ``.duration``.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> fatsat = design.make_fat_saturation_pulse(b0=3.0, voxel_size=2e-3)
    >>> len(fatsat)
    2
    >>> round(fatsat.duration * 1e3, 2)
    9.42

    Append it once per TR, before the excitation::

        for module in (fatsat, excitation):
            for block in module:
                seq.add_block(*block)

    Saturation is confined to the fat resonance and leaves water untouched:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       from _figures import preparation_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       preparation_figure(design.make_fat_saturation_pulse(b0=3.0, voxel_size=1e-3, system=system),
                          span=800, title="fat saturation at 3 T")

    See Also
    --------
    make_frequency_selective_pulse : the underlying spectral pulse.
    make_spsp_pulse : spectral-spatial alternative that needs no spoiler.
    """
    system = default_system(system)
    if freq_offset is None:
        if b0 is None:
            raise ValueError("provide either freq_offset or b0")
        freq_offset = fat_shift_hz(b0, system.gamma)
    rf = make_frequency_selective_pulse(
        flip_angle,
        bandwidth,
        freq_offset=freq_offset,
        time_bw_product=time_bw_product,
        pulse_type="sat",
        system=system,
        use="saturation",
    ).rf
    spoiler = make_spoiler(system, voxel_size, dephasing_cycles=spoil_cycles)
    return FatSaturationPulse(
        system=system,
        rf=rf,
        spoiler=spoiler,
        labels_on=_labels(1),
        labels_off=_labels(0),
        duration=pp.calc_duration(rf) + pp.calc_duration(*spoiler),
    )


class MTPulse(RfModule):
    def __init__(self, system, rf, spoiler: tuple, duration: float):
        self.rf = rf
        self.spoiler = spoiler
        self.duration = duration
        super().__init__(system, ((rf,), spoiler))


def make_mt_pulse(
    *,
    flip_angle: float = np.deg2rad(500.0),
    freq_offset: float = -1500.0,
    duration: float = 8e-3,
    voxel_size: float,
    spoil_cycles: float = 4.0,
    system: pp.Opts | None = None,
) -> MTPulse:
    """Create a magnetization-transfer saturation module.

    A long, high-flip Gaussian pulse placed far off resonance saturates the
    broad macromolecular pool without directly exciting free water; the
    saturation then transfers to water, reducing its signal in proportion to
    the local MT effect. A spoiler follows to remove any residual transverse
    magnetization.

    Parameters
    ----------
    flip_angle : float, optional
        Saturation flip angle (rad); 500 degrees by default.
    freq_offset : float, optional
        Off-resonance offset (Hz); must be non-zero.
    duration : float, optional
        Pulse duration (s).
    voxel_size : float
        Reference length (m) the spoiler dephases across.
    spoil_cycles : float, optional
        Dephasing cycles per ``voxel_size``.
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    MTPulse
        Two-block module: RF, then spoiler; total length in ``.duration``.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> mt = design.make_mt_pulse(voxel_size=2e-3)
    >>> len(mt)
    2

    Acquire an MT-on / MT-off pair by amplitude-gating the same module::

        for scale in (1.0, 0.0):
            mt.set_state(amplitude_scale=scale)
            for block in mt:
                seq.add_block(*block)

    The saturation band sits far off resonance, where only the bound pool absorbs:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       from _figures import preparation_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       preparation_figure(design.make_mt_pulse(voxel_size=1e-3, system=system),
                          span=3000, title="MT saturation, -1500 Hz offset")

    See Also
    --------
    make_ihmt_pulse : dual-offset variant isolating the inhomogeneous pool.
    """
    system = default_system(system)
    if freq_offset == 0:
        raise ValueError("freq_offset must be non-zero for MT")
    rf = pp.make_gauss_pulse(
        flip_angle=flip_angle,
        duration=duration,
        freq_offset=freq_offset,
        time_bw_product=4.0,
        system=system,
        use="saturation",
    )
    spoiler = make_spoiler(system, voxel_size, dephasing_cycles=spoil_cycles)
    return MTPulse(
        system=system,
        rf=rf,
        spoiler=spoiler,
        duration=pp.calc_duration(rf) + pp.calc_duration(*spoiler),
    )


class IhMTPulse(RfModule):
    def __init__(self, system, rf_positive, rf_negative, spoiler: tuple, repetitions: int, duration: float):
        self.rf_positive = rf_positive
        self.rf_negative = rf_negative
        self.spoiler = spoiler
        self.repetitions = repetitions
        self.duration = duration
        pulses = tuple((rf_positive if index % 2 == 0 else rf_negative,) for index in range(repetitions))
        super().__init__(system, (*pulses, spoiler))


def make_ihmt_pulse(
    *,
    flip_angle: float = np.deg2rad(250.0),
    freq_offset: float = 7000.0,
    duration: float = 6e-3,
    repetitions: int = 4,
    voxel_size: float,
    spoil_cycles: float = 4.0,
    system: pp.Opts | None = None,
) -> IhMTPulse:
    """Create an inhomogeneous-MT pulse train alternating +/- offset.

    Inhomogeneous MT isolates the dipolar-order contribution by comparing
    saturation at a *single* offset with saturation alternating between ``+f``
    and ``-f``. This factory builds the alternating (dual-offset) arm: one
    Gaussian per repetition, sign-flipped each time, then a single spoiler.
    Pair it with :func:`make_mt_pulse` at the same offset and total power for
    the single-offset arm.

    Parameters
    ----------
    flip_angle : float, optional
        Per-pulse flip angle (rad); 250 degrees by default.
    freq_offset : float, optional
        Offset magnitude (Hz); must be positive — both signs are generated.
    duration : float, optional
        Duration of each pulse in the train (s).
    repetitions : int, optional
        Number of pulses in the train (>= 2).
    voxel_size : float
        Reference length (m) the spoiler dephases across.
    spoil_cycles : float, optional
        Dephasing cycles per ``voxel_size``.
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    IhMTPulse
        Module of ``repetitions`` RF blocks followed by the spoiler; total
        length in ``.duration``.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> ihmt = design.make_ihmt_pulse(voxel_size=2e-3, repetitions=4)
    >>> len(ihmt), ihmt.repetitions
    (5, 4)
    >>> ihmt.rf_positive.freq_offset, ihmt.rf_negative.freq_offset
    (7000.0, -7000.0)

    Append the alternating RF train and its final spoiler::

        seq = pp.Sequence(ihmt.system)
        for block in ihmt:
            seq.add_block(*block)

    The module's blocks, as they are played:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import preparation_figure
       preparation_figure(design.make_ihmt_pulse(voxel_size=1e-3, system=system), span=10000)

    See Also
    --------
    make_mt_pulse : the single-offset reference arm.
    """
    system = default_system(system)
    if freq_offset <= 0:
        raise ValueError("freq_offset must be > 0; both signs are generated")
    if repetitions < 2:
        raise ValueError("repetitions must be >= 2")
    rf_positive = pp.make_gauss_pulse(
        flip_angle=flip_angle,
        duration=duration,
        freq_offset=freq_offset,
        time_bw_product=4.0,
        system=system,
        use="saturation",
    )
    rf_negative = copy.deepcopy(rf_positive)
    rf_negative.freq_offset = -freq_offset
    spoiler = make_spoiler(system, voxel_size, dephasing_cycles=spoil_cycles)
    total = repetitions * pp.calc_duration(rf_positive) + pp.calc_duration(*spoiler)
    return IhMTPulse(system, rf_positive, rf_negative, spoiler, repetitions, total)


class BlochSiegertPulse(RfPulse):
    def __init__(self, system, rf, kbs: float, kbs_per_gauss2: float):
        self.kbs = kbs
        self.kbs_per_gauss2 = kbs_per_gauss2
        super().__init__(system, rf)


def make_bloch_siegert_pulse(
    *,
    freq_offset: float = 4000.0,
    duration: float = 8e-3,
    peak_b1: float = 500.0,
    flat_fraction: float = 0.8,
    transition_fraction: float = 0.02,
    dwell: float = 10e-6,
    system: pp.Opts | None = None,
) -> BlochSiegertPulse:
    """Create a Bloch-Siegert B1-mapping pulse and its calibration constants.

    A strong off-resonant Fermi pulse shifts the water resonance by an amount
    proportional to ``|B1|^2``; acquiring two images with ``+f`` and ``-f``
    turns that shift into a phase difference, and hence a B1 map. The pulse
    itself deposits the shift — the readout is an ordinary one.

    The returned module carries the calibration constants the reconstruction
    needs: ``kbs`` (rad per unit squared Pulseq amplitude) and
    ``kbs_per_gauss2`` (rad/G^2, the conventional reporting unit).

    Parameters
    ----------
    freq_offset : float, optional
        Off-resonance offset (Hz); must be non-zero. Flip its sign for the
        second acquisition of the pair.
    duration : float, optional
        Pulse duration (s).
    peak_b1 : float, optional
        Peak amplitude in Pulseq units (Hz).
    flat_fraction : float, optional
        Fraction of the duration held at ``peak_b1``, in ``(0, 1)``.
    transition_fraction : float, optional
        Fermi transition width as a fraction of the duration.
    dwell : float, optional
        RF raster (s).
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    BlochSiegertPulse
        Single-block module exposing ``.kbs`` and ``.kbs_per_gauss2``.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> pulse = design.make_bloch_siegert_pulse(freq_offset=4000.0)
    >>> round(pulse.kbs_per_gauss2, 1)
    86.6

    Acquire the sign-reversed pair::

        for offset in (4000.0, -4000.0):
            pulse = design.make_bloch_siegert_pulse(freq_offset=offset)
            for module in (pulse, readout):
                for block in module:
                    seq.add_block(*block)

    The off-resonant pulse that imprints the B1-dependent phase shift:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       from _figures import rf_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       rf_figure(design.make_bloch_siegert_pulse(system=system),
                 "none", title="Bloch-Siegert pulse, +4 kHz")

    References
    ----------
    Sacolick et al., Bloch-Siegert B1 mapping, DOI ``10.1002/mrm.22357``.
    """
    system = default_system(system)
    if freq_offset == 0:
        raise ValueError("freq_offset must be non-zero")
    if duration <= 0 or peak_b1 <= 0 or dwell <= 0:
        raise ValueError("duration, peak_b1, and dwell must be > 0")
    if not 0 < flat_fraction < 1 or transition_fraction <= 0:
        raise ValueError("flat_fraction must lie in (0, 1) and transition_fraction must be > 0")
    n = max(2, round(duration / dwell))
    actual_duration = n * dwell
    time = (np.arange(n) + 0.5) * dwell
    half_flat = 0.5 * flat_fraction * actual_duration
    transition = transition_fraction * actual_duration
    envelope = peak_b1 / (1.0 + np.exp((np.abs(time - actual_duration / 2.0) - half_flat) / transition))
    rf = pp.make_arbitrary_rf(
        signal=envelope,
        flip_angle=1.0,
        no_signal_scaling=True,
        dwell=dwell,
        freq_offset=freq_offset,
        system=system,
        use="preparation",
    )
    denominator = 2.0 * 2.0 * np.pi * freq_offset
    kbs = float(np.sum((2.0 * np.pi * envelope) ** 2 / denominator) * dwell)
    normalized = envelope / peak_b1
    gamma_hz_per_gauss = system.gamma * 1e-4
    kbs_per_gauss2 = float(np.sum((2.0 * np.pi * gamma_hz_per_gauss * normalized) ** 2 / denominator) * dwell)
    return BlochSiegertPulse(system, rf, kbs, kbs_per_gauss2)


def _half_passages(system: pp.Opts, duration: float, dwell: float) -> tuple:
    full = pp.make_adiabatic_pulse(
        pulse_type="hypsec",
        duration=2.0 * duration,
        dwell=dwell,
        system=system,
        use="preparation",
    )
    half = np.asarray(full.signal[: full.signal.size // 2])
    down = pp.make_arbitrary_rf(
        half,
        flip_angle=np.pi / 2.0,
        no_signal_scaling=True,
        dwell=dwell,
        system=system,
        use="preparation",
    )
    up = pp.make_arbitrary_rf(
        np.conj(half[::-1]),
        flip_angle=np.pi / 2.0,
        no_signal_scaling=True,
        dwell=dwell,
        system=system,
        use="preparation",
    )
    return down, up


class T2PrepPulse(RfModule):
    def __init__(
        self,
        system,
        rf90_down,
        rf180,
        rf90_up,
        delay1: float,
        delay2: float,
        spoiler: tuple,
        echo_time: float,
        duration: float,
        final_tip: str,
    ):
        self.rf90_down = rf90_down
        self.rf180 = rf180
        self.rf90_up = rf90_up
        self.delay1 = delay1
        self.delay2 = delay2
        self.spoiler = spoiler
        self.echo_time = float(echo_time)
        self.duration = duration
        self.final_tip = final_tip
        blocks: list[tuple] = [(self.rf90_down,)]
        if self.delay1 > 0:
            blocks.append((pp.make_delay(self.delay1),))
        blocks.append((self.rf180,))
        if self.delay2 > 0:
            blocks.append((pp.make_delay(self.delay2),))
        blocks.extend(((self.rf90_up,), self.spoiler))
        super().__init__(system, blocks)


def make_t2prep_pulse(
    echo_time: float,
    *,
    adiabatic: bool = True,
    final_tip: str = "up",
    voxel_size: float,
    spoil_cycles: float = 4.0,
    half_passage_duration: float = 4e-3,
    refocusing_duration: float = 10.24e-3,
    dwell: float = 10e-6,
    system: pp.Opts | None = None,
) -> T2PrepPulse:
    """Create a T2 preparation module (90 - 180 - 90 with a terminating spoiler).

    Magnetization is tipped into the transverse plane, allowed to decay for
    ``echo_time``, and restored to the longitudinal axis — so the *stored*
    magnetization carries T2 weighting into whatever readout follows,
    independent of that readout's own TE. Inter-pulse delays are solved from
    the actual pulse centres and rounded to the gradient raster.

    ``final_tip='up'`` (default) restores to ``+z``: pure T2 preparation.
    ``'down'`` restores to ``-z``, which additionally starts a T1 recovery and
    gives the hybrid T1/T2 weighting exposed as
    :func:`make_t1t2_prep_pulse`.

    Adiabatic half-passages and an adiabatic 180 are used by default, for
    B1 robustness; ``adiabatic=False`` swaps in short hard/SLR pulses.

    Parameters
    ----------
    echo_time : float
        Preparation echo time (s), from the first pulse centre to the last.
    adiabatic : bool, optional
        Use adiabatic half-passages and refocusing (default True).
    final_tip : {'up', 'down'}, optional
        Restore to ``+z`` or ``-z``.
    voxel_size : float
        Reference length (m) the spoiler dephases across.
    spoil_cycles : float, optional
        Dephasing cycles per ``voxel_size``.
    half_passage_duration : float, optional
        Duration of each adiabatic half-passage (s).
    refocusing_duration : float, optional
        Duration of the adiabatic refocusing pulse (s).
    dwell : float, optional
        RF raster (s).
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    T2PrepPulse
        Module holding the three pulses, their delays, and the spoiler; total
        length in ``.duration``.

    Raises
    ------
    ValueError
        If ``echo_time`` is too short for the requested pulses.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> prep = design.make_t2prep_pulse(50e-3, voxel_size=2e-3)
    >>> prep.final_tip
    'up'
    >>> round(prep.duration * 1e3, 1)
    59.4

    Sweep the preparation TE to fit T2::

        for te in (0.0, 30e-3, 60e-3, 90e-3):
            prep = design.make_t2prep_pulse(te, voxel_size=2e-3)
            for module in (prep, readout_train):
                for block in module:
                    seq.add_block(*block)

    The module's blocks, as they are played:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import preparation_figure
       preparation_figure(
           design.make_t2prep_pulse(50e-3, voxel_size=1e-3, system=system),
           t2_values=np.linspace(20e-3, 160e-3, 121),
           title="T2 preparation, 50 ms")

    See Also
    --------
    make_t1t2_prep_pulse : the ``final_tip='down'`` hybrid.
    """
    system = default_system(system)
    if final_tip not in ("up", "down"):
        raise ValueError("final_tip must be 'up' or 'down'")
    if adiabatic:
        rf90_down, rf90_up = _half_passages(system, half_passage_duration, dwell)
        rf180 = pp.make_adiabatic_pulse(
            pulse_type="hypsec",
            duration=refocusing_duration,
            dwell=dwell,
            system=system,
            use="refocusing",
        )
    else:
        rf90_down = make_hard_pulse(np.pi / 2.0, duration=0.5e-3, system=system, use="preparation").rf
        rf90_up = make_hard_pulse(np.pi / 2.0, duration=0.5e-3, system=system, use="preparation").rf
        rf180 = make_refocusing_pulse(duration=1e-3, system=system).rf

    rf90_up.phase_offset += np.pi if final_tip == "up" else 0.0
    half_te = echo_time / 2.0
    d90 = pp.calc_duration(rf90_down)
    c90 = pp.calc_rf_center(rf90_down)[0]
    d180 = pp.calc_duration(rf180)
    c180 = pp.calc_rf_center(rf180)[0]
    c90_up = pp.calc_rf_center(rf90_up)[0]
    delay1 = round_to_raster(half_te - (d90 - c90) - c180)
    delay2 = round_to_raster(half_te - (d180 - c180) - c90_up)
    if delay1 < -1e-9 or delay2 < -1e-9:
        raise ValueError("echo_time is too short for the requested T2-prep pulses")
    delay1, delay2 = max(0.0, delay1), max(0.0, delay2)
    spoiler = make_spoiler(system, voxel_size, dephasing_cycles=spoil_cycles)
    total = d90 + delay1 + d180 + delay2 + pp.calc_duration(rf90_up) + pp.calc_duration(*spoiler)
    return T2PrepPulse(system, rf90_down, rf180, rf90_up, delay1, delay2, spoiler, echo_time, total, final_tip)


def make_t1t2_prep_pulse(
    echo_time: float,
    *,
    voxel_size: float,
    adiabatic: bool = True,
    spoil_cycles: float = 4.0,
    half_passage_duration: float = 4e-3,
    refocusing_duration: float = 10.24e-3,
    dwell: float = 10e-6,
    system: pp.Opts | None = None,
) -> T2PrepPulse:
    """Create hybrid T1/T2 preparation, storing the magnetization on -z.

    :func:`make_t2prep_pulse` with ``final_tip='down'``. Restoring to ``-z``
    rather than ``+z`` means the stored magnetization both carries the T2
    weighting accrued during ``echo_time`` *and* recovers through the null
    afterwards, so a single module yields joint T1/T2 contrast — no separate
    inversion pulse is needed.

    Parameters
    ----------
    echo_time : float
        Preparation echo time (s).
    voxel_size : float
        Reference length (m) the spoiler dephases across.
    adiabatic : bool, optional
        Use adiabatic half-passages and refocusing.
    spoil_cycles : float, optional
        Dephasing cycles per ``voxel_size``.
    half_passage_duration : float, optional
        Duration of each adiabatic half-passage (s).
    refocusing_duration : float, optional
        Duration of the adiabatic refocusing pulse (s).
    dwell : float, optional
        RF sample spacing (s).
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    T2PrepPulse
        Module with ``final_tip == 'down'``.

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> prep = design.make_t1t2_prep_pulse(50e-3, voxel_size=2e-3)
    >>> prep.final_tip
    'down'

    Append all preparation blocks::

        seq = pp.Sequence(prep.system)
        for block in prep:
            seq.add_block(*block)

    The final tip stores the T2-weighted magnetization on ``-z``:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       from _figures import preparation_figure
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       preparation_figure(
           design.make_t1t2_prep_pulse(50e-3, voxel_size=1e-3, system=system),
           t2_values=np.linspace(20e-3, 160e-3, 121),
           title="T1/T2 preparation, 50 ms")

    See Also
    --------
    make_t2prep_pulse : the general form.
    make_inversion_pulse : pure T1 preparation.
    """
    return make_t2prep_pulse(
        echo_time,
        adiabatic=adiabatic,
        final_tip="down",
        voxel_size=voxel_size,
        spoil_cycles=spoil_cycles,
        half_passage_duration=half_passage_duration,
        refocusing_duration=refocusing_duration,
        dwell=dwell,
        system=system,
    )


class DiffusionPrepPulse(RfModule):
    def __init__(
        self,
        system,
        rf90_down,
        rf180,
        rf90_up,
        gradient,
        delay1: float,
        delay2: float,
        spoiler: tuple,
        labels_on: tuple,
        labels_off: tuple,
        b_value: float,
        duration: float,
    ):
        self.rf90_down = rf90_down
        self.rf180 = rf180
        self.rf90_up = rf90_up
        self.gradient = gradient
        self.delay1 = delay1
        self.delay2 = delay2
        self.spoiler = spoiler
        self.labels_on = labels_on
        self.labels_off = labels_off
        self.b_value = b_value
        self.duration = duration
        self._requested_b_value = b_value
        blocks: list[tuple] = [(self.rf90_down,), (self.gradient,)]
        gradient_block_indices = [1]
        if self.delay1 > 0:
            blocks.append((pp.make_delay(self.delay1),))
        blocks.append((self.rf180,))
        if self.delay2 > 0:
            blocks.append((pp.make_delay(self.delay2),))
        gradient_block_indices.append(len(blocks))
        blocks.extend(((self.gradient,), (self.rf90_up,), self.spoiler))
        self._diffusion_gradient_block_indices = tuple(gradient_block_indices)
        super().__init__(system, blocks, flags=dict.fromkeys(FOV_EXEMPT_FLAGS, 1))

    def _set_state(
        self,
        *,
        b_value: float | None = None,
        rotation=None,
        freq_offset_hz: float = 0.0,
        phase_offset_rad: float = 0.0,
        amplitude_scale: float = 1.0,
    ):
        """Set requested b-value/direction and optional RF offsets for this use."""
        self._requested_b_value = self.b_value if b_value is None else float(b_value)
        self.gradient_scale(self._requested_b_value)
        return super()._set_state(
            rotation=rotation,
            freq_offset_hz=freq_offset_hz,
            phase_offset_rad=phase_offset_rad,
            amplitude_scale=amplitude_scale,
        )

    def _build_blocks(self) -> tuple[tuple, ...]:
        blocks = [list(block) for block in super()._build_blocks()]
        scale = self.gradient_scale(self._requested_b_value)
        for index in self._diffusion_gradient_block_indices:
            blocks[index][0] = pp.scale_grad(blocks[index][0], scale, system=self.system)
        return tuple(tuple(block) for block in blocks)

    @property
    def requested_b_value(self) -> float:
        return self._requested_b_value

    def gradient_scale(self, b_value: float) -> float:
        """Return the gradient amplitude scale for a lower requested b-value."""
        if not 0 <= b_value <= self.b_value:
            raise ValueError("b_value must lie between zero and the design b_value")
        return math.sqrt(b_value / self.b_value) if self.b_value else 0.0

    def gradient_for_b_value(self, b_value: float):
        """Return a copied z gradient scaled for ``b_value``."""
        return pp.scale_grad(self.gradient, self.gradient_scale(b_value))


def _trapezoid_samples(event, dwell: float) -> np.ndarray:
    count = round(pp.calc_duration(event) / dwell)
    time = (np.arange(count) + 0.5) * dwell
    knots = np.asarray(
        [
            0.0,
            event.rise_time,
            event.rise_time + event.flat_time,
            event.rise_time + event.flat_time + event.fall_time,
        ]
    )
    return np.interp(time, knots, [0.0, event.amplitude, event.amplitude, 0.0])


def _diffusion_pair_b_value(event, middle_duration: float, dwell: float) -> float:
    lobe = _trapezoid_samples(event, dwell)
    middle_count = round(middle_duration / dwell)
    # The 180 reverses the sign of the second physical lobe in the toggling
    # frame. event amplitudes are Hz/m, so 2*pi converts q to rad/m.
    effective_gradient = np.concatenate((lobe, np.zeros(middle_count), -lobe))
    q_rad_per_m = 2.0 * np.pi * np.cumsum(effective_gradient) * dwell
    return float(np.sum(q_rad_per_m**2) * dwell / 1e6)


def make_diffusion_prep(
    b_value: float,
    *,
    gradient_duration: float = 20e-3,
    gradient_separation: float = 30e-3,
    voxel_size: float,
    spoil_cycles: float = 4.0,
    system: pp.Opts | None = None,
) -> DiffusionPrepPulse:
    """Create a diffusion preparation along z, at its full design b-value.

    A stimulated-echo-style preparation (90 - G - 180 - G - 90) that stores
    diffusion-weighted magnetization on the longitudinal axis, so any readout
    can follow. Exactly **one** canonical module is designed, along z and at
    the requested ``b_value``; directions and lower b-values are obtained from
    it rather than by redesigning:

    - **direction** — attach a rotation extension to the gradient blocks
      (:meth:`~pulserver.ScanLoop.to_rotations`, then ``set_state(rotation=...)``);
    - **b-value** — :meth:`DiffusionPrepPulse.gradient_for_b_value`, or
      ``set_state(b_value=...)``, which scales the gradient by
      ``sqrt(b / b_design)``.

    The b-value is computed from the actual rasterized trapezoids and the sign
    reversal the 180 imposes, not from an idealized formula, and the request
    is rejected if the system cannot reach it. NOPOS/NOROT keep the module out
    of the imaging FOV transform; NOROT does *not* suppress an explicit block
    rotation extension, which is what makes per-direction rotation work.

    Parameters
    ----------
    b_value : float
        Design (maximum) b-value, s/mm^2.
    gradient_duration : float, optional
        Duration of each diffusion lobe (s).
    gradient_separation : float, optional
        Onset-to-onset lobe separation (s); must exceed
        ``gradient_duration``.
    voxel_size : float
        Reference length (m) the terminating spoiler dephases across.
    spoil_cycles : float, optional
        Dephasing cycles per ``voxel_size``.
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    DiffusionPrepPulse
        Module with ``.gradient``, ``.b_value``, and total ``.duration``.

    Raises
    ------
    ValueError
        If the requested b-value exceeds what the system can reach with the
        given timing (the achievable maximum is reported in the message).

    Examples
    --------
    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> prep = design.make_diffusion_prep(500.0, voxel_size=2e-3)
    >>> prep.b_value
    500.0
    >>> prep.gradient_scale(125.0)
    0.5

    Sweep directions and b-values from the single design::

        from pulserver import ScanLoop

        directions = _lowlevel.make_golden_means_3d_tilt(6).positions
        for rotation in ScanLoop(directions, (np.arange(len(directions)),)).to_rotations():
            prep.set_state(b_value=250.0, rotation=rotation)
            for block in prep:
                seq.add_block(*block)

    The module's blocks, as they are played:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.design as design
       from pulserver.design import _lowlevel
       import pulserver.pypulseq as pp
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       from _figures import preparation_figure
       preparation_figure(design.make_diffusion_prep(1000.0, voxel_size=2e-3, system=system))

    See Also
    --------
    pulserver.ScanLoop.to_rotations : diffusion-direction rotation matrices.
    """
    system = default_system(system)
    if b_value <= 0 or gradient_duration <= 0 or gradient_separation <= gradient_duration:
        raise ValueError("require b_value > 0 and gradient_separation > gradient_duration > 0")
    max_gradient = pp.make_trapezoid(
        channel="z",
        amplitude=system.max_grad,
        duration=gradient_duration,
        system=system,
    )
    rf90_down = make_hard_pulse(np.pi / 2.0, duration=0.5e-3, system=system, use="preparation").rf
    rf180 = make_refocusing_pulse(duration=1e-3, system=system).rf
    rf90_up = make_hard_pulse(
        np.pi / 2.0,
        duration=0.5e-3,
        phase_offset=np.pi,
        system=system,
        use="preparation",
    ).rf
    remaining = gradient_separation - pp.calc_duration(max_gradient) - pp.calc_duration(rf180)
    if remaining < -1e-9:
        raise ValueError("gradient_separation is too short for the diffusion and refocusing pulses")
    delay1 = round_to_raster(max(0.0, remaining / 2.0))
    delay2 = round_to_raster(max(0.0, remaining - delay1))
    middle_duration = delay1 + pp.calc_duration(rf180) + delay2
    maximum_b_value = _diffusion_pair_b_value(max_gradient, middle_duration, system.grad_raster_time)
    if b_value > maximum_b_value * (1.0 + 1e-9):
        raise ValueError(
            f"requested diffusion preparation exceeds system limits (maximum {maximum_b_value:.1f} s/mm^2)"
        )
    gradient = pp.scale_grad(max_gradient, math.sqrt(b_value / maximum_b_value), system=system)
    spoiler = make_spoiler(system, voxel_size, dephasing_cycles=spoil_cycles)
    total = (
        pp.calc_duration(rf90_down)
        + 2.0 * pp.calc_duration(gradient)
        + delay1
        + pp.calc_duration(rf180)
        + delay2
        + pp.calc_duration(rf90_up)
        + pp.calc_duration(*spoiler)
    )
    return DiffusionPrepPulse(
        system,
        rf90_down,
        rf180,
        rf90_up,
        gradient,
        delay1,
        delay2,
        spoiler,
        _labels(1),
        _labels(0),
        b_value,
        total,
    )
