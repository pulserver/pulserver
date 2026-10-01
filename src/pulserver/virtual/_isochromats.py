"""Isochromats whose magnetisation the Bloch equation carries from one block to the next."""

from __future__ import annotations

import functools

import numpy as np
import pypulseqpp as pp

from .._accelerators import require
from . import _finufft
from ._motion import _Stretches, diffusion_phase, motion_phase

__all__ = ["Isochromats", "Repetitions"]

#: Most bytes a run of repetitions holds beside its isochromats: for its maps,
#: and again for the column sums its fixed points are read from.
MEMORY = 4 << 30


@functools.cache
def _kernels():
    kernels = require("bloch")
    _finufft.install(kernels)
    return kernels


def _per_isochromat(value, count: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    try:
        return np.ascontiguousarray(np.broadcast_to(values, (count,)))
    except ValueError:
        raise ValueError(
            f"{name} must be one value or one per isochromat, got shape {values.shape}"
        ) from None


def _sensitivities(value, count: int, name: str):
    if value is None:
        return None
    values = np.asarray(value, dtype=complex)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != count:
        raise ValueError(
            f"{name} must be (isochromats,) or (isochromats, channels), got shape {values.shape}"
        )
    return np.ascontiguousarray(values)


def _axes(gradients) -> list:
    axes = [None, None, None] if gradients is None else list(gradients)
    if len(axes) != 3:
        raise ValueError("gradients must hold the three axes, x, y and z")
    return [
        None
        if axis is None or np.size(axis) == 0
        else np.ascontiguousarray(axis, dtype=float)
        for axis in axes
    ]


def _rotation(rotation):
    if rotation is None:
        return None
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"rotation must be a finite (3, 3) matrix, got shape {matrix.shape}"
        )
    return np.ascontiguousarray(matrix)


def _field(rf, system) -> tuple:
    """Return the field ``rf`` plays as ``(start, step, samples)``, samples ``(channels, steps)``."""
    if rf is None:
        return 0.0, 0.0, None
    if hasattr(rf, "signal"):
        frequency, phase = pp.calc_absolute_offsets(rf, system=system)
        return _kernels().pulse_steps(
            np.ascontiguousarray(rf.t, dtype=float).ravel(),
            np.ascontiguousarray(rf.signal, dtype=complex).ravel(),
            float(rf.delay),
            phase,
            frequency,
            float((pp.Opts.default if system is None else system).rf_raster_time),
        )
    start, step, samples = rf
    samples = np.asarray(samples, dtype=complex)
    if samples.ndim == 1:
        samples = samples[None, :]
    return float(start), float(step), np.ascontiguousarray(samples)


def _window(adc, system) -> tuple:
    """Return the sample times and the phase each is demodulated by, None for plain times."""
    if adc is None:
        return None, None
    if not hasattr(adc, "num_samples"):
        return np.ascontiguousarray(adc, dtype=float), None
    count = int(adc.num_samples)
    dwell = float(adc.dwell)
    if count < 0 or not dwell > 0.0:
        raise ValueError("an ADC needs a count of samples and a positive dwell time")
    modulation = getattr(adc, "phase_modulation", None)
    modulation = (
        np.zeros(0)
        if modulation is None
        else np.asarray(modulation, dtype=float).ravel()
    )
    if modulation.size not in (0, count):
        raise ValueError(
            f"an ADC's phase modulation must hold one value per sample, {count}, got {modulation.size}"
        )
    frequency, phase = pp.calc_absolute_offsets(adc, system=system)
    return _kernels().adc_window(
        count,
        dwell,
        float(adc.delay),
        phase,
        frequency,
        np.ascontiguousarray(modulation),
    )


class Isochromats:
    """Isochromats, and the magnetisation the Bloch equation carries from one block to the next.

    The magnetisation turns about the field ``b = (Re b1, Im b1, bz)``, in Hz,
    in the frame rotating at the reference frequency, as the magnetic moment
    of a nucleus of positive gyromagnetic ratio precesses: ``dM/dt = 2 pi M x
    b``, clockwise seen from the tip of ``b``. A 90 degree pulse along ``+x``
    takes ``+z`` to ``+y``, and an isochromat at a positive off-resonance or
    along a positive gradient accrues ``Mx + i My`` a negative phase.

    Free precession under a piecewise-linear gradient, with relaxation, is
    integrated exactly and applied when the magnetisation is next needed. An
    RF pulse is a sequence of steps, each a rotation about the step's mean
    field between two half steps of relaxation; isochromats that see the same
    field during a pulse share its computation, and a pulse that differs from
    an earlier one by its phase alone, under the same gradient, reuses the
    earlier one's computation turned about z by that phase. A pulse played
    under no gradient or one held throughout it is computed on a grid of the
    field an isochromat sees and interpolated, to within about ``1e-7`` of the
    equilibrium magnetisation, wherever the grid costs fewer maps than the
    isochromats' groups: without transmit sensitivities, or with every channel
    playing one waveform times a weight of its own, when an isochromat's
    transmit field is that waveform times one complex drive, whose magnitude
    the grid spans too and whose phase turns the computation about z. An ADC
    window under a gradient held throughout it is read by a non-uniform FFT
    of the isochromats of each T2, to within about ``1e-13`` of the sum of
    the magnitudes of their transverse magnetisations times their receive
    sensitivities, wherever that costs less than turning every isochromat at
    every sample. A window under any other gradient whose k moves along axes
    on which the isochromats lie on a lattice is read by FINUFFT, wherever
    that costs less: each isochromat's value is summed onto its lattice point
    for each coil and each of a few Chebyshev points across the window,
    between which its decay and precession are interpolated, and the lattice
    is transformed to each sample's k, to within about ``1e-11`` of the sum of
    the magnitudes of the terms each sample sums. :meth:`play` reads either to
    within a tolerance instead where given one.

    Parameters
    ----------
    positions : array_like
        ``(n, 3)`` positions, in m, along the axes the gradients are played on.
    proton_density : float or array_like, default=1.0
        Equilibrium longitudinal magnetisation, per isochromat.
    t1, t2 : float or array_like, default=inf
        Relaxation times, in s; ``inf`` for none.
    off_resonance : float or array_like, default=0.0
        Precession frequency at rest, in Hz from the reference frequency:
        field inhomogeneity and chemical shift together.
    transmit : array_like, default=None
        Complex transmit sensitivities, ``(n,)`` or ``(n, channels)``, scaling
        the field of each RF channel. By default one channel of unit
        sensitivity, onto which the channels of a pTx pulse are summed.
    receive : array_like, default=None
        Complex receive sensitivities, ``(n,)`` or ``(n, coils)``. By default
        one coil of unit sensitivity. A C-contiguous complex128 array, a
        memory-mapped one included, is read in place into the isochromats'
        own layout, so that it is never copied whole into memory.
    diffusion : float or array_like, default=0.0
        Isotropic diffusion coefficient, in m²/s, per isochromat.
    motion : callable, default=None
        Where the isochromats lie as their subject moves: ``motion(t,
        positions)`` returns the ``(n, 3)`` positions, in m, that the
        ``positions`` at rest move to at time ``t``, in s on the isochromats'
        clock (:attr:`elapsed`); :class:`~pulserver.virtual.RigidMotion` for a
        rigid body. Every isochromat keeps its other properties as it moves.
    seed : int or numpy.random.Generator, default=None
        Seed, or generator, of the Brownian walks that ``diffusion`` drives.
    threads : int, default=0
        Worker threads; 0 for every core.
    device : str or torch.device, default=None
        A torch device the windows read on a lattice are summed and
        transformed on: a CUDA device, on which a Triton kernel sums them and
        cuFINUFFT transforms them, or the CPU, under Triton's interpreter
        alone (``TRITON_INTERPRET=1`` before triton is first imported), on
        which FINUFFT transforms them. Needs the ``gpu`` extra. By default
        the engine reads them itself.

    Raises
    ------
    ValueError
        If a property has the wrong shape or is not finite, a relaxation
        time is not positive, or a diffusion coefficient is negative.
    ImportError
        If ``device`` is given without torch and Triton, or is a CUDA device
        without cuFINUFFT.
    RuntimeError
        If ``device`` is the CPU outside Triton's interpreter.

    Notes
    -----
    Every isochromat starts at equilibrium, along ``+z``. The engine computes
    what the sequence does to these isochromats; it does not model the
    scanner's hardware.

    Isochromats that move or diffuse play block by block. Before each block,
    ``motion`` places them where they are at its start, and they stay there
    through the block; after it, each is turned by the phase the block's
    gradient adds along its path within the block, from the end of the
    block's RF pulse on, or from the start of a block without one. A
    diffusing isochromat keeps its position and follows a Brownian walk of
    its own, which turns it by the phase the gradient plays along the walk:
    from the walk so far, and from its increments over the block, drawn with
    the displacement they make, from their joint normal distribution. A
    voxel's diffusion attenuation is the mean of its isochromats' phase
    factors, which holds to about one over the square root of their number.
    Samples within a block are read with every isochromat where it stood at
    the block's start.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.virtual import Isochromats
    >>> spins = Isochromats([[0.0, 0.0, 0.0]], t2=0.1)

    A 25 kHz field along ``+x`` held for 10 us turns ``+z`` by 90 degrees,
    to ``+y``; the transverse magnetisation then decays with ``t2``:

    >>> pulse = (0.0, 10e-6, [25e3])
    >>> signal = spins.play(10.01e-3, rf=pulse, adc=[10e-6, 10.01e-3])
    >>> np.round(signal, 4)
    array([[0.+1.j    , 0.+0.9048j]])
    """

    def __init__(
        self,
        positions,
        *,
        proton_density=1.0,
        t1=np.inf,
        t2=np.inf,
        off_resonance=0.0,
        transmit=None,
        receive=None,
        diffusion=0.0,
        motion=None,
        seed=None,
        threads: int = 0,
        device=None,
    ):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (n, 3), got shape {positions.shape}")
        count = positions.shape[0]
        diffusion = _per_isochromat(diffusion, count, "diffusion")
        if not np.all(np.isfinite(diffusion)) or np.any(diffusion < 0.0):
            raise ValueError("diffusion coefficients must be finite and not negative")
        # Where a motion places the isochromats from, and last placed them.
        self._rest = None if motion is None else np.array(positions)
        self._placed = self._rest
        self._motion = motion
        self._diffusion = diffusion if np.any(diffusion > 0.0) else None
        self._walk = None if self._diffusion is None else np.zeros((count, 3))
        self._rng = np.random.default_rng(seed)
        self._native = _kernels().Isochromats(
            np.ascontiguousarray(positions),
            _per_isochromat(proton_density, count, "proton_density"),
            _per_isochromat(t1, count, "t1"),
            _per_isochromat(t2, count, "t2"),
            _per_isochromat(off_resonance, count, "off_resonance"),
            _sensitivities(transmit, count, "transmit"),
            _sensitivities(receive, count, "receive"),
            int(threads),
        )
        if device is not None:
            from ._device import LatticeDevice

            if not isinstance(device, LatticeDevice):
                device = LatticeDevice(device)
            self._native.use_lattice_device(device)

    def __len__(self) -> int:
        return self._native.size

    @property
    def coils(self) -> int:
        """Number of receive coils; one without receive sensitivities."""
        return self._native.coils

    @property
    def elapsed(self) -> float:
        """Time played since construction or the last :meth:`reset`, in s."""
        return self._native.elapsed

    @property
    def lattice_windows(self) -> int:
        """ADC windows read on a lattice by FINUFFT since construction."""
        return self._native.lattice_windows

    @property
    def device_windows(self) -> int:
        """Of :attr:`lattice_windows`, those read on the ``device``."""
        return self._native.device_windows

    @property
    def moving(self) -> bool:
        """Whether the isochromats move or diffuse, and so play block by block."""
        return self._motion is not None or self._diffusion is not None

    @property
    def positions(self) -> np.ndarray:
        """``(n, 3)`` positions, in m, as the last block played them, a copy.

        Assigning moves the isochromats there, once the free precession
        pending has been applied where they stood; isochromats given a
        ``motion`` or ``diffusion`` are placed by them alone.
        """
        return self._native.positions()

    @positions.setter
    def positions(self, value) -> None:
        if self.moving:
            raise ValueError(
                "isochromats given a motion or diffusion are placed by them"
            )
        values = np.asarray(value, dtype=float)
        if values.shape != (len(self), 3):
            raise ValueError(
                f"positions must be {(len(self), 3)}, got shape {values.shape}"
            )
        self._native.set_positions(np.ascontiguousarray(values))

    @property
    def magnetization(self) -> np.ndarray:
        """``(n, 3)`` magnetisation, a copy; assigning replaces it."""
        return self._native.magnetization()

    @magnetization.setter
    def magnetization(self, value) -> None:
        values = np.broadcast_to(np.asarray(value, dtype=float), (len(self), 3))
        self._native.set_magnetization(np.ascontiguousarray(values))

    def reset(self) -> None:
        """Return every isochromat to equilibrium, along ``+z``, the clock to zero, and a diffusing one to the start of a new walk."""
        self._native.reset()
        if self._walk is not None:
            self._walk[:] = 0.0

    def play(
        self,
        duration: float,
        *,
        gradients=None,
        rotation=None,
        rf=None,
        adc=None,
        system=None,
        tolerance: float = 0.0,
    ) -> np.ndarray:
        """Play one block's events and return what each coil receives at each ADC sample.

        Times are in s from the block's start. An RF or ADC event plays as
        :doc:`/explanations/bloch-simulation` describes; the field and sample
        times the events amount to can be given instead.

        Parameters
        ----------
        duration : float
            Block duration, in s.
        gradients : sequence, default=None
            Three entries, for the x, y and z axes, each ``None`` or a
            ``(2, m)`` array of corner times over gradient amplitude, in Hz/m.
            The gradient is linear between the corners and zero outside them.
        rotation : array_like, default=None
            ``(3, 3)`` matrix from the axes ``gradients`` are given along to
            the axes of the positions, a reflection allowed. Each axis plays
            the sum its row weights, exact on the union of the corners, with
            a gradient's step from or to zero at its first or last corner
            kept as a step.
        rf : RF event or tuple, default=None
            An RF event: ``signal``, in Hz, at the times ``t`` from the pulse's
            start, which is ``delay`` into the block, with its phase and
            frequency offsets; a dynamic pTx pulse holds its channels one
            after another over one time base. Or the transverse field ``b1``,
            as ``(start, step, samples)``: ``(steps,)`` or
            ``(channels, steps)`` complex samples, in Hz, each held for
            ``step`` from ``start`` on. Without transmit sensitivities the
            channels are summed; with them there must be one channel each.
        adc : ADC event or array_like, default=None
            An ADC event, sampled at the middle of each dwell from its delay
            on, whose samples are returned demodulated by its phase offset,
            frequency offset and phase modulation. Or increasing sample
            times, whose samples are returned as received. No sample may fall
            inside the RF pulse.
        system : Opts, default=None
            The RF raster, over whose steps an RF event with a time shape is
            held, and the gamma and B0 the events' ppm offsets are resolved
            at; the default system when None.
        tolerance : float, default=0.0
            Above zero, an ADC window read by a transform is read to within
            it, relative to the sum of the magnitudes of the terms each sample
            sums.

        Returns
        -------
        NDArray[np.complex128]
            ``(coils, samples)``: at each sample, the sum over the isochromats of
            the receive sensitivity times ``Mx + i My``, demodulated for an ADC
            event.

        Raises
        ------
        ValueError
            If an event does not fit in the block, an ADC sample falls inside
            the RF pulse, the RF channels do not match the transmit
            sensitivities, or an event or the rotation is malformed.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> from pulserver.virtual import Isochromats
        >>> spins = Isochromats([[0.0, 0.0, 0.0]])
        >>> rf = pp.make_block_pulse(np.pi / 2, duration=1e-3)
        >>> adc = pp.make_adc(1, dwell=10e-6, delay=1e-3)
        >>> np.round(spins.play(1.01e-3, rf=rf, adc=adc), 6)
        array([[0.+1.j]])
        """
        start, step, samples = _field(rf, system)
        times, receiver = _window(adc, system)
        axes, turn = _axes(gradients), _rotation(rotation)
        began = self.elapsed if self.moving else 0.0
        if self._motion is not None:
            self._place(began)
        signal = self._native.play(
            float(duration),
            axes,
            turn,
            start,
            step,
            samples,
            times,
            float(tolerance),
        )
        if self.moving:
            pulse_end = 0.0 if samples is None else start + step * samples.shape[1]
            self._move_on(began, float(duration), axes, turn, min(pulse_end, duration))
        return signal if receiver is None else signal * np.exp(1j * receiver)

    def _place(self, t: float) -> None:
        """Move the isochromats to where ``motion`` places them at ``t``."""
        placed = np.asarray(self._motion(t, self._rest), dtype=float)
        if placed.shape != self._rest.shape or not np.all(np.isfinite(placed)):
            raise ValueError(
                f"a motion must place every isochromat at finite (n, 3) positions, "
                f"got shape {placed.shape} at {t} s"
            )
        if not np.array_equal(placed, self._placed):
            self._native.set_positions(np.ascontiguousarray(placed))
            self._placed = placed

    def _move_on(self, began, duration, axes, turn, pulse_end) -> None:
        """Turn each isochromat by the phase its motion and walk over the block add, and advance the walk."""
        played = any(axis is not None for axis in axes)
        stretches = (
            _Stretches(axes, turn, pulse_end, duration)
            if played and duration > pulse_end
            else None
        )
        phase = None
        if self._motion is not None and stretches is not None:
            phase = motion_phase(
                stretches, self._motion, self._rest, began, self._placed
            )
        if self._diffusion is not None:
            walked = diffusion_phase(
                stretches,
                pulse_end,
                duration - pulse_end,
                self._diffusion,
                self._walk,
                self._rng,
            )
            phase = walked if phase is None else phase + walked
        if phase is not None and np.any(phase):
            self._native.precess(np.ascontiguousarray(phase))

    def repetitions(
        self,
        blocks,
        phases,
        areas=None,
        *,
        adc_phases=None,
        readouts=None,
        nets=None,
        system=None,
        tolerance: float = 0.0,
    ) -> Repetitions:
        """Repetitions of a sequence of blocks, from the magnetisation where it stands.

        Repetition ``n`` plays ``blocks`` with every RF event's phase offset
        larger by ``phases[n]``, every ADC event's by ``adc_phases[n]``, and
        gradients that differ from the blocks' by a waveform zero during every
        RF pulse and held through every ADC window, at ``readouts[n, w]``
        through the repetition's ``w``-th window, whose area at that window's
        first sample is ``areas[n, w]``, by the start of any RF pulse zero,
        and over the repetition ``nets[n]``: a phase encoding, or a readout
        turned with its prephaser, rewound or not.

        One repetition applies an affine map to each isochromat's
        magnetisation, and another to its transverse magnetisation at each
        window's first sample. Four plays of ``blocks``, from no magnetisation
        and from a unit magnetisation along each axis, give both; every
        repetition then follows from them, a pulse turned by a phase offset
        turning the maps alike. The windows are read as :meth:`play` reads a
        window under a gradient held throughout it.

        Parameters
        ----------
        blocks : sequence of dict
            One repetition, each block as :meth:`play`'s keyword arguments:
            ``duration`` and, where it plays them, ``gradients``,
            ``rotation``, ``rf`` and ``adc``. Every window must be read under
            a gradient held throughout it, its samples on one side of any RF
            pulse in its block.
        phases : array_like
            ``(repetitions,)`` increase of every RF event's phase offset, in
            rad.
        areas : array_like, default=None
            ``(repetitions, windows, 3)`` phase-encoding area at each window's
            first sample, in 1/m, along the axes the positions are given
            along; none by default.
        adc_phases : array_like, default=None
            ``(repetitions,)`` increase of every ADC event's phase offset, in
            rad; ``phases`` by default.
        readouts : array_like, default=None
            ``(repetitions, windows, 3)`` gradient the difference holds
            through each window, in Hz/m, along the axes the positions are
            given along; none by default. A window with any is read along each
            repetition's own gradient, and the magnetisation is not split.
        nets : array_like, default=None
            ``(repetitions, 3)`` area the difference leaves over each
            repetition, in 1/m, along the same axes; none by default. With
            any, each isochromat's transverse magnetisation is turned by its
            own phase at the end of each repetition, and the magnetisation is
            not split.
        system : Opts, default=None
            As :meth:`play` takes it.
        tolerance : float, default=0.0
            Accuracy of the samples, relative to the sum of the magnitudes of
            the terms each sums; exact to rounding at zero. Above zero, the
            windows are read by a kernel just wide enough for it, and from
            1e-4 on the magnetisation is carried in single precision. Where,
            besides, the phases step by one increment, to within 1e-4 rad,
            and the phase encodings run along at most two axes along which the
            isochromats take few coordinates, as on a lattice, each
            isochromat's magnetisation is split into its fixed point under the
            mean increment and a transient about it: the fixed points' samples
            are summed over the columns of isochromats that share their
            coordinates along those axes, and an isochromat whose transient
            falls below ``tolerance`` times its proton density stands at its
            fixed point from then on.

        Returns
        -------
        Repetitions
            The repetitions, which play in turn; the isochromats hold the
            magnetisation at the start of the next to be played.

        Raises
        ------
        ValueError
            If the phases, ADC phases and areas do not describe one set of
            repetitions, a window is not read under a held gradient, a block
            is one :meth:`play` would refuse, or the isochromats move or
            diffuse.

        Examples
        --------
        Balanced steady-state free precession, the RF phase alternating:

        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> from pulserver.virtual import Isochromats
        >>> spins = Isochromats([[0.0, 0.0, 0.0]], t1=1.0, t2=0.1)
        >>> tr = [
        ...     dict(duration=1e-3, rf=pp.make_block_pulse(np.pi / 4, duration=0.5e-3)),
        ...     dict(duration=4e-3, adc=pp.make_adc(1, dwell=10e-6, delay=1.5e-3)),
        ... ]
        >>> scan = spins.repetitions(tr, np.pi * np.arange(4))
        >>> scan.play().shape
        (4, 1, 1)
        """
        if self.moving:
            raise ValueError("isochromats that move or diffuse play block by block")
        conversions = []
        receivers = []
        for block in blocks:
            start, step, samples = _field(block.get("rf"), system)
            times, receiver = _window(block.get("adc"), system)
            if times is not None and times.size:
                receivers.append(np.zeros(times.size) if receiver is None else receiver)
            conversions.append(
                (
                    float(block["duration"]),
                    _axes(block.get("gradients")),
                    _rotation(block.get("rotation")),
                    start,
                    step,
                    samples,
                    times,
                    receiver,
                )
            )
        phases = np.ascontiguousarray(np.asarray(phases, dtype=float).ravel())
        count = phases.size
        adc_phases = (
            phases
            if adc_phases is None
            else np.ascontiguousarray(np.asarray(adc_phases, dtype=float).ravel())
        )
        if adc_phases.size != count:
            raise ValueError(
                f"adc_phases must hold one value per repetition, {count}, got {adc_phases.size}"
            )
        windows = len(receivers)
        encoded = _per_window(areas, "areas", count, windows)
        turned = _per_window(readouts, "readouts", count, windows)
        left = np.zeros((count, 3)) if nets is None else np.asarray(nets, dtype=float)
        if left.shape != (count, 3):
            raise ValueError(
                f"nets must be (repetitions, 3), {(count, 3)}, got {left.shape}"
            )
        native = _kernels().Repetitions(
            self._native,
            conversions,
            phases,
            adc_phases,
            np.ascontiguousarray(encoded).ravel(),
            np.ascontiguousarray(turned).ravel(),
            np.ascontiguousarray(left).ravel(),
            float(tolerance),
        )
        steady = (
            _steady(native, encoded, float(tolerance), [r.size for r in receivers])
            if tolerance
            else None
        )
        return Repetitions(
            self,
            native,
            demodulation=(
                -phases,
                adc_phases,
                np.concatenate(receivers) if receivers else np.zeros(0),
            ),
            steady=steady,
        )


def _per_window(given, name: str, count: int, windows: int) -> np.ndarray:
    """Return ``given`` as ``(repetitions, windows, 3)``, zero where None.

    Raises
    ------
    ValueError
        If it holds another shape.
    """
    values = (
        np.zeros((count, windows, 3))
        if given is None
        else np.asarray(given, dtype=float)
    )
    if values.shape != (count, windows, 3):
        raise ValueError(
            f"{name} must be (repetitions, windows, 3), {(count, windows, 3)}, "
            f"got {values.shape}"
        )
    return values


def _steady(
    native, areas: np.ndarray, tolerance: float, samples: list[int]
) -> list | None:
    """Split the magnetisation; return what reads each window's samples from the fixed points, or None where it cannot.

    A window's fixed points are read by columns (:class:`_Columns`) along the
    axes :func:`_encoded` finds; where the columns of every window, of
    ``samples`` samples each, would take more than :data:`MEMORY` bytes, the
    magnetisation is not split.
    """
    readers = _encoded(native, areas, tolerance)
    if readers is None:
        return None
    sizes = [
        16
        * native.coils
        * count
        * int(np.prod([native.lattice(axis).size for axis in along]))
        for count, along in zip(samples, readers, strict=True)
    ]
    if sum(sizes) > MEMORY or not native.split():
        return None
    kept = (MEMORY - sum(sizes)) // max(len(readers), 1)
    parts = []
    for w, along in enumerate(readers):
        constant = np.where(
            np.isin(np.arange(3), along),
            0.0,
            areas[:, w].mean(axis=0) if len(areas) else 0.0,
        )
        sums = native.column_sums(w, along, np.ascontiguousarray(constant, dtype=float))
        parts.append(
            _Columns(
                sums,
                [native.lattice(axis) for axis in along],
                areas[:, w][:, along],
                kept,
            )
        )
    return parts


def _encoded(native, areas: np.ndarray, tolerance: float) -> list | None:
    """Each window's phase-encoded axes, the outer axis first; None where a window's are more than two or not tabulated.

    An axis along which the areas vary by less than a quarter of the
    tolerance, in phase over the isochromats' reach, counts as encoded by
    their mean. The outer axis is the one of fewer distinct areas.
    """
    reach = native.reach * np.sqrt(3.0)
    readers = []
    for w in range(native.windows):
        spread = np.ptp(areas[:, w], axis=0) if len(areas) else np.zeros(3)
        along = [
            axis
            for axis in range(3)
            if 2.0 * np.pi * spread[axis] * reach > tolerance / 4.0
        ]
        if len(along) > 2 or not all(native.lattice(axis).size for axis in along):
            return None
        distinct = [np.unique(areas[:, w, axis]).size for axis in along]
        readers.append([along[k] for k in np.argsort(distinct, kind="stable")])
    return readers


class _Columns:
    """A window's steady samples from its column sums, per repetition.

    ``sums`` is ``(*values, coils, samples)`` over the tabulated coordinates
    along the encoded axes, and ``encodings`` ``(repetitions, axes)`` each
    repetition's area along them, in 1/m. A repetition's samples are the sums
    times exp(-2 pi i area . coordinate), summed over the columns. With two
    axes, the sum along the first, the outer axis, is taken once per area,
    for the areas repetitions need together, and as many are kept as fit in
    ``kept`` bytes.
    """

    def __init__(
        self, sums: np.ndarray, values: list, encodings: np.ndarray, kept: int
    ) -> None:
        self._values = values
        self._encodings = encodings
        self._shape = sums.shape[len(values) :]
        width = int(np.prod(self._shape))
        if len(values) < 2:
            self._sums = sums.reshape(-1, width)
            return
        self._inner = 1
        self._areas, self._which = np.unique(encodings[:, 0], return_inverse=True)
        self._outer = values[0]
        self._sums = sums.reshape(self._outer.size, -1)
        self._kept = {}
        self._keep = max(1, kept // (self._sums.itemsize * self._sums.shape[1]))

    def _outer_sums(self, indices: np.ndarray) -> None:
        """Keep the sums along the outer axis at the areas of ``indices``, those missing in one product."""
        missing = [int(index) for index in indices if int(index) not in self._kept]
        if not missing:
            return
        turn = np.exp(-2j * np.pi * np.outer(self._areas[missing], self._outer))
        made = (turn @ self._sums).reshape(
            len(missing), self._values[self._inner].size, -1
        )
        needed = {int(index) for index in indices}
        for index in list(self._kept):
            if len(self._kept) + len(missing) <= self._keep:
                break
            if index not in needed:
                del self._kept[index]
        for index, part in zip(missing, made, strict=True):
            self._kept[index] = part

    def samples(self, first: int, count: int) -> np.ndarray:
        chosen = self._encodings[first : first + count]
        if not self._values:
            return np.broadcast_to(
                self._sums.reshape(self._shape), (count, *self._shape)
            ).copy()
        if len(self._values) == 1:
            turn = np.exp(-2j * np.pi * np.outer(chosen[:, 0], self._values[0]))
            return (turn @ self._sums).reshape(count, *self._shape)
        out = np.empty((count, int(np.prod(self._shape))), dtype=complex)
        which = self._which[first : first + count]
        indices = np.unique(which)
        self._outer_sums(indices)
        for index in indices:
            rows = np.flatnonzero(which == index)
            turn = np.exp(
                -2j
                * np.pi
                * np.outer(chosen[rows, self._inner], self._values[self._inner])
            )
            out[rows] = turn @ self._kept[int(index)]
        return out.reshape(count, *self._shape)


class Repetitions:
    """Repetitions of a sequence of blocks, played in turn on isochromats.

    Made by :meth:`Isochromats.repetitions`, which states what each repetition
    plays. The isochromats hold the magnetisation at the start of the next
    repetition to be played; blocks played on them in between break the
    repetitions that follow.
    """

    def __init__(
        self, isochromats: Isochromats, native, *, demodulation, steady=None
    ) -> None:
        self._isochromats = isochromats
        self._native = native
        self._turns, self._adc_phases, self._receiver = demodulation
        self._steady = steady

    def __len__(self) -> int:
        return self._native.count

    @property
    def played(self) -> int:
        """Repetitions played so far."""
        return self._native.played

    @property
    def samples(self) -> int:
        """ADC samples per repetition, per coil, its windows in play order."""
        return self._native.samples

    def play(self, count: int | None = None) -> np.ndarray:
        """Play the next ``count`` repetitions and return what each coil receives.

        Parameters
        ----------
        count : int, default=None
            Repetitions to play; every one left by default.

        Returns
        -------
        NDArray[np.complex128]
            ``(count, coils, samples)``: each repetition's samples, its windows
            in play order, demodulated as :meth:`Isochromats.play` demodulates
            an ADC event.

        Raises
        ------
        ValueError
            If fewer than ``count`` repetitions remain.
        """
        first = self.played
        count = len(self) - first if count is None else int(count)
        signal = self._native.play(count)
        if self._steady and count:
            chosen = slice(first, first + count)
            steady = np.concatenate(
                [part.samples(first, count) for part in self._steady], axis=2
            )
            phase = (
                self._turns[chosen, None]
                + self._adc_phases[chosen, None]
                + self._receiver
            )
            signal += steady * np.exp(1j * phase)[:, None, :]
        return signal

    @property
    def carried(self) -> int:
        """Isochromats carried through the next repetition; the rest stand at their fixed points."""
        return self._native.carried
