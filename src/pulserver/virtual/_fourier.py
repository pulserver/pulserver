"""The Fourier engine: a phantom's tissue simulated as event streams, its images encoded along the trajectory."""

from __future__ import annotations

__all__ = ["FourierPlayer"]

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ._timeline import Timeline
from ._tissue import Tissue

#: Dephasing across a voxel, in cycles, from which the net moment between two
#: events shifts the configuration states by one order.
SHIFT_CYCLES = 0.5

#: Relative residual to which the signals of the tissue's classes are spanned
#: by a temporal basis across the readouts, and their decay and precession by
#: a basis across each readout's samples.
TOLERANCE = 1e-2

#: Most terms of either basis.
MAX_TERMS = 32

#: Phase, in rad, an off-resonance accrues between neighbouring bins of the
#: field over the longest time from an excitation to an echo.
_PHASE_PER_BIN = 0.3

#: Bins of the field over the shortest interval between pulses with no shift
#: in it, across which a balanced sequence's response repeats.
_BINS_PER_PERIOD = 32

#: Narrowest and widest bins of the field, in Hz.
_FINEST_BIN, _COARSEST_BIN = 0.5, 16.0

#: Width of a bin of the transmit field, relative to its nominal amplitude.
_B1_BIN = 0.05

#: The flip angles across a pulse's band, relative to the one on resonance,
#: entries are grouped at: halving below a tenth, as a side lobe turns, and in
#: steps of a twentieth above.
_LEVELS = np.concatenate(
    [[0.0], 2.0 ** -np.arange(7, 3, -1), np.arange(0.1, 2.0001, 0.05)]
)

#: Gradients pulses may play under beyond which they select by frequency
#: alone: between them, pulses whose bands cover the object, as a ZTE scan's
#: are.
_GRADIENTS = 16

#: Widest steps, in Hz, of the frequencies the basis across a readout is
#: tabulated at and fitted to; and the cycles neighbouring frequencies part by
#: over the longest time the basis spans, from which the steps narrow.
_TABLE_STEP, _SVD_STEP = 0.5, 4.0
_TABLED_TURN, _FITTED_TURN = 0.015, 0.25

#: Most terms of the basis across a readout.
_READOUT_TERMS = 512

#: Largest spacing, in m, of the points the receive sensitivities are
#: evaluated at, between which they are interpolated.
_COIL_SPACING = 4e-3

#: How far past the widest k a station's grid reaches, and the fewest points
#: along each of its axes.
_OVERSAMPLING, _FEWEST_POINTS = 1.25, 16

#: Bytes of signals and of image batches held on the device at once.
_BUDGET = 1 << 28

#: Samples a span of the scan holds at least, unless it ends the scan.
_SPAN_SAMPLES = 1 << 18

#: Entries whose weights the terms the images are of are fitted to.
_MIXED_SAMPLE = 1 << 16

#: Groups times atoms whose signals a temporal basis is fitted to, beyond
#: which the rest are projected onto it.
_SKETCH = 4096

#: Deviation from the periodic state, relative to the equilibrium
#: magnetization, below which one period of a stream stands for every later
#: one: relaxation contracts the deviation by at least ``exp(-t / T1)``.
_SETTLED = 1e-3

#: Resolution, in rad, of the phase increments two periods of a stream are
#: compared by: a sequence file keeps a phase to about 1e-5.
_PHASE_STEP = 2.0 * math.pi / 2**16

#: Values a period is tried at, from those the stream's middle TR recurs at.
_PERIODS = 64

# The cache's RF uses.
_EXCITATION = 1
_REFOCUSING = 2


class FourierPlayer:
    """The blocks the cache beside a sequence file plays, acquired of a phantom's tissue by the Fourier engine.

    Each pulse turns an entry ideally, through its flip angle on resonance
    times the magnitude of the transmit field the entry sees and the pulse's
    own profile at the field the entry sees during it, about its axis turned
    by the transmit field's phase. Between pulses and readouts, the net moment
    of the gradients across a voxel shifts the configuration states by one
    order where it dephases by at least :data:`SHIFT_CYCLES` cycles along the
    axes the tissue spans; its part that changes from one repetition to the
    next counts where the intervals between two pulses do not cancel it, and
    anything less is taken as balanced. Entries a pulse turns alike form a
    group, and the groups an excitation turns form a station: its readouts,
    and a stream of the events that act on its groups, a train per group,
    over each class of tissue, bin of the field and bin of the transmit field
    the entries interpolate between, simulated by TorchSim's extended phase
    graphs. At each readout's echo, the sample nearest the centre of k-space,
    those signals are spanned by a temporal basis. Across the readout, each
    entry decays with its T2 from the echo, dephases with its T2' over the
    time the magnetization has gone unrefocused, and precesses at its
    frequency from the echo, spanned by a basis of the samples' times. A
    station's readouts encode one image per pair of terms, sampled on a grid
    along the logical axes the trajectory encodes, at the resolution its
    widest k reaches, over the entries it excites, each entry a cube of
    uniform magnetization. Each image times each coil's sensitivity is
    transformed to the samples by bartorch's NUFFT.

    Parameters
    ----------
    seq_path
        The first file of the sequence's chain, beside its cache.
    tissue
        The phantom's tissue.
    cache_ext
        The extension of the cache beside each file.
    rotation
        ``(3, 3)`` rotation of the prescription from logical to physical axes;
        the identity by default.
    device
        Where the simulation runs; a card where there is one by default.
    tolerance
        The relative residual of the bases.

    Attributes
    ----------
    played
        The playout's blocks, as :func:`pulserver.ir.playout` records them.
    """

    def __init__(
        self,
        seq_path: Path | str,
        tissue: Tissue,
        cache_ext: str = ".pseg",
        *,
        rotation: np.ndarray | None = None,
        device: torch.device | str | None = None,
        tolerance: float = TOLERANCE,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._timeline = timeline = Timeline(
            seq_path, cache_ext, rotation=rotation, device=self.device
        )
        self.played = timeline.played
        self._tolerance = tolerance
        self._coils = tissue.coils
        pulses = timeline.pulses
        events = _events(timeline)
        everything = _Entries.of(tissue, self.device)
        voxel = _voxel(timeline, everything, tissue.spacing)
        spanned = _spanned(everything, tissue.axes, timeline.rotation, voxel)
        shifted = _shifted(timeline, events, voxel, spanned)
        selector_of, selectors = _selectors(pulses)
        profiles = _profiles(everything, selectors, pulses)
        excitations = torch.as_tensor(
            np.unique(selector_of[pulses.use == _EXCITATION]), device=self.device
        )
        excited = (profiles[:, excitations] > 0).any(dim=1)
        self._entries = everything.subset(torch.nonzero(excited).reshape(-1))
        groups, group_of = torch.unique(profiles[excited], dim=0, return_inverse=True)
        del everything, profiles, excited
        self._groups = groups.cpu().numpy()
        self._group_of = group_of
        self._stations, self._station_of = _stations(timeline, selector_of)
        step = _field_step(timeline, events, shifted)
        self._atoms = _Atoms(self._entries, step)
        self._bases = [
            self._temporal(station, events, shifted, selector_of)
            for station in self._stations
        ]
        self._readout = _ReadoutBasis(
            timeline, self._entries, tolerance, from_excitation=step is None
        )
        self._grids = [
            self._image(basis, tissue, timeline.rotation) for basis in self._bases
        ]
        del self._entries, self._atoms, self._group_of
        self._ends = self._span_ends()

    @property
    def blocks(self) -> int:
        """Blocks in the scan."""
        return int(self.played["duration_us"].size)

    def boundary(self, block: int) -> int:
        """Return the first block from ``block`` on at which a span of the scan may end."""
        at = int(np.searchsorted(self._ends, block))
        return int(self._ends[min(at, self._ends.size - 1)])

    def readouts(self, first: int, last: int) -> Iterator[np.ndarray]:
        """Yield each readout of the blocks from ``first`` to before ``last``: ``(coils, samples)`` complex64, demodulated."""
        timeline = self._timeline
        start, stop = np.searchsorted(timeline.readouts.block, [first, last])
        if stop <= start:
            return
        samples = self._acquired(int(start), int(stop))
        firsts = (
            timeline.readouts.first[start : stop + 1] - timeline.readouts.first[start]
        )
        for at in range(stop - start):
            yield samples[:, firsts[at] : firsts[at + 1]]

    def _temporal(self, station: int, events, shifted, selector_of) -> _Basis:
        """Return the basis spanning the signals of a station's groups and atoms at its readouts' echoes."""
        from torchsim.sequence import EpgEngine, TissueProperties
        from torchsim.sequence._accelerators import pack_description

        readouts = np.flatnonzero(self._station_of == station)
        members = np.flatnonzero(self._groups[:, station] > 0)
        member_t = torch.as_tensor(members, device=self.device)
        local = torch.nonzero(torch.isin(self._group_of, member_t)).reshape(-1)
        atoms = torch.unique(self._atoms.index[local])
        held = self._atoms
        stream = _stream(
            self._timeline,
            events,
            shifted,
            selector_of,
            (self._groups[members] > 0).any(axis=0),
            readouts,
        )
        settle_us = 1e6 * float(held.t1[atoms].max()) * math.log(1.0 / _SETTLED)
        stream, column = _periodic(stream, settle_us)
        played = int(np.count_nonzero(stream.kind == 2))
        description = _description(stream, self._groups[members], self.device)
        packed = pack_description(
            description, repetitions=1, record="all", device=self.device
        )
        count = members.size
        step = max(1, _BUDGET // max(8 * count * played, 1))

        def simulated(chosen: torch.Tensor) -> torch.Tensor:
            """Return the signals of the chosen atoms, ``(groups, atoms, readouts)``."""
            parts = []
            for first in range(0, chosen.numel(), step):
                some = chosen[first : first + step]
                result = EpgEngine().simulate(
                    description,
                    TissueProperties(
                        t1_ms=1e3 * held.t1[some],
                        t2_ms=1e3 * held.t2[some],
                        b0_hz=-held.frequency[some],
                        b1=held.b1[some],
                    ),
                    record="all",
                    device=self.device,
                    events=packed,
                )
                parts.append(result.signal.reshape(count, -1, played).conj())
            return torch.cat(parts, dim=1)

        sketched = max(1, _SKETCH // count)
        if atoms.numel() <= sketched:
            left, right = _leading(
                simulated(atoms).reshape(-1, played), self._tolerance
            )
            coefficients = left.reshape(count, atoms.numel(), -1)
        else:
            generator = torch.Generator(device=self.device).manual_seed(0)
            order = torch.randperm(
                atoms.numel(), device=self.device, generator=generator
            )
            sample = atoms[order[:sketched]]
            _, right = _leading(simulated(sample).reshape(-1, played), self._tolerance)
            coefficients = torch.empty(
                (count, atoms.numel(), right.shape[0]),
                dtype=torch.complex64,
                device=self.device,
            )
            for first in range(0, atoms.numel(), step):
                chosen = atoms[first : first + step]
                coefficients[:, first : first + step] = simulated(chosen) @ right.mH
        return _Basis(
            readouts=readouts,
            members=member_t,
            local=local,
            atoms=atoms,
            coefficients=coefficients,
            temporal=right,
            column=torch.as_tensor(column, device=self.device),
        )

    def _image(self, basis: _Basis, tissue: Tissue, rotation: np.ndarray) -> _Grid:
        """Return the images a station's readouts encode.

        Each entry weighs every pair of a temporal and a readout term; the
        fewest combinations of those pairs spanning the weights of a sample of
        the entries, to :data:`TOLERANCE`, are the terms the images are of.
        Each entry is a cube of uniform magnetization: an image is the inverse
        transform of the entries' spectrum times the cube's, at the grid's
        frequencies. Entries sharing a position share one point of the
        transform.
        """
        from bartorch import linop

        timeline = self._timeline
        turn = torch.as_tensor(rotation, dtype=torch.float32, device=self.device)
        logical = self._entries.positions[basis.local] @ turn
        reach = timeline.readouts.reach[basis.readouts].max(axis=0)
        grid = _Grid.around(logical, tissue.spacing, reach)
        cells, shared = torch.unique(grid.cells(logical), dim=0, return_inverse=True)
        del logical
        count = basis.local.numel()
        generator = torch.Generator(device=self.device).manual_seed(0)
        sample = torch.randperm(count, device=self.device, generator=generator)
        _, mix = _leading(
            self._coefficients(basis, sample[:_MIXED_SAMPLE]), self._tolerance
        )
        grid.mix = mix
        terms = mix.shape[0]
        weights = torch.zeros(
            (cells.shape[0], terms), dtype=torch.complex64, device=self.device
        )
        chunk = max(1, _BUDGET // (8 * 8 * mix.shape[1]))
        for first in range(0, count, chunk):
            part = slice(first, first + chunk)
            weights.index_add_(
                0, shared[part], self._coefficients(basis, part) @ mix.mH
            )
        del shared
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        cube = grid.cube(tissue.axes @ rotation, tissue.spacing, self.device)
        images = torch.empty(
            (terms, *grid.shape), dtype=torch.complex64, device=self.device
        )
        batch = max(1, min(terms, _BUDGET // (8 * grid.points)))
        spread = linop.NUFFT(-cells[None], (batch, *grid.shape), toeplitz=False)
        dims = tuple(range(1, 1 + len(grid.shape)))
        for first in range(0, terms, batch):
            last = min(first + batch, terms)
            held = torch.zeros(
                (batch, 1, cells.shape[0]), dtype=torch.complex64, device=self.device
            )
            held[: last - first, 0] = weights[:, first:last].T
            spectrum = spread.H(held)[: last - first]
            spectrum = spectrum * (math.sqrt(grid.points) * cube)
            images[first:last] = torch.fft.fftshift(
                torch.fft.ifftn(torch.fft.ifftshift(spectrum, dim=dims), dim=dims),
                dim=dims,
            )
        grid.images = images
        grid.sensitivities = grid.coils(tissue, rotation, self.device)
        return grid

    def _coefficients(self, basis: _Basis, part: slice | torch.Tensor) -> torch.Tensor:
        """Return the weight in each pair of terms of the entries of a station at ``part`` of them, ``(entries, rank * terms)``."""
        local = basis.local[part]
        rows = torch.searchsorted(basis.members, self._group_of[local])
        columns = torch.searchsorted(basis.atoms, self._atoms.index[local])
        gathered = basis.coefficients[rows[:, None], columns]
        temporal = (gathered * self._atoms.weight[local][..., None]).sum(1)
        across = self._readout.coefficients(local)
        entries = self._entries
        scale = entries.density[local].to(torch.complex64)
        if entries.transmit is not None:
            scale = scale * torch.exp(-1j * torch.angle(entries.transmit[local]))
        mixed = temporal[:, :, None] * across[:, None, :]
        return mixed.reshape(local.numel(), -1) * scale[:, None]

    def _acquired(self, first: int, last: int) -> np.ndarray:
        """Return the samples of the readouts from ``first`` to before ``last``, ``(coils, samples)``."""
        from bartorch import linop

        timeline = self._timeline
        times, k, receiver = timeline.kspace(first, last)
        firsts = timeline.readouts.first
        readout = np.repeat(np.arange(first, last), np.diff(firsts[first : last + 1]))
        echo_phase = timeline.readouts.receiver[readout]
        out = torch.zeros(
            (self._coils, times.size), dtype=torch.complex64, device=self.device
        )
        stations = self._station_of[readout]
        for index, station in enumerate(self._stations):
            chosen = np.flatnonzero(stations == station)
            if not chosen.size:
                continue
            grid, basis = self._grids[index], self._bases[index]
            position = np.searchsorted(basis.readouts, readout[chosen])
            temporal = basis.temporal[
                :, basis.column[torch.as_tensor(position, device=self.device)]
            ]
            within = firsts[first] + chosen - firsts[readout[chosen]]
            across = self._readout.basis(readout[chosen], within)
            weights = grid.mix @ (temporal[:, None, :] * across[None, :, :]).reshape(
                -1, chosen.size
            )
            phase = -2.0 * math.pi * (k[chosen] @ grid.centre) + (
                receiver[chosen] - echo_phase[chosen]
            )
            factor = torch.as_tensor(
                math.sqrt(grid.points) * np.exp(1j * phase),
                dtype=torch.complex64,
                device=self.device,
            )
            traj = torch.as_tensor(
                grid.trajectory(k[chosen]), dtype=torch.float32, device=self.device
            )[None]
            terms = grid.images.shape[0]
            coils = grid.sensitivities.shape[0]
            per_coil = max(1, min(coils, _BUDGET // (8 * grid.points * terms)))
            nufft = linop.NUFFT(traj, (per_coil * terms, *grid.shape), toeplitz=False)
            where = torch.as_tensor(chosen, device=self.device)
            for c0 in range(0, coils, per_coil):
                c1 = min(c0 + per_coil, coils)
                held = torch.zeros(
                    (per_coil, terms, *grid.shape),
                    dtype=torch.complex64,
                    device=self.device,
                )
                held[: c1 - c0] = grid.sensitivity(c0, c1)[:, None] * grid.images[None]
                transformed = nufft(held.reshape(-1, *grid.shape)).reshape(
                    per_coil, terms, -1
                )[: c1 - c0]
                out[c0:c1, where] = (
                    torch.einsum("ctn,tn->cn", transformed, weights) * factor
                )
        return out.cpu().numpy()

    def _span_ends(self) -> np.ndarray:
        """Return the blocks at which spans may end: after at least :data:`_SPAN_SAMPLES` samples, and the scan's end."""
        played = self.played
        acquired = np.where(played["adc"] != 0, played["adc_samples"], 0)
        total = np.cumsum(acquired.astype(np.int64))
        marks = np.arange(_SPAN_SAMPLES, int(total[-1]) + 1, _SPAN_SAMPLES)
        ends = np.searchsorted(total, marks) + 1
        return np.unique(np.concatenate([ends[ends < self.blocks], [self.blocks]]))


@dataclass(frozen=True, eq=False)
class _Entries:
    """A phantom's entries on the device: positions in m, frequencies in Hz, densities, relaxation times in s and transmit fields."""

    positions: torch.Tensor
    frequency: torch.Tensor
    density: torch.Tensor
    t1: torch.Tensor
    t2: torch.Tensor
    t2_prime: torch.Tensor
    transmit: torch.Tensor | None

    @classmethod
    def of(cls, tissue: Tissue, device: torch.device) -> _Entries:
        def put(values, dtype=torch.float32):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)

        return cls(
            positions=put(tissue.positions),
            frequency=put(tissue.frequency),
            density=put(tissue.density),
            t1=put(tissue.t1),
            t2=put(tissue.t2),
            t2_prime=put(tissue.t2_prime),
            transmit=None
            if tissue.transmit is None
            else put(tissue.transmit, torch.complex64),
        )

    def subset(self, index: torch.Tensor) -> _Entries:
        return _Entries(
            positions=self.positions[index],
            frequency=self.frequency[index],
            density=self.density[index],
            t1=self.t1[index],
            t2=self.t2[index],
            t2_prime=self.t2_prime[index],
            transmit=None if self.transmit is None else self.transmit[index],
        )


@dataclass(frozen=True, eq=False)
class _Basis:
    """A station's readouts, groups, entries and atoms, and the temporal basis spanning their signals.

    Attributes
    ----------
    readouts
        The readouts it acquires.
    members
        Its groups, among all of them.
    local
        Its entries, among the excited ones.
    atoms
        Its atoms, among all of them.
    coefficients
        ``(groups, atoms, rank)`` each group's atoms' coefficients.
    temporal
        ``(rank, played)`` the basis, at the readouts the station's stream
        plays.
    column
        ``(readouts,)`` per readout, the played one whose signal it records.
    """

    readouts: np.ndarray
    members: torch.Tensor
    local: torch.Tensor
    atoms: torch.Tensor
    coefficients: torch.Tensor
    temporal: torch.Tensor
    column: torch.Tensor


@dataclass(frozen=True)
class _Events:
    """Pulses and readouts merged in play order: what each is, its index among its kind, its block and time."""

    kind: np.ndarray
    index: np.ndarray
    block: np.ndarray
    time_us: np.ndarray


def _events(timeline: Timeline) -> _Events:
    pulses, readouts = timeline.pulses, timeline.readouts
    kind = np.concatenate(
        [np.ones(pulses.block.size, np.int64), np.full(len(readouts), 2, np.int64)]
    )
    index = np.concatenate([np.arange(pulses.block.size), np.arange(len(readouts))])
    block = np.concatenate([pulses.block, readouts.block])
    time_us = np.concatenate([pulses.time_us, readouts.echo_us])
    order = np.lexsort((kind, time_us))
    return _Events(kind[order], index[order], block[order], time_us[order])


def _voxel(timeline: Timeline, entries: _Entries, spacing: float) -> np.ndarray:
    """Return the voxel along each logical axis, in m: the resolution the widest k reaches, the excited slab where k does not move, and the tissue's extent where neither bounds it."""
    reach = timeline.readouts.reach.max(axis=0, initial=0.0)
    if entries.positions.shape[0]:
        logical = entries.positions @ torch.as_tensor(
            timeline.rotation, dtype=torch.float32, device=entries.positions.device
        )
        voxel = (logical.amax(0) - logical.amin(0)).double().cpu().numpy() + spacing
    else:
        voxel = np.full(3, spacing)
    with np.errstate(divide="ignore"):
        voxel = np.minimum(voxel, np.where(reach > 0.0, 0.5 / reach, np.inf))
    pulses = timeline.pulses
    for at in np.flatnonzero(pulses.use == _EXCITATION):
        gradient = pulses.gradient[at] @ timeline.rotation
        detuning, ratio = pulses.profiles[pulses.profile[at]]
        half = detuning[ratio >= 0.5]
        width = float(half[-1] - half[0]) if half.size else 0.0
        with np.errstate(divide="ignore"):
            voxel = np.minimum(
                voxel, np.where(gradient != 0.0, width / np.abs(gradient), np.inf)
            )
    return voxel


def _spanned(
    entries: _Entries, axes: np.ndarray, rotation: np.ndarray, voxel: np.ndarray
) -> np.ndarray:
    """Return the projection onto the logical axes the tissue spans within a voxel, ``(3, 3)``.

    An axis is spanned where an entry's cube extends along it, or where the
    entries' positions spread across more than half a voxel along it.
    """
    cube = (np.abs(axes @ rotation) > 1e-6).any(axis=0)
    if entries.positions.shape[0]:
        logical = entries.positions @ torch.as_tensor(
            rotation, dtype=torch.float32, device=entries.positions.device
        )
        spread = (logical.amax(0) - logical.amin(0)).cpu().numpy()
    else:
        spread = np.zeros(3)
    return np.diag((cube | (spread > 0.5 * voxel)).astype(float))


def _shifted(
    timeline: Timeline, events: _Events, voxel: np.ndarray, spanned: np.ndarray
) -> np.ndarray:
    """Return, for each interval between consecutive events, by how many orders its moment shifts the configuration states.

    The moment dephases a voxel along the logical axes it spans, ``spanned``
    projecting onto them, and is read in cycles across the voxel. While the
    magnetization precesses freely from an excitation, the states shift by one
    order over the interval to the next pulse where k winds by
    :data:`SHIFT_CYCLES` or more by then, and each readout reads the pathway
    the timeline found it reads: the shifts fall before the readouts of the
    echoes of earlier intervals, as an SSFP-echo readout reads one, and after
    those of the free induction. Around a refocusing
    pulse, and between pulses that excite nothing, an interval's own moment
    counts: split, over the intervals at the same place in the repetitions
    the cache plays (the same pair of block positions of the same segments),
    into the mean and what changes about it, what changes being phase
    encoding where the intervals between consecutive pulses sum it to
    nothing, and counted where they do not, as the turned readout of a
    radial scan is. An interval shifts the states where the median over its
    place dephases by :data:`SHIFT_CYCLES`.
    """
    readouts = timeline.readouts
    played = timeline.played
    if events.kind.size < 2:
        return np.zeros(0, dtype=bool)
    since = events.time_us - timeline.starts_us[events.block]
    moment = timeline.moment(events.block, since)
    nets = (np.diff(moment, axis=0) @ spanned) * voxel
    place = np.column_stack(
        [played["segment"][events.block], played["position"][events.block], events.kind]
    )
    _, at = _rows(place)
    _, pair = _rows(np.column_stack([at[:-1], at[1:]]))
    pairs = int(pair.max()) + 1
    mean = np.zeros((pairs, 3))
    np.add.at(mean, pair, nets)
    mean /= np.bincount(pair, minlength=pairs)[:, None]
    changing = nets - mean[pair]
    interval = np.cumsum(events.kind == 1)[:-1]
    summed = np.zeros((int(interval.max()) + 1, 3))
    np.add.at(summed, interval, changing)
    cancelled = np.linalg.norm(summed, axis=1) < SHIFT_CYCLES
    dephasing = mean[pair] + np.where(cancelled[interval][:, None], 0.0, changing)
    use = np.where(
        events.kind == 1,
        timeline.pulses.use[np.where(events.kind == 1, events.index, 0)],
        0,
    )
    excitation = use == _EXCITATION
    refocusing = use == _REFOCUSING
    every = np.arange(events.kind.size)
    last = np.maximum.accumulate(np.where(excitation, every, -1))
    turned = np.cumsum(refocusing)
    free = (last >= 0) & (turned == turned[np.clip(last, 0, None)])
    free_gap = free[:-1] & ~refocusing[1:]
    following = events.kind[1:]
    reading = free_gap & (following == 2)
    winding = free_gap & (following == 1) & ((events.kind[:-1] == 2) | excitation[:-1])
    wound = ((moment[1:] - moment[np.clip(last[:-1], 0, None)]) @ spanned) * voxel
    dephasing = np.where(winding[:, None], wound, dephasing)
    size = np.linalg.norm(dephasing, axis=1)
    order = np.lexsort((size, pair))
    counts = np.bincount(pair, minlength=pairs)
    firsts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranked = size[order]
    typical = 0.5 * (ranked[firsts + (counts - 1) // 2] + ranked[firsts + counts // 2])
    shifts = (typical[pair] >= SHIFT_CYCLES).astype(np.int64)
    pathway = np.where(
        events.kind == 2,
        readouts.pathway[np.where(events.kind == 2, events.index, 0)],
        0,
    )
    before = np.where(events.kind[:-1] == 2, pathway[:-1], 0)
    shifts = np.where(reading, np.clip(pathway[1:] - before, 0, None), shifts)
    return np.where(winding, np.clip(shifts - before, 0, None), shifts)


def _leading(
    values: torch.Tensor, tolerance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fewest leading singular terms of a matrix that span it to a relative residual, as ``(left * sigma, right)``.

    At most :data:`MAX_TERMS`; from a randomized decomposition where the
    matrix is much larger than that.
    """
    if min(values.shape) <= 4 * MAX_TERMS:
        u, s, vh = torch.linalg.svd(values, full_matrices=False)
    else:
        q = min(MAX_TERMS + 8, min(values.shape))
        u, s, v = torch.svd_lowrank(values, q=q, niter=2)
        vh = v.mH
    total = torch.linalg.vector_norm(values).double() ** 2
    residual = 1.0 - torch.cumsum(s.double() ** 2, 0) / total.clamp_min(1e-300)
    rank = int(torch.count_nonzero(residual > tolerance**2)) + 1
    rank = max(1, min(rank, MAX_TERMS, s.numel()))
    return u[:, :rank] * s[:rank], vh[:rank]


def _span(
    blocks: Callable[[], Iterator[torch.Tensor]],
    rows: int,
    tolerance: float,
    device: torch.device,
) -> torch.Tensor:
    """Return an orthonormal basis of the fewest leading directions spanning the columns ``blocks`` yields, to a relative residual, ``(rows, terms)``.

    A randomized range finder with one power iteration, over column blocks
    generated as it goes, its sketch doubled until the residual is met or it
    reaches :data:`_READOUT_TERMS`.
    """
    sketch = min(64, rows, _READOUT_TERMS)
    generator = torch.Generator(device=device).manual_seed(0)
    while True:
        guess = torch.zeros((rows, sketch), dtype=torch.complex64, device=device)
        total = 0.0
        for block in blocks():
            probe = torch.randn(
                (block.shape[1], sketch),
                dtype=torch.complex64,
                device=device,
                generator=generator,
            )
            guess += block @ probe
            total += float(torch.linalg.vector_norm(block)) ** 2
        found = torch.linalg.qr(guess).Q
        powered = torch.zeros_like(guess)
        for block in blocks():
            powered += block @ (block.mH @ found)
        found = torch.linalg.qr(powered).Q
        gram = torch.zeros((sketch, sketch), dtype=torch.complex64, device=device)
        for block in blocks():
            projected = found.mH @ block
            gram += projected @ projected.mH
        energy, directions = torch.linalg.eigh(gram)
        energy, directions = energy.flip(0).double(), directions.flip(1)
        residual = 1.0 - torch.cumsum(energy, 0) / max(total, 1e-300)
        rank = int(torch.count_nonzero(residual > tolerance**2)) + 1
        if rank < sketch or sketch >= min(rows, _READOUT_TERMS):
            rank = max(1, min(rank, sketch, rows))
            return found @ directions[:, :rank]
        sketch = min(2 * sketch, rows, _READOUT_TERMS)


def _rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the distinct rows of a 2D array, in sorted order of their bytes, and which each row is."""
    values = np.ascontiguousarray(values)
    keys = values.view(np.dtype((np.void, values.dtype.itemsize * values.shape[1])))
    _, first, inverse = np.unique(keys.ravel(), return_index=True, return_inverse=True)
    return values[first], inverse.reshape(-1)


def _selecting(pulses) -> np.ndarray:
    """Return the gradient each pulse selects under, ``(n, 3)``: zero for all where they play under more than :data:`_GRADIENTS`."""
    gradient = np.round(pulses.gradient, 3)
    if _rows(gradient)[0].shape[0] > _GRADIENTS:
        return np.zeros_like(gradient)
    return gradient


def _selectors(pulses) -> tuple[np.ndarray, np.ndarray]:
    """Return which selector each pulse is, and each selector's first pulse.

    A selector is a gradient, a frequency and a profile.
    """
    offset = (
        pulses.band[:, 0]
        - np.array([profile[0][0] for profile in pulses.profiles])[pulses.profile]
    )
    keys = np.column_stack([_selecting(pulses), np.round(offset, 3), pulses.profile])
    _, selector_of = _rows(keys)
    first = np.full(int(selector_of.max(initial=-1)) + 1, selector_of.size)
    np.minimum.at(first, selector_of, np.arange(selector_of.size))
    return selector_of, first


def _profiles(entries: _Entries, selectors: np.ndarray, pulses) -> torch.Tensor:
    """Return the level of :data:`_LEVELS` nearest each entry's flip angle under each selector, over the one on resonance; zero outside its band, ``(entries, selectors)``."""
    device = entries.positions.device
    held = torch.zeros(
        (entries.positions.shape[0], selectors.size), dtype=torch.int8, device=device
    )
    gradients = _selecting(pulses)
    for at, pulse in enumerate(selectors):
        detuning, ratio = pulses.profiles[pulses.profile[pulse]]
        offset = pulses.band[pulse, 0] - detuning[0]
        field = entries.frequency - float(offset)
        if np.any(gradients[pulse]):
            gradient = torch.as_tensor(
                pulses.gradient[pulse], dtype=torch.float32, device=device
            )
            field = field + entries.positions @ gradient
        step = float(detuning[1] - detuning[0]) if detuning.size > 1 else 1.0
        place = (field - float(detuning[0])) / step
        below = torch.floor(place)
        inside = (below >= 0) & (below < detuning.size - 1)
        index = below.clamp(0, max(detuning.size - 2, 0)).long()
        table = torch.as_tensor(ratio, dtype=torch.float32, device=device)
        upper = table[(index + 1).clamp(max=detuning.size - 1)]
        flip = table[index] + (place - below) * (upper - table[index])
        flip = torch.where(inside, flip, 0.0)
        levels = torch.as_tensor(_LEVELS, dtype=torch.float32, device=device)
        nearest = torch.searchsorted(levels[1:] + levels[:-1], 2.0 * flip.contiguous())
        held[:, at] = nearest.to(torch.int8)
    return held


def _stations(
    timeline: Timeline, selector_of: np.ndarray
) -> tuple[list[int], np.ndarray]:
    """Return the selectors whose excitations readouts follow, and each readout's; -1 before the first excitation."""
    pulses = timeline.pulses
    excitations = np.flatnonzero(pulses.use == _EXCITATION)
    last = (
        np.searchsorted(pulses.time_us[excitations], timeline.readouts.echo_us, "right")
        - 1
    )
    station_of = np.where(
        last >= 0, selector_of[excitations][np.clip(last, 0, None)], -1
    )
    return [int(s) for s in np.unique(station_of) if s >= 0], station_of


def _field_step(
    timeline: Timeline, events: _Events, shifted: np.ndarray
) -> float | None:
    """Return the width of a bin of the field the states are simulated in, in Hz; None where every interval between two pulses shifts them.

    Where every such interval shifts the states, the pathways a readout
    reads have all precessed for the time the magnetization has gone
    unrefocused, and the field turns the samples rather than the states.
    """
    pulses, readouts = timeline.pulses, timeline.readouts
    rf = np.flatnonzero(events.kind == 1)
    passed = np.concatenate([[0], np.cumsum(shifted)])
    balanced = passed[rf[1:]] == passed[rf[:-1]]
    if not balanced.any():
        return None
    excitations = pulses.time_us[pulses.use == _EXCITATION]
    step = _COARSEST_BIN
    if excitations.size and len(readouts):
        last = np.searchsorted(excitations, readouts.echo_us, side="right") - 1
        since = 1e-6 * (readouts.echo_us[last >= 0] - excitations[last[last >= 0]])
        if since.size and since.max() > 0.0:
            step = min(step, _PHASE_PER_BIN / (2.0 * math.pi * since.max()))
    periods = 1e-6 * (events.time_us[rf[1:]] - events.time_us[rf[:-1]])[balanced]
    periods = periods[periods > 0.0]
    if periods.size:
        step = min(step, 1.0 / (_BINS_PER_PERIOD * float(periods.min())))
    return float(np.clip(step, _FINEST_BIN, _COARSEST_BIN))


class _Atoms:
    """The classes of tissue, bins of the field and bins of the transmit field the entries interpolate between.

    Attributes
    ----------
    index, weight
        ``(entries, 4)`` the atoms each entry interpolates between, and its
        weights; ``(entries, 2)`` without bins of the field.
    t1, t2, frequency, b1
        Each atom's relaxation times in s, field in Hz and transmit field.
    """

    def __init__(self, entries: _Entries, step: float | None) -> None:
        relaxation = torch.stack([entries.t1, entries.t2, entries.t2_prime], dim=1)
        classes, class_of = torch.unique(relaxation, dim=0, return_inverse=True)
        if step is None:
            lowest, step = 0.0, 1.0
            field_bin = torch.zeros_like(entries.frequency)
            field_weight = torch.zeros_like(entries.frequency)
            ups = (0,)
        else:
            lowest = (
                float(entries.frequency.min()) if entries.frequency.numel() else 0.0
            )
            field = (entries.frequency - lowest) / step
            field_bin = torch.floor(field)
            field_weight = field - field_bin
            ups = (0, 1)
        if entries.transmit is None:
            magnitude = torch.ones_like(entries.frequency)
        else:
            magnitude = entries.transmit.abs()
        weakest = float(magnitude.min()) if magnitude.numel() else 1.0
        strength = (magnitude - weakest) / _B1_BIN
        strength_bin = torch.floor(strength)
        strength_weight = strength - strength_bin
        field_bin, strength_bin = field_bin.long(), strength_bin.long()
        fields = int(field_bin.max()) + 2 if field_bin.numel() else 2
        strengths = int(strength_bin.max()) + 2 if strength_bin.numel() else 2
        keys, weights = [], []
        for up_field in ups:
            for up_strength in (0, 1):
                keys.append(
                    (class_of * fields + field_bin + up_field) * strengths
                    + strength_bin
                    + up_strength
                )
                weights.append(
                    (field_weight if up_field else 1.0 - field_weight)
                    * (strength_weight if up_strength else 1.0 - strength_weight)
                )
        unique, index = torch.unique(torch.stack(keys, dim=1), return_inverse=True)
        self.index = index
        self.weight = torch.stack(weights, dim=1)
        held_class, rest = unique // (fields * strengths), unique % (fields * strengths)
        self.t1 = classes[held_class, 0]
        self.t2 = classes[held_class, 1]
        self.frequency = lowest + step * (rest // strengths).to(torch.float32)
        self.b1 = weakest + _B1_BIN * (rest % strengths).to(torch.float32)


@dataclass(frozen=True)
class _Stream:
    """A station's events in play order: its pulses and its readouts, and the shifts after each.

    ``selector`` is a pulse's selector and ``flip`` its nominal flip angle in
    rad, -1 and 0 for a readout; ``phase`` is a pulse's phase or a readout's
    receiver phase, in rad; ``orders`` shift the states at ``shift_us``;
    ``readout`` is a readout's index among the timeline's, -1 for a pulse.
    """

    kind: np.ndarray
    time_us: np.ndarray
    selector: np.ndarray
    flip: np.ndarray
    use: np.ndarray
    phase: np.ndarray
    orders: np.ndarray
    shift_us: np.ndarray
    readout: np.ndarray

    def subset(self, chosen: np.ndarray) -> _Stream:
        return _Stream(
            *(getattr(self, name)[chosen] for name in self.__dataclass_fields__)
        )


def _stream(timeline, events, shifted, selector_of, acting, readouts) -> _Stream:
    """Return the pulses that act on a station's groups, under the selectors ``acting`` marks, and its ``readouts``.

    Between two of them, the states shift by as many orders as the interval
    between them does where they are consecutive in the scan, and by one
    where any interval of the scan between them shifts them.
    """
    pulses = timeline.pulses
    own = np.zeros(len(timeline.readouts), dtype=bool)
    own[readouts] = True
    pulse = events.kind == 1
    index = events.index
    kept = np.flatnonzero(
        np.where(
            pulse,
            acting[selector_of[np.where(pulse, index, 0)]],
            own[np.where(pulse, 0, index)],
        )
    )
    passed = np.concatenate([[0], np.cumsum(shifted)])
    following = np.append(kept[1:], events.kind.size - 1)
    orders = passed[following] - passed[kept]
    orders = np.where(following - kept > 1, np.minimum(orders, 1), orders)
    every = np.arange(shifted.size)
    first_shift = np.minimum.accumulate(
        np.where(shifted > 0, every, shifted.size)[::-1]
    )[::-1]
    gap = (
        first_shift[np.minimum(kept, max(shifted.size - 1, 0))]
        if shifted.size
        else kept
    )
    gap = np.minimum(gap, max(events.kind.size - 2, 0))
    shift_us = np.where(
        orders > 0,
        0.5
        * (
            events.time_us[gap]
            + events.time_us[np.minimum(gap + 1, events.kind.size - 1)]
        ),
        events.time_us[kept],
    )
    is_pulse = pulse[kept]
    at = index[kept]
    rf = np.where(is_pulse, at, 0)
    readout = np.where(is_pulse, -1, at)
    return _Stream(
        kind=events.kind[kept],
        time_us=events.time_us[kept],
        selector=np.where(is_pulse, selector_of[rf], -1),
        flip=np.where(is_pulse, pulses.flip[rf], 0.0),
        use=np.where(is_pulse, pulses.use[rf], 0),
        phase=np.where(
            is_pulse,
            pulses.phase[rf],
            timeline.readouts.receiver[np.maximum(readout, 0)],
        ),
        orders=orders,
        shift_us=shift_us,
        readout=readout,
    )


def _signatures(stream: _Stream) -> np.ndarray:
    """Return what makes each event of a stream act as another does after the same history, ``(events, 8)``.

    A pulse's phase counts by its increase over the pulse before it, and a
    readout's by its increase over the last pulse's: two periods whose pulses
    and receivers differ by one phase throughout record the same signal.
    """
    count = stream.kind.size
    every = np.arange(count)
    pulse = stream.kind == 1
    last = np.maximum.accumulate(np.where(pulse, every, -1))
    before = np.where(pulse, np.concatenate([[-1], last[:-1]]), last)
    reference = np.where(before >= 0, stream.phase[np.maximum(before, 0)], 0.0)
    turns = 2**16
    increase = (
        np.round(np.mod(stream.phase - reference, 2.0 * math.pi) / _PHASE_STEP).astype(
            np.int64
        )
        % turns
    )
    step_ns = np.round(
        1e3 * np.diff(stream.time_us, append=stream.time_us[-1:])
    ).astype(np.int64)
    shift_ns = np.round(1e3 * (stream.shift_us - stream.time_us)).astype(np.int64)
    return np.stack(
        [
            stream.kind,
            stream.selector,
            np.round(stream.flip * 1e7).astype(np.int64),
            stream.use,
            increase,
            stream.orders,
            step_ns,
            shift_ns,
        ],
        axis=1,
    ).astype(np.int64)


def _hashed(rows: np.ndarray) -> np.ndarray:
    """Return a 64-bit hash of each row."""
    multipliers = np.array(
        [
            0x9E3779B97F4A7C15,
            0xC2B2AE3D27D4EB4F,
            0x165667B19E3779F9,
            0x27D4EB2F165667C5,
            0xFF51AFD7ED558CCD,
            0xC4CEB9FE1A85EC53,
            0x94D049BB133111EB,
            0xD6E8FEB86659FD93,
        ],
        dtype=np.uint64,
    )[: rows.shape[1]]
    with np.errstate(over="ignore"):
        mixed = (rows.astype(np.uint64) * multipliers).sum(axis=1, dtype=np.uint64)
        mixed ^= mixed >> np.uint64(31)
        mixed *= np.uint64(0xBF58476D1CE4E5B9)
        mixed ^= mixed >> np.uint64(29)
    return mixed


def _periodic(stream: _Stream, settle_us: float) -> tuple[_Stream, np.ndarray]:
    """Return the events of a stream to play and, per readout, the readout among them whose signal it records.

    What the magnetization goes through is the pulses, the increase of each
    one's phase over the one before, the time from each to the next and the
    shifts between them; a readout records it after the pulse before it. The
    longest stretch of pulses that recurs every ``q`` of them plays its first
    periods until ``settle_us`` has passed, its last ``(length mod q)``
    pulses, and none between. A readout after a skipped pulse records what
    the same readout of the last period played records: after each of its
    pulses, that period reads what any period of the stretch reads there,
    which leaves its pulses acting as they do. The events after the stretch
    play on from the last period, their times and phases moved back by what
    the skipped pulses took and turned through. Where no stretch is long
    enough to skip a period, or two periods read after the same pulse
    differently, every event plays.

    Returns
    -------
    _Stream
        The events that play, with ``readout`` -1 for those read only so that
        a skipped readout has one to record.
    numpy.ndarray
        ``(readouts,)`` per readout of the stream, its index among the played
        ones.
    """
    count = stream.kind.size
    reading = stream.kind == 2
    everything = stream, np.arange(int(reading.sum()))
    pulses = np.flatnonzero(stream.kind == 1)
    if pulses.size < 4:
        return everything

    rows = _signatures(stream)
    acting = rows[pulses].copy()
    acting[:, 5] = np.add.reduceat(stream.orders, pulses)
    following = np.append(stream.time_us[pulses[1:]], stream.time_us[-1])
    acting[:, 6] = np.round(1e3 * (following - stream.time_us[pulses])).astype(np.int64)
    acting[:, 7] = 0
    signature = _hashed(acting)
    middle = pulses.size // 2
    periods = np.unique(np.abs(np.flatnonzero(signature == signature[middle]) - middle))
    best = (0, 0, 0)
    for q in periods[periods > 0][:_PERIODS]:
        q = int(q)
        match = np.concatenate([[False], signature[q:] == signature[:-q], [False]])
        edges = np.flatnonzero(np.diff(match.astype(np.int8)))
        lengths = edges[1::2] - edges[::2]
        longest = int(np.argmax(lengths))
        if int(lengths[longest]) + q > best[0]:
            best = (int(lengths[longest]) + q, int(edges[2 * longest]), q)
        if best[0] >= pulses.size - q:
            break
    length, first, q = best
    if q == 0:
        return everything
    period_us = float(stream.time_us[pulses[first + q]] - stream.time_us[pulses[first]])
    settle = math.ceil(settle_us / max(period_us, 1e-9)) + 1
    whole, rest = divmod(length, q)
    if whole <= settle + 1 or not np.array_equal(
        acting[first : first + length - q], acting[first + q : first + length]
    ):
        return everything
    skip_from, skip_to = first + settle * q, first + length - rest

    # What each pulse reads before the next: each readout's offset, phase
    # increase, shifts and their offset; the same after the same pulse of
    # every period of the stretch, or nothing.
    owner = np.cumsum(stream.kind == 1) - 1
    after = np.arange(count) - np.searchsorted(owner, owner)
    owned = owner[reading]
    reads = rows[reading]
    relative = np.stack(
        [
            np.round(
                1e3
                * (
                    stream.time_us[reading]
                    - stream.time_us[pulses[np.maximum(owned, 0)]]
                )
            ).astype(np.int64),
            reads[:, 4],
            reads[:, 5],
            reads[:, 7],
        ],
        axis=1,
    )
    slots = int(after.max()) + 1
    with np.errstate(over="ignore"):
        powers = np.cumprod(
            np.full(slots, 0x100000001B3, dtype=np.uint64), dtype=np.uint64
        )
        pattern = np.zeros(pulses.size, dtype=np.uint64)
        led = owned >= 0
        np.add.at(
            pattern, owned[led], _hashed(relative[led]) * powers[after[reading][led]]
        )
    line = (np.arange(pulses.size) - first) % q
    rich = np.flatnonzero(pattern != 0)
    rich = rich[(rich >= first) & (rich < first + length)]
    lines, donors = np.unique(line[rich], return_index=True)
    donor_of = np.full(q, -1, dtype=np.int64)
    donor_of[lines] = rich[donors]
    if np.any(pattern[rich] != pattern[donor_of[line[rich]]]):
        return everything

    names = tuple(stream.__dataclass_fields__)
    chunks = [{name: getattr(stream, name)[: pulses[skip_from - q]] for name in names}]
    for at in range(skip_from - q, skip_from):
        own, stop = int(pulses[at]), int(pulses[at + 1])
        donor = int(donor_of[line[at]])
        if pattern[at] != 0 or donor < 0:
            chunks.append({name: getattr(stream, name)[own:stop] for name in names})
            continue
        lead = int(pulses[donor])
        tail = int(pulses[donor + 1]) if donor + 1 < pulses.size else count
        indices = np.concatenate([[own], np.arange(lead + 1, tail)])
        chunk = {name: getattr(stream, name)[indices].copy() for name in names}
        moved = stream.time_us[own] - stream.time_us[lead]
        chunk["orders"][0] = stream.orders[lead]
        chunk["shift_us"][0] = stream.shift_us[lead] + moved
        chunk["time_us"][1:] += moved
        chunk["shift_us"][1:] += moved
        chunk["phase"][1:] += stream.phase[own] - stream.phase[lead]
        chunk["readout"][1:] = -1
        chunks.append(chunk)
    natural = int(pulses[skip_from])
    resume = int(pulses[skip_to]) if skip_to < pulses.size else count
    later = {name: getattr(stream, name)[resume:].copy() for name in names}
    if resume < count:
        moved = stream.time_us[resume] - stream.time_us[natural]
        later["time_us"] -= moved
        later["shift_us"] -= moved
        later["phase"] -= (
            stream.phase[pulses[skip_to - 1]] - stream.phase[pulses[skip_from - 1]]
        )
    chunks.append(later)
    played = _Stream(
        **{name: np.concatenate([c[name] for c in chunks]) for name in names}
    )

    # A readout is keyed by the pulse before it, among the stream's, and its
    # place after that pulse; one after a skipped pulse by the same pulse of
    # the last period played.
    kept = np.concatenate([np.arange(skip_from), np.arange(skip_to, pulses.size)])
    played_owner = np.cumsum(played.kind == 1) - 1
    played_after = np.arange(played.kind.size) - np.searchsorted(
        played_owner, played_owner
    )
    read = played.kind == 2
    led = played_owner[read] >= 0
    played_keys = (
        np.where(led, kept[np.maximum(played_owner[read], 0)] + 1, 0) * slots
        + played_after[read]
    )
    source = np.where(
        (owned >= skip_from) & (owned < skip_to),
        skip_from - q + (owned - first) % q,
        owned,
    )
    wanted = (source + 1) * slots + after[reading]
    column = np.searchsorted(played_keys, wanted)
    found = column < played_keys.size
    if not found.all() or np.any(played_keys[column] != wanted):
        return everything
    return played, column


def _description(stream: _Stream, groups: np.ndarray, device) -> object:
    """Return a stream as a TorchSim description, a train per group.

    Each pulse turns a group through its flip angle times the group's profile
    under the pulse's selector.
    """
    from torchsim.sequence import (
        AdcRole,
        EventAction,
        EventType,
        RfUse,
        SequenceDescription,
        SequenceEvent,
        ideal_rf_definition,
    )

    count = groups.shape[0]
    flips = _LEVELS[groups].astype(np.float32)
    amplitudes: dict[tuple[int, float], object] = {}

    def amplitude(selector: int, flip: float):
        key = (selector, flip)
        if key not in amplitudes:
            scaled = flip * flips[:, selector]
            amplitudes[key] = (
                float(scaled[0])
                if count == 1
                else torch.as_tensor(scaled, dtype=torch.float32, device=device)
            )
        return amplitudes[key]

    uses = {int(use) for use in RfUse}
    events = []
    for kind, when, selector, flip, use, phase, orders, shift_us in zip(
        stream.kind.tolist(),
        stream.time_us.tolist(),
        stream.selector.tolist(),
        stream.flip.tolist(),
        stream.use.tolist(),
        stream.phase.tolist(),
        stream.orders.tolist(),
        stream.shift_us.tolist(),
        strict=True,
    ):
        if kind == 1:
            events.append(
                SequenceEvent.rf(
                    when,
                    0,
                    RfUse(use) if use in uses else RfUse.UNKNOWN,
                    amplitude(selector, flip),
                    phase,
                )
            )
        else:
            events.append(SequenceEvent.adc(when, AdcRole.SINGLE, phase))
        events.extend(
            SequenceEvent(EventType.WAIT, shift_us, (), EventAction.SHIFT_AFTER)
            for _ in range(orders)
        )
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=float(stream.time_us[-1]) if stream.time_us.size else 0.0,
        events=tuple(events),
        rf_definitions={0: ideal_rf_definition()},
    )


class _ReadoutBasis:
    """The decay and precession of each entry across a readout, from its echo, spanned by a basis of the samples' times.

    An entry decays with its T2 from the echo, and dephases with its T2' over
    the time the magnetization has gone unrefocused, which the extended phase
    graphs leave out. It precesses at its frequency from the echo where the
    states carry the field, and over the time the magnetization has gone
    unrefocused where they do not (``from_excitation``).
    """

    def __init__(
        self,
        timeline: Timeline,
        entries: _Entries,
        tolerance: float,
        *,
        from_excitation: bool,
    ) -> None:
        played, readouts = timeline.played, timeline.readouts
        device = entries.frequency.device
        self.device = device
        dwell = played["adc_dwell_ns"][readouts.block].astype(np.int64)
        counts = np.diff(readouts.first)
        unrefocused = np.nan_to_num(timeline.unrefocused_us(readouts.echo_us))
        shapes = np.column_stack(
            [dwell, counts, readouts.echo, np.round(unrefocused).astype(np.int64)]
        )
        unique, self._shape_of = _rows(shapes)
        self._offset = np.concatenate([[0], np.cumsum(unique[:, 1])])
        tau = np.concatenate([1e-9 * d * (np.arange(n) - e) for d, n, e, _ in unique])
        since = np.concatenate([np.full(n, 1e-6 * u) for _, n, _, u in unique]) + tau
        relaxation, relaxation_of = torch.unique(
            torch.stack([entries.t2, entries.t2_prime], dim=1),
            dim=0,
            return_inverse=True,
        )
        self._relaxation_of = relaxation_of
        self._frequency = entries.frequency
        turning = since if from_excitation else tau
        reach = max(float(np.abs(turning).max(initial=0.0)), 1e-6)
        self._step = min(_TABLE_STEP, _TABLED_TURN / reach)
        fitted = min(_SVD_STEP, _FITTED_TURN / reach)
        if entries.frequency.numel():
            low = float(entries.frequency.min())
            high = float(entries.frequency.max())
        else:
            low = high = 0.0
        self._lowest = low - self._step
        steps = math.ceil((high - self._lowest) / self._step) + 2

        def put(values):
            return torch.as_tensor(values, dtype=torch.float32, device=device)

        tau_t, since_t, turning_t = put(tau), put(np.abs(since)), put(turning)
        rates = (1.0 / relaxation.double()).float()

        def kernels(r: int, frequencies: torch.Tensor) -> torch.Tensor:
            decay = torch.exp(-rates[r, 0] * tau_t - rates[r, 1] * since_t)
            cycles = torch.remainder(frequencies[None, :] * turning_t[:, None], 1.0)
            return decay[:, None] * torch.exp(-2j * math.pi * cycles)

        columns = []
        for r in range(relaxation.shape[0]):
            held = entries.frequency[relaxation_of == r]
            cells = torch.unique(torch.round((held - low) / fitted))
            columns.append((r, low + fitted * cells))
        width = max(1, _BUDGET // (8 * max(tau.size, 1)))

        def blocks() -> Iterator[torch.Tensor]:
            for r, frequencies in columns:
                for first in range(0, frequencies.numel(), width):
                    yield kernels(r, frequencies[first : first + width])

        basis = _span(blocks, tau.size, tolerance, device)
        self.terms = basis.shape[1]
        self._basis = basis
        self._table = torch.empty(
            (relaxation.shape[0], steps, self.terms),
            dtype=torch.complex64,
            device=device,
        )
        table = self._lowest + self._step * torch.arange(
            steps, dtype=torch.float64, device=device
        )
        for r in range(relaxation.shape[0]):
            for first in range(0, steps, width):
                frequencies = table[first : first + width].float()
                self._table[r, first : first + width] = (
                    kernels(r, frequencies).T @ basis.conj()
                )

    def coefficients(self, local: torch.Tensor) -> torch.Tensor:
        """Return the coefficients in each term of the excited entries at ``local``, ``(n, terms)``, interpolated in frequency."""
        position = (self._frequency[local] - self._lowest) / self._step
        below = torch.floor(position)
        above = (position - below)[:, None]
        relaxation = self._relaxation_of[local]
        below = below.long()
        return (
            self._table[relaxation, below] * (1.0 - above)
            + self._table[relaxation, below + 1] * above
        )

    def basis(self, readout: np.ndarray, sample: np.ndarray) -> torch.Tensor:
        """Return each term at samples, given their readouts and their indices within them, ``(terms, n)``."""
        rows = self._offset[self._shape_of[readout]] + sample
        return self._basis[torch.as_tensor(rows, device=self.device)].T


class _Grid:
    """A grid along the logical axes a station's trajectory encodes, its images and its coils' sensitivities.

    Point ``i`` along an axis lies ``(i - size // 2) * delta`` from the
    grid's centre.
    """

    def __init__(self, axes, delta, size, centre) -> None:
        self.axes = axes
        self.delta = delta
        self.size = size
        self.centre = centre
        self.shape = tuple(int(n) for n in size[::-1])
        self.points = int(np.prod(size))
        self.images: torch.Tensor | None = None
        self.sensitivities: torch.Tensor | None = None
        self.mix: torch.Tensor | None = None

    @classmethod
    def around(cls, logical: torch.Tensor, spacing: float, reach: np.ndarray) -> _Grid:
        """Return the grid over positions along the logical axes the widest k, ``reach`` in 1/m, encodes across them.

        Its frequencies reach :data:`_OVERSAMPLING` times past ``reach``, and
        at least :data:`_FEWEST_POINTS` points span the positions along each
        axis, so that k between two of its frequencies is read off enough of
        them.
        """
        low = logical.amin(0).double().cpu().numpy() - 0.5 * spacing
        high = logical.amax(0).double().cpu().numpy() + 0.5 * spacing
        extent = high - low
        axes = np.flatnonzero(reach * extent >= 0.5)
        delta = np.minimum(
            0.5 / (_OVERSAMPLING * reach[axes]), extent[axes] / _FEWEST_POINTS
        )
        size = 2 * np.ceil(0.5 * (extent[axes] / delta + 2.0)).astype(np.int64)
        centre = 0.5 * (low + high)
        centre[axes] = np.round(centre[axes] / delta) * delta
        return cls(axes, delta, size, centre)

    def cells(self, logical: torch.Tensor) -> torch.Tensor:
        """Return positions in grid cells from the grid's centre, ``(n, axes)``, its first axis first."""
        axes = torch.as_tensor(self.axes, device=logical.device)
        centre = torch.as_tensor(
            self.centre[self.axes], dtype=torch.float32, device=logical.device
        )
        delta = torch.as_tensor(self.delta, dtype=torch.float32, device=logical.device)
        return (logical[:, axes] - centre) / delta

    def cube(self, axes: np.ndarray, spacing: float, device) -> torch.Tensor:
        """Return the spectrum of a cube ``spacing`` wide along ``axes``, ``(m, 3)`` logical, at the grid's frequencies, one at zero."""
        spectrum = torch.ones(self.shape, dtype=torch.float32, device=device)
        frequencies = [
            torch.as_tensor(
                (np.arange(int(n)) - int(n) // 2) / (float(n) * float(d)),
                dtype=torch.float32,
                device=device,
            )
            for n, d in zip(self.size, self.delta, strict=True)
        ]
        mesh = torch.meshgrid(*frequencies[::-1], indexing="ij")
        for axis in axes:
            along = sum(
                mesh[at] * float(axis[self.axes[len(self.axes) - 1 - at]])
                for at in range(len(mesh))
            )
            spectrum = spectrum * torch.sinc(spacing * along)
        return spectrum

    def trajectory(self, k: np.ndarray) -> np.ndarray:
        """Return k-space locations in the grid's units, ``(n, axes)``, its first axis first."""
        return k[:, self.axes] * (self.size * self.delta)

    def coils(self, tissue: Tissue, rotation: np.ndarray, device) -> torch.Tensor:
        """Return each coil's sensitivity on a coarser grid spanning this one, ``(coils, *coarse)``."""
        extent = self.size * self.delta
        coarse = np.maximum(2, np.ceil(extent / _COIL_SPACING).astype(np.int64) + 1)
        lines = [
            self.centre[axis]
            + delta * np.linspace(-(size // 2), size - 1 - size // 2, int(n))
            for axis, n, delta, size in zip(
                self.axes, coarse, self.delta, self.size, strict=True
            )
        ]
        mesh = np.meshgrid(*lines, indexing="ij")
        logical = np.tile(self.centre, (mesh[0].size if mesh else 1, 1))
        for at, axis in enumerate(self.axes):
            logical[:, axis] = mesh[at].ravel()
        received = tissue.receive(logical @ rotation.T)
        if received is None:
            received = np.ones((logical.shape[0], 1))
        values = np.asarray(received).reshape(*[int(n) for n in coarse], -1)
        values = np.moveaxis(values, -1, 0)
        values = np.transpose(values, (0, *range(len(self.axes), 0, -1)))
        return torch.as_tensor(
            values, dtype=torch.complex64, device=device
        ).contiguous()

    def sensitivity(self, first: int, last: int) -> torch.Tensor:
        """Return the sensitivities of coils ``first`` to before ``last`` on the grid."""
        held = self.sensitivities[first:last]
        if held.shape[1:] == self.shape:
            return held
        mode = {1: "linear", 2: "bilinear", 3: "trilinear"}[len(self.shape)]

        def up(part: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(
                part[None], size=self.shape, mode=mode, align_corners=True
            )[0]

        return torch.complex(up(held.real), up(held.imag))
