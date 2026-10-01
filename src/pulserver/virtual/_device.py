"""ADC windows read on a torch device in the isochromat engine's place: on the lattice by Triton sums and cuFINUFFT transforms, and sample by sample by a Triton kernel."""

from __future__ import annotations

import functools
import os
import time
from collections import OrderedDict

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:  # without the gpu extra, which WindowDevice names
    torch = triton = tl = None

__all__ = ["WindowDevice"]

#: Most bytes a window's lattice sums take at once on the CPU; on a CUDA
#: device, a quarter of the memory free when the first window is read. A
#: window of more coils is summed and transformed a share of its coils at a
#: time.
MEMORY = 1 << 30

#: Lattice transforms kept planned, the most recently used last.
PLANS = 4

#: Isochromats the sample sums take at a time, consecutive in an order that
#: keeps each such block compact in space, and how many of them each of the
#: kernel's matrix products sums over.
BLOCK = 64
CHUNK = 16

#: Least isochromats times samples times coils of a window a CUDA device
#: reads sample by sample; the engine reads smaller ones faster than the
#: device starts.
SMALLEST = 1 << 22

#: Engines whose isochromats a device holds, the most recently read last.
ENGINES = 2


if triton is not None:

    @triton.jit
    def _lattice_sums(
        starts_ptr,
        magnetization_ptr,
        shift_ptr,
        decay_of_ptr,
        decay_ptr,
        nodes_ptr,
        receive_ptr,
        receive_stride,
        sums_ptr,
        points,
        coils,
        count,
        POINTS: tl.constexpr,
        ISOCHROMATS: tl.constexpr,
        COILS: tl.constexpr,
        NODES: tl.constexpr,
    ):
        """Sum the isochromats of POINTS lattice points into COILS coils' sums at the Chebyshev points.

        Positions count the isochromats in lattice order. Each isochromat's value
        at Chebyshev point l is its transverse magnetisation times its decay at
        the point times exp(2 pi i shift t_l), summed into
        sums[coil][l][point], real and imaginary parts interleaved.
        """
        real = magnetization_ptr.dtype.element_ty
        coil = tl.program_id(1) * COILS + tl.arange(0, COILS)
        node = tl.arange(0, NODES)
        nodes = tl.load(nodes_ptr + node)
        for p in range(POINTS):
            q = tl.program_id(0) * POINTS + p
            if q < points:
                start = tl.load(starts_ptr + q)
                end = tl.load(starts_ptr + q + 1)
                sum_re = tl.zeros((COILS, NODES), dtype=real)
                sum_im = tl.zeros((COILS, NODES), dtype=real)
                for first in range(start, end, ISOCHROMATS):
                    j = first + tl.arange(0, ISOCHROMATS)
                    held = j < end
                    m_re = tl.load(magnetization_ptr + 2 * j, mask=held, other=0.0)
                    m_im = tl.load(magnetization_ptr + 2 * j + 1, mask=held, other=0.0)
                    shift = tl.load(shift_ptr + j, mask=held, other=0.0)
                    kind = tl.load(decay_of_ptr + j, mask=held, other=0)
                    decay = tl.load(decay_ptr + kind[:, None] * NODES + node[None, :])
                    turn = 6.283185307179586 * (shift[:, None] * nodes[None, :])
                    c = decay * tl.cos(turn)
                    s = decay * tl.sin(turn)
                    f_re = m_re[:, None] * c - m_im[:, None] * s
                    f_im = m_re[:, None] * s + m_im[:, None] * c
                    at = j[:, None].to(tl.int64) * receive_stride + 2 * coil[None, :]
                    read = held[:, None] & (coil[None, :] < coils)
                    r_re = tl.trans(tl.load(receive_ptr + at, mask=read, other=0.0))
                    r_im = tl.trans(tl.load(receive_ptr + at + 1, mask=read, other=0.0))
                    sum_re = tl.dot(r_re, f_re, sum_re, input_precision="ieee")
                    sum_re = tl.dot(-r_im, f_im, sum_re, input_precision="ieee")
                    sum_im = tl.dot(r_re, f_im, sum_im, input_precision="ieee")
                    sum_im = tl.dot(r_im, f_re, sum_im, input_precision="ieee")
                out = 2 * (
                    (coil[:, None].to(tl.int64) * count + node[None, :]) * points + q
                )
                written = (coil[:, None] < coils) & (node[None, :] < count)
                tl.store(sums_ptr + out, sum_re, mask=written)
                tl.store(sums_ptr + out + 1, sum_im, mask=written)


if triton is not None:

    @triton.jit
    def _cycle(u):
        """Return cos(2 pi u) and sin(2 pi u) for u in [-1/2, 1/2], by polynomials on the quarter turn nearest it, to within about 1e-7 in single precision."""
        q = tl.floor(4.0 * u + 0.5)
        x = 6.283185307179586 * (u - 0.25 * q)
        x2 = x * x
        s = x * (
            1.0
            + x2
            * (
                -0.16666666666666666
                + x2
                * (
                    0.008333333333333333
                    + x2 * (-0.0001984126984126984 + x2 * 2.755731922398589e-06)
                )
            )
        )
        c = 1.0 + x2 * (
            -0.5
            + x2
            * (
                0.041666666666666664
                + x2 * (-0.001388888888888889 + x2 * 2.48015873015873e-05)
            )
        )
        k = (q.to(tl.int32) + 4) % 4
        cos = tl.where(k == 0, c, tl.where(k == 1, -s, tl.where(k == 2, -c, s)))
        sin = tl.where(k == 0, s, tl.where(k == 1, c, tl.where(k == 2, -s, -c)))
        return cos, sin

    @triton.jit
    def _sample_sums(
        offsets_ptr,
        centres_ptr,
        shift_ptr,
        rate_ptr,
        magnetization_ptr,
        receive_ptr,
        k_ptr,
        base_ptr,
        time_ptr,
        partial_ptr,
        padded,
        samples,
        coils,
        blocks_per_split,
        blocks,
        BLOCK: tl.constexpr,
        CHUNK: tl.constexpr,
        SAMPLES: tl.constexpr,
        COILS: tl.constexpr,
        ONE: tl.constexpr,
        RECEIVE: tl.constexpr,
        SINGLE: tl.constexpr,
    ):
        """Sum each coil's sensitivity times every isochromat's transverse magnetisation at SAMPLES samples, over one split of the isochromats.

        The isochromats come in blocks of BLOCK, each block's centre in
        float64 and each isochromat's offset from it, and are summed CHUNK
        at a time. An isochromat's turn
        at a sample, in cycles, is its offset times the sample's k, plus its
        off-resonance from the window's middle times the time, in the sums'
        precision, plus the block centre's turn and the middle off-resonance's,
        in float64 and reduced to one cycle: rounded in proportion to the
        offset rather than to the position. The split's sums are written to
        partial[split][coil][sample], real and imaginary parts interleaved.
        """
        real = offsets_ptr.dtype.element_ty
        s = tl.program_id(0) * SAMPLES + tl.arange(0, SAMPLES)
        sampled = s < samples
        kx = tl.load(k_ptr + s, mask=sampled, other=0.0)
        ky = tl.load(k_ptr + samples + s, mask=sampled, other=0.0)
        kz = tl.load(k_ptr + 2 * samples + s, mask=sampled, other=0.0)
        base = tl.load(base_ptr + s, mask=sampled, other=0.0)
        t = tl.load(time_ptr + s, mask=sampled, other=0.0)
        kx_r, ky_r, kz_r = kx.to(real), ky.to(real), kz.to(real)
        coil = tl.program_id(1) * COILS + tl.arange(0, COILS)
        received = coil < coils
        split = tl.program_id(2)
        first = split * blocks_per_split
        last = tl.minimum(first + blocks_per_split, blocks)
        if ONE:
            sum_re = tl.zeros((SAMPLES,), dtype=real)
            sum_im = tl.zeros((SAMPLES,), dtype=real)
        else:
            sum_re = tl.zeros((COILS, SAMPLES), dtype=real)
            sum_im = tl.zeros((COILS, SAMPLES), dtype=real)
        for b in range(first, last):
            centre = (
                tl.load(centres_ptr + 3 * b) * kx
                + tl.load(centres_ptr + 3 * b + 1) * ky
                + tl.load(centres_ptr + 3 * b + 2) * kz
                + base
            )
            centre = (centre - tl.floor(centre + 0.5)).to(real)
            for q in range(0, BLOCK, CHUNK):
                j = b * BLOCK + q + tl.arange(0, CHUNK)
                ox = tl.load(offsets_ptr + j)
                oy = tl.load(offsets_ptr + padded + j)
                oz = tl.load(offsets_ptr + 2 * padded + j)
                shift = tl.load(shift_ptr + j)
                rate = tl.load(rate_ptr + j)
                m_re = tl.load(magnetization_ptr + j)
                m_im = tl.load(magnetization_ptr + padded + j)
                cycles = (
                    ox[:, None] * kx_r[None, :]
                    + oy[:, None] * ky_r[None, :]
                    + oz[:, None] * kz_r[None, :]
                    + shift[:, None] * t[None, :]
                    + centre[None, :]
                )
                cycles = cycles - tl.floor(cycles + 0.5)
                if SINGLE:
                    cos, sin = _cycle(cycles)
                else:
                    cos = tl.cos(6.283185307179586 * cycles)
                    sin = tl.sin(6.283185307179586 * cycles)
                decay = tl.exp(-rate[:, None] * t[None, :])
                e_re = decay * cos
                e_im = -decay * sin
                if ONE:
                    if RECEIVE:
                        r_re = tl.load(receive_ptr + j)
                        r_im = tl.load(receive_ptr + padded + j)
                        a_re = r_re * m_re - r_im * m_im
                        a_im = r_re * m_im + r_im * m_re
                    else:
                        a_re = m_re
                        a_im = m_im
                    sum_re += tl.sum(
                        a_re[:, None] * e_re - a_im[:, None] * e_im, axis=0
                    )
                    sum_im += tl.sum(
                        a_re[:, None] * e_im + a_im[:, None] * e_re, axis=0
                    )
                else:
                    at = coil[:, None].to(tl.int64) * padded + j[None, :]
                    r_re = tl.load(receive_ptr + at, mask=received[:, None], other=0.0)
                    r_im = tl.load(
                        receive_ptr + coils * padded + at,
                        mask=received[:, None],
                        other=0.0,
                    )
                    a_re = r_re * m_re[None, :] - r_im * m_im[None, :]
                    a_im = r_re * m_im[None, :] + r_im * m_re[None, :]
                    sum_re = tl.dot(a_re, e_re, sum_re, input_precision="ieee")
                    sum_re = tl.dot(-a_im, e_im, sum_re, input_precision="ieee")
                    sum_im = tl.dot(a_re, e_im, sum_im, input_precision="ieee")
                    sum_im = tl.dot(a_im, e_re, sum_im, input_precision="ieee")
        if ONE:
            out = 2 * (split.to(tl.int64) * samples + s)
            tl.store(partial_ptr + out, sum_re, mask=sampled)
            tl.store(partial_ptr + out + 1, sum_im, mask=sampled)
        else:
            out = 2 * (
                (split.to(tl.int64) * coils + coil[:, None]) * samples + s[None, :]
            )
            written = received[:, None] & sampled[None, :]
            tl.store(partial_ptr + out, sum_re, mask=written)
            tl.store(partial_ptr + out + 1, sum_im, mask=written)


@functools.cache
def _tuned_sample_sums():
    """Return the sample sums autotuned on a CUDA device, the first window of each shape and precision, over 16 to 128 samples per program and 4 or 8 warps."""
    configs = [
        triton.Config({"SAMPLES": samples}, num_warps=warps)
        for samples in (16, 32, 64, 128)
        for warps in (4, 8)
    ]
    return triton.autotune(configs=configs, key=["coils", "samples"])(_sample_sums)


def _sum_samples(
    arguments: tuple, samples: int, coil_blocks: int, splits: int, constants: dict
) -> None:
    """Launch the sample sums: in the interpreter's configuration, or in the one autotuned for their shape on a CUDA device."""
    if _interpreted():
        programs = (triton.cdiv(samples, 32), coil_blocks, splits)
        _sample_sums[programs](*arguments, SAMPLES=32, **constants)
        return

    def tuned(meta):
        return (triton.cdiv(samples, meta["SAMPLES"]), coil_blocks, splits)

    _tuned_sample_sums()[tuned](*arguments, **constants)


@functools.cache
def _tuned_lattice_sums():
    """Return the kernel autotuned on a CUDA device, the first window of each shape, over 4 or 1 lattice points per program and 4 or 8 warps."""
    configs = [
        triton.Config({"POINTS": points}, num_warps=warps)
        for points in (4, 1)
        for warps in (4, 8)
    ]
    return triton.autotune(configs=configs, key=["coils", "count"])(_lattice_sums)


def _sum_onto_lattice(arguments: tuple, points: int, coils: int, nodes: int) -> None:
    """Launch the lattice sums of ``coils`` coils: in the interpreter's configuration, or in the one autotuned for their shape on a CUDA device."""
    constants = {"ISOCHROMATS": 16, "COILS": 16, "NODES": nodes}
    if _interpreted():
        programs = (triton.cdiv(points, 4), triton.cdiv(coils, 16))
        _lattice_sums[programs](*arguments, POINTS=4, **constants)
        return

    def tuned(meta):
        return (triton.cdiv(points, meta["POINTS"]), triton.cdiv(coils, 16))

    _tuned_lattice_sums()[tuned](*arguments, **constants)


def _interpreted() -> bool:
    try:
        return bool(triton.knobs.runtime.interpret)
    except AttributeError:
        return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


def _upsampling(tolerance: float) -> float:
    """FINUFFT's upsampling, as the engine plans it."""
    return 1.25 if tolerance >= 1e-9 else 2.0


def _host(array: np.ndarray, dtype=None) -> torch.Tensor:
    """Copy one of the engine's arrays into a tensor of its own."""
    return torch.from_numpy(np.array(array, dtype=dtype, copy=True))


class _Lattice:
    """A lattice's order and the isochromats' properties in it, held on the device."""

    def __init__(self, window, device: torch.device):
        self.order = _host(window["order"], np.int64).to(device)
        self.starts = _host(window["starts"], np.int32).to(device)
        self.off_resonance = _host(window["off_resonance"]).to(device)
        self.decay_of = _host(window["decay_of"], np.int32).to(device)
        self.receive = {}

    def sensitivities(
        self, window, complex_dtype, device: torch.device
    ) -> torch.Tensor:
        """Each isochromat's sensitivities in lattice order, its coils together, as interleaved reals."""
        held = self.receive.get(complex_dtype)
        if held is not None:
            return held
        n = len(self.order)
        coils = int(window["coils"])
        if window["receive_re"] is None:
            values = torch.ones((n, 1), dtype=complex_dtype, device=device)
        else:
            values = torch.empty((n, coils), dtype=complex_dtype, device=device)
            for c in range(coils):
                re = _host(window["receive_re"][c]).to(device)[self.order]
                im = _host(window["receive_im"][c]).to(device)[self.order]
                values[:, c] = torch.complex(re, im).to(complex_dtype)
        held = torch.view_as_real(values)
        self.receive[complex_dtype] = held
        return held


def _spread_bits(v: torch.Tensor) -> torch.Tensor:
    """Spread the 21 low bits of ``v`` to every third bit."""
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    return (v | (v << 2)) & 0x1249249249249249


def _z_order(positions: torch.Tensor) -> torch.Tensor:
    """Return the order of ``(n, 3)`` positions along a Z-order curve, on which consecutive points lie close together."""
    low = positions.amin(dim=0)
    extent = float((positions.amax(dim=0) - low).max())
    cells = ((positions - low) * ((2**21 - 1) / max(extent, 1e-300))).round()
    cells = cells.to(torch.int64)
    key = (
        _spread_bits(cells[:, 0])
        | (_spread_bits(cells[:, 1]) << 1)
        | (_spread_bits(cells[:, 2]) << 2)
    )
    return torch.argsort(key)


def _padded(values: torch.Tensor, length: int) -> torch.Tensor:
    """Return ``values`` along its last axis, then zeros up to ``length``."""
    out = values.new_zeros((*values.shape[:-1], length))
    out[..., : values.shape[-1]] = values
    return out


class _Engine:
    """What a device holds of one engine's isochromats: their lattices and layouts at the positions' latest revision, and their properties other than positions, which an engine keeps."""

    def __init__(self, device: torch.device):
        self.device = device
        self.layout = None
        self.lattices: dict[int, _Lattice] = {}
        self.layouts: dict[torch.dtype, _Layout] = {}
        self.off_resonance = None
        self.rate = None
        self.receive: dict[torch.dtype, torch.Tensor] = {}

    def at(self, layout: int) -> _Engine:
        """Let go of the lattices and layouts of positions since moved."""
        if layout != self.layout:
            self.layout = layout
            self.lattices.clear()
            self.layouts.clear()
        return self

    def properties(self, window) -> None:
        """Hold the off-resonances and T2 rates a window read sample by sample hands over, on first use."""
        if self.rate is None:
            self.off_resonance = _host(window["off_resonance"]).to(self.device)
            self.rate = _host(window["rates"]).to(self.device)[
                _host(window["decay_of"], np.int64).to(self.device)
            ]

    def sensitivities(self, window, complex_dtype) -> torch.Tensor | None:
        """``(coils, n)`` receive sensitivities in ``complex_dtype``; None for one coil of unit sensitivity."""
        if window["receive_re"] is None:
            return None
        held = self.receive.get(complex_dtype)
        if held is None:
            held = torch.complex(
                _host(window["receive_re"]), _host(window["receive_im"])
            ).to(self.device, complex_dtype)
            self.receive[complex_dtype] = held
        return held


class _Layout:
    """The isochromats of one revision of the positions in Z-order, in blocks of :data:`BLOCK`, on the device.

    Each block's centre is held in float64 and each isochromat's offset from
    it, its off-resonance from the window's middle, its T2 rate and its
    sensitivities in the sums' precision; the last block is filled with
    isochromats of no magnetisation at the last one's position.
    """

    def __init__(self, window, engine: _Engine, real, complex_dtype):
        device = engine.device
        engine.properties(window)
        positions = torch.from_numpy(
            np.stack([window["x"], window["y"], window["z"]], axis=1)
        ).to(device)
        self.order = _z_order(positions)
        self.count = len(self.order)
        self.blocks = triton.cdiv(self.count, BLOCK)
        self.padded = self.blocks * BLOCK
        placed = positions[self.order]
        placed = torch.cat(
            [placed, placed[-1:].expand(self.padded - self.count, 3)]
        ).reshape(self.blocks, BLOCK, 3)
        self.centres = (0.5 * (placed.amin(dim=1) + placed.amax(dim=1))).contiguous()
        self.offsets = (
            (placed - self.centres[:, None, :])
            .reshape(self.padded, 3)
            .T.to(real)
            .contiguous()
        )
        shift = engine.off_resonance[self.order] - float(window["frequency"])
        self.shift = _padded(shift, self.padded).to(real)
        self.rate = _padded(engine.rate[self.order], self.padded).to(real)
        receive = engine.sensitivities(window, complex_dtype)
        self.receive = (
            None
            if receive is None
            else _padded(
                torch.view_as_real(receive[:, self.order]).permute(2, 0, 1), self.padded
            ).contiguous()
        )

    def magnetization(self, staged: torch.Tensor) -> torch.Tensor:
        """Return the ``(2, n)`` transverse magnetisation in the layout's order, ``(2, padded)``."""
        return _padded(staged[:, self.order], self.padded).contiguous()


class WindowDevice:
    """Reads the windows the isochromat engine hands it, on a torch device.

    A window under a changing gradient whose isochromats lie on a lattice is
    read there: each isochromat's value at each Chebyshev point is summed
    onto its lattice point for each coil by a Triton kernel, and the sums
    are transformed to the samples by cuFINUFFT on a CUDA device, or by
    FINUFFT on the CPU. Any other window outside a run, under a held
    gradient or with its isochromats off any lattice, is summed sample by
    sample by a second Triton kernel: each coil's sensitivity times each
    isochromat's transverse magnetisation, turned and decayed to every
    sample, the turn of each isochromat rounded in proportion to its offset
    from the centre of a block of isochromats close together in space. On
    the CPU the kernels run only under Triton's interpreter
    (``TRITON_INTERPRET=1`` before triton is first imported). The engine
    computes everything else about a window, and the state it leaves.

    Parameters
    ----------
    device : str or torch.device, default="cuda"
        The device the sums and transforms run on.
    memory : int, default=None
        Most bytes a window's lattice sums take on the device at once;
        :data:`MEMORY` on the CPU, and a quarter of the memory free when the
        first window is read on a CUDA device, without it.
    smallest : int, default=None
        Least isochromats times samples times coils of a window read sample
        by sample; smaller windows are left to the engine. :data:`SMALLEST`
        on a CUDA device and 0 on the CPU without it.
    profile : bool, default=False
        Time each stage of every window, waiting for the device between
        them, into :attr:`stages`, in s.

    Raises
    ------
    ImportError
        Without torch and Triton, or on a CUDA device without cuFINUFFT.
    RuntimeError
        On the CPU outside Triton's interpreter.
    """

    def __init__(
        self,
        device="cuda",
        *,
        memory: int | None = None,
        smallest: int | None = None,
        profile=False,
    ):
        if triton is None:
            raise ImportError(
                "windows read on a device need torch and Triton: install the gpu extra"
            )
        self.device = torch.device(device)
        self.memory = memory
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            try:
                import cufinufft
            except ImportError as error:
                raise ImportError(
                    "windows read on a CUDA device are transformed by cuFINUFFT: "
                    "install the cufinufft package"
                ) from error
            self._library = cufinufft
        elif self.device.type == "cpu":
            if not _interpreted():
                raise RuntimeError(
                    "the lattice sums run on the CPU only under Triton's interpreter: "
                    "set TRITON_INTERPRET=1 before triton is first imported"
                )
            import finufft

            self._library = finufft
        else:
            raise ValueError(f"no lattice transforms on a {self.device.type} device")
        if smallest is None:
            smallest = SMALLEST if self.device.type == "cuda" else 0
        self.smallest = smallest
        self._engines: OrderedDict[int, _Engine] = OrderedDict()
        self._processors = None
        self._plans: OrderedDict = OrderedDict()
        self._staged: torch.Tensor | None = None
        self._fetched: torch.Tensor | None = None
        self._pending = None
        self.stages: dict | None = {} if profile else None

    def lattice(self, window) -> bool:
        """Start reading ``window`` on its lattice, as the engine hands it; True, as every such window is read.

        The arrays the engine hands are read before this returns, and its
        ``out`` is written by :meth:`finish`.
        """
        began = self._clock()
        single = bool(window["single"])
        real = torch.float32 if single else torch.float64
        complex_dtype = torch.complex64 if single else torch.complex128
        lattice = self._lattice(window)
        receive = lattice.sensitivities(window, complex_dtype, self.device)

        magnetization = self._stage(window, real)[:, lattice.order].T.contiguous()
        shift = (window["frequency"] - lattice.off_resonance).to(real)
        segments = len(window["nodes"])
        nodes_padded = max(16, triton.next_power_of_2(segments))
        nodes = torch.zeros(nodes_padded, dtype=real)
        nodes[:segments] = _host(window["nodes"])
        decay = torch.zeros((len(window["decay"]), nodes_padded), dtype=real)
        decay[:, :segments] = _host(window["decay"])
        nodes, decay = nodes.to(self.device), decay.to(self.device)
        basis = _host(window["basis"]).to(self.device).to(real)
        modes = tuple(int(m) for m in window["modes"])
        points = int(np.prod(modes))
        x = [_host(axis).to(self.device).to(real) for axis in window["x"]]
        samples = len(window["basis"])
        coils = receive.shape[1]
        began = self._lap("upload", began)

        at_once = self._coils_at_once(
            coils, segments * points * 2 * receive.element_size()
        )
        read = torch.empty((coils, samples), dtype=complex_dtype, device=self.device)
        for first in range(0, coils, at_once):
            k = min(at_once, coils - first)
            sums = torch.empty((k, segments, points, 2), dtype=real, device=self.device)
            arguments = (
                lattice.starts,
                magnetization,
                shift,
                lattice.decay_of,
                decay,
                nodes,
                receive[:, first:],
                receive.stride(0),
                sums,
                points,
                k,
                segments,
            )
            _sum_onto_lattice(arguments, points, k, nodes_padded)
            began = self._lap("sums", began)
            values = self._transform(
                torch.view_as_complex(sums).reshape(k * segments, *modes[::-1]),
                x,
                modes,
                float(window["tolerance"]),
                complex_dtype,
            ).reshape(k, segments, samples)
            began = self._lap("transform", began)
            read[first : first + k] = (values * basis.T[None]).sum(dim=1)
            began = self._lap("basis", began)
        self._pending = (window["out"], self._fetch(read))
        return True

    def samples(self, window) -> bool:
        """Start summing ``window`` sample by sample, as the engine hands it; False where it is empty or smaller than :attr:`smallest`.

        The arrays the engine hands are read before this returns, and its
        ``out`` is written by :meth:`finish`.
        """
        coils = int(window["coils"])
        samples = len(window["time"])
        work = len(window["x"]) * samples * coils
        if work == 0 or work < self.smallest:
            return False
        began = self._clock()
        single = bool(window["single"])
        real = torch.float32 if single else torch.float64
        complex_dtype = torch.complex64 if single else torch.complex128
        layout = self._layout(window, real, complex_dtype)
        magnetization = layout.magnetization(self._stage(window, real))
        k = _host(np.ascontiguousarray(window["k"].T)).to(self.device)
        times = _host(window["time"])
        base = (float(window["frequency"]) * times).to(self.device)
        t = times.to(self.device, real)
        began = self._lap("upload", began)

        one = coils == 1
        coils_per_program = (
            1 if one else min(64, max(16, triton.next_power_of_2(coils)))
        )
        coil_blocks = triton.cdiv(coils, coils_per_program)
        splits = self._splits(layout.blocks, triton.cdiv(samples, 32) * coil_blocks)
        per_split = triton.cdiv(layout.blocks, splits)
        splits = triton.cdiv(layout.blocks, per_split)
        partial = torch.empty(
            (splits, coils, samples, 2), dtype=real, device=self.device
        )
        receive = layout.receive if layout.receive is not None else magnetization
        arguments = (
            layout.offsets,
            layout.centres,
            layout.shift,
            layout.rate,
            magnetization,
            receive,
            k,
            base,
            t,
            partial,
            layout.padded,
            samples,
            coils,
            per_split,
            layout.blocks,
        )
        constants = {
            "BLOCK": BLOCK,
            "CHUNK": CHUNK,
            "COILS": coils_per_program,
            "ONE": one,
            "RECEIVE": layout.receive is not None,
            "SINGLE": single,
        }
        _sum_samples(arguments, samples, coil_blocks, splits, constants)
        read = torch.view_as_complex(partial.sum(dim=0))
        began = self._lap("samples", began)
        self._pending = (window["out"], self._fetch(read))
        return True

    def _engine(self, window) -> _Engine:
        """Return what the device holds of the engine handing ``window``, at the positions the window is read at."""
        engine = self._engines.pop(window["engine"], None)
        if engine is None:
            engine = _Engine(self.device)
        self._engines[window["engine"]] = engine
        while len(self._engines) > ENGINES:
            self._engines.popitem(last=False)
        return engine.at(window["layout"])

    def _lattice(self, window) -> _Lattice:
        """Return the lattice ``window`` is read on, held for the positions' revision it was found at."""
        engine = self._engine(window)
        lattice = engine.lattices.get(window["axes"])
        if lattice is None:
            lattice = engine.lattices[window["axes"]] = _Lattice(window, self.device)
        return lattice

    def _layout(self, window, real, complex_dtype) -> _Layout:
        """Return the isochromats of ``window`` in Z-order, held for the positions' revision and precision it is read at."""
        engine = self._engine(window)
        layout = engine.layouts.get(real)
        if layout is None:
            layout = engine.layouts[real] = _Layout(window, engine, real, complex_dtype)
        return layout

    def _splits(self, blocks: int, programs: int) -> int:
        """Return how many splits of the isochromats ``programs`` programs per split sum, to keep the device's processors busy; two on the CPU, where the interpreter runs them one after another."""
        if self.device.type != "cuda":
            return min(blocks, 2)
        if self._processors is None:
            self._processors = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
        return max(1, min(blocks, triton.cdiv(4 * self._processors, programs)))

    def _coils_at_once(self, coils: int, per_coil: int) -> int:
        """Return how many of ``coils`` coils' lattice sums, ``per_coil`` bytes each, fit in :attr:`memory` together; at least one."""
        if self.memory is None:
            self.memory = (
                torch.cuda.mem_get_info(self.device)[0] // 4
                if self.device.type == "cuda"
                else MEMORY
            )
        return max(1, min(coils, self.memory // max(per_coil, 1)))

    def finish(self) -> None:
        """Write the window :meth:`__call__` started reading into the ``out`` it was handed."""
        began = self._clock()
        out, (fetched, done) = self._pending
        self._pending = None
        if done is not None:
            done.synchronize()
        out[...] = fetched.numpy()
        self._lap("download", began)
        if self.stages is not None:
            self.stages["windows"] = self.stages.get("windows", 0) + 1

    def _fetch(self, read: torch.Tensor):
        """Start copying ``read`` to the host: the copy, and the event that marks it done, if any."""
        if self.device.type != "cuda":
            return read, None
        if (
            self._fetched is None
            or self._fetched.shape != read.shape
            or self._fetched.dtype != read.dtype
        ):
            self._fetched = torch.empty(read.shape, dtype=read.dtype, pin_memory=True)
        self._fetched.copy_(read, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        return self._fetched, done

    def _clock(self) -> float | None:
        if self.stages is None:
            return None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _lap(self, stage: str, began: float | None) -> float | None:
        """Add the time since ``began`` to ``stage``'s, where the stages are timed."""
        if began is None:
            return None
        now = self._clock()
        self.stages[stage] = self.stages.get(stage, 0.0) + now - began
        return now

    def _stage(self, window, real) -> torch.Tensor:
        """Put the transverse magnetisation, (2, isochromats), on the device in ``real``.

        On a CUDA device it passes through a pinned buffer of the window's
        precision, which the next window overwrites only after this one is
        read.
        """
        n = len(window["mx"])
        if self.device.type != "cuda":
            return torch.from_numpy(np.stack((window["mx"], window["my"]))).to(real)
        if (
            self._staged is None
            or self._staged.shape[1] != n
            or self._staged.dtype != real
        ):
            self._staged = torch.empty((2, n), dtype=real, pin_memory=True)
        staged = self._staged.numpy()
        np.copyto(staged[0], window["mx"], casting="same_kind")
        np.copyto(staged[1], window["my"], casting="same_kind")
        return self._staged.to(self.device, non_blocking=True)

    def _transform(self, modes, x, shape, tolerance, complex_dtype):
        """Transform each vector of ``modes`` from the lattice to the samples' coordinates ``x``, lowest axis first."""
        vectors = modes.shape[0]
        key = (shape, vectors, tolerance, complex_dtype)
        plan = self._plans.pop(key, None)
        dtype = "complex64" if complex_dtype == torch.complex64 else "complex128"
        if plan is None:
            options = {"upsampfac": _upsampling(tolerance)}
            if self.device.type == "cuda":
                options["gpu_device_id"] = self.device.index
            else:
                options.update(nthreads=1, showwarn=0)
            plan = self._library.Plan(
                2, shape[::-1], n_trans=vectors, eps=tolerance, dtype=dtype, **options
            )
        self._plans[key] = plan
        while len(self._plans) > PLANS:
            self._plans.popitem(last=False)
        if self.device.type == "cuda":
            plan.setpts(*x[::-1])
            return torch.as_tensor(plan.execute(modes), device=self.device)
        plan.setpts(*(axis.numpy() for axis in x[::-1]))
        return torch.from_numpy(plan.execute(modes.numpy()))
