"""Windows read on the isochromats' lattice on a torch device: sums by a Triton kernel, transforms by cuFINUFFT."""

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
except ImportError:  # without the gpu extra, which LatticeDevice names
    torch = triton = tl = None

__all__ = ["LatticeDevice"]

#: Most bytes a window's lattice sums take at once on the CPU; on a CUDA
#: device, a quarter of the memory free when the first window is read. A
#: window of more coils is summed and transformed a share of its coils at a
#: time.
MEMORY = 1 << 30

#: Lattice transforms kept planned, the most recently used last.
PLANS = 4


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


class LatticeDevice:
    """Reads the windows the isochromat engine reads on a lattice, on a torch device.

    Each isochromat's value at each Chebyshev point is summed onto its
    lattice point for each coil by a Triton kernel, and the sums are
    transformed to the samples by cuFINUFFT on a CUDA device, or by FINUFFT
    on the CPU, where the kernel runs only under Triton's interpreter
    (``TRITON_INTERPRET=1`` before triton is first imported). The engine
    computes everything else about the window, and the state it leaves.

    Parameters
    ----------
    device : str or torch.device, default="cuda"
        The device the sums and transforms run on.
    memory : int, default=None
        Most bytes a window's lattice sums take on the device at once;
        :data:`MEMORY` on the CPU, and a quarter of the memory free when the
        first window is read on a CUDA device, without it.
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

    def __init__(self, device="cuda", *, memory: int | None = None, profile=False):
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
        self._lattices: dict[tuple[int, int], _Lattice] = {}
        self._plans: OrderedDict = OrderedDict()
        self._staged: torch.Tensor | None = None
        self._fetched: torch.Tensor | None = None
        self._pending = None
        self.stages: dict | None = {} if profile else None

    def __call__(self, window) -> bool:
        """Start reading ``window``, as the engine hands it; True, as every window is read.

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

    def _lattice(self, window) -> _Lattice:
        """Return the lattice ``window`` is read on, held for the positions' revision it was found at."""
        key = (window["layout"], window["axes"])
        lattice = self._lattices.get(key)
        if lattice is None:
            # Lattices of positions since moved are not read again.
            self._lattices = {
                held: kept
                for held, kept in self._lattices.items()
                if held[0] == window["layout"]
            }
            lattice = self._lattices[key] = _Lattice(window, self.device)
        return lattice

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
