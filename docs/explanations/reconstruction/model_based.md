# Model-based reconstruction

Pulserver's reconstruction layer is a facade, not another numerical backend.
MRI-NUFFT owns non-Cartesian transforms, DeepInverse owns inverse-problem
machinery and denoisers, and Torch owns CPU/CUDA dispatch. Pulserver supplies
the MRI-specific composition rules needed to make those pieces behave as one
operator.

## Build the forward model by composition

Begin with one acquisition operator and add only the effects present in the
data:

```python
import pulserver.recon as recon

physics = recon.physics.NonCartesian2D(
    trajectory,
    (256, 256),
    coil_maps=coil_maps,
)
physics = recon.physics.OffResonance(physics, field_map, readout_time)
physics = recon.physics.Subspace(physics, basis)
physics = recon.physics.Toeplitz(
    physics,
    support="radial",
    radius=1.0,
    chunk_size=65_536,
    coil_batch_size=1,
)
```

Off-resonance must precede subspace composition. In that order Pulserver can
combine their temporal factors into one coefficient/interpolation-segment
transfer when the spatial interpolation factors are shared. Cartesian FFT
normal operations are already exact, and stacked NUFFT retains its exact
normal operation.

For 3D Cartesian data, apply the exact one-dimensional FFT along the fully
sampled axis and use batched 2D Cartesian physics for the resulting planes;
{func}`pulserver.recon.preprocessing.cartesian_3d_to_2d` performs that first
step.

## Compact Toeplitz normal operators

Base MRI-NUFFT operators use their native Toeplitz kernels. Dynamic subspace
and off-resonance compositions use a Torch-native matrix-valued transfer on
CPU or CUDA. The persistent representation applies three reductions:

- it stores only the Hermitian upper triangle;
- it remains real when the temporal basis permits it;
- with `support="radial"`, it retains only the centered circle or sphere in
  the oversampled Fourier grid.

`support="full"` preserves the complete embedding. `radius` is normalized to
the per-axis Nyquist radius. Radial support assumes the same circular or
spherical filtering in the reconstruction model.

The compact kernel is created lazily by the first normal-operator call. Its
`storage_nbytes`, `dense_nbytes`, and `compression_ratio` attributes expose
the persistent memory footprint. `chunk_size` bounds temporary unpacking
memory, while `coil_batch_size` trades working memory for throughput.

On CPU, the packed Hermitian multiplication uses a small ahead-of-time C++
extension distributed in the wheel. Independent frequency locations are
partitioned across standard C++ threads, while each worker dispatches at
runtime between scalar, AVX2, and AVX-512 implementations for both real and
complex64 kernels. Scanner hosts therefore need neither Numba nor a C++
toolchain. Unsupported dtypes and autograd-enabled tensors retain the Torch
reference path.

On CUDA, `cuda_mode="auto"` chooses a full-residency path when two padded
coefficient banks plus a conservative cuFFT workspace estimate fit below
`cuda_max_device_fraction`. Otherwise it uses a one-volume compact path.
`"resident"` requires the batched path and raises `MemoryError` when it does
not fit; `"compact"` forces the lower-memory implementation.

For a real transfer, the default `cuda_transfer_precision="auto"` selects
BF16 coefficient storage when the GPU supports BF16 natively, halving packed
storage while retaining complex64 spectra and FP32 accumulation. It falls
back to FP32 otherwise. Complex-Hermitian kernels remain complex64: Pulserver
does not cast complex spectra or emulate a complex-BF16 arithmetic type.
Explicit `"float16"` and `"bfloat16"` choices are restricted to real packed
kernels.

All Triton programs parallelize over the large independent frequency-location
axis. The low-memory streamed path stages packed rows directly into the same
fused kernel; it does not expand a location-by-rank-by-rank matrix or call a
small-matrix BLAS routine.

## Reconstruction and regularization

Without a denoiser, {func}`pulserver.recon.algorithms.pics` uses conjugate
gradients to solve

$$
(A^H A + \lambda I)x = A^H y.
$$

With a denoiser it uses plug-and-play FISTA, where `regularization` is the
denoiser threshold or noise level:

```python
regularizers = [
    recon.denoisers.LLR(dimension=3, block_size=8, block_batch_size=1024),
    recon.denoisers.Wavelet(dimension=3, level=3, complex_data=True),
]

coefficients = recon.pics(
    dynamic_kspace,
    physics,
    denoiser=regularizers,
    regularization=0.02,
    polynomial_degree=3,
    iterations=30,
)
```

A denoiser sequence is an equal-weight proximal average. A positive
`polynomial_degree` applies the L2-optimal polynomial preconditioner to the
FISTA gradient. Degree $d$ adds $d$ normal-operator applications per
iteration, so it is most useful when $A^H A$ is much cheaper than denoising.

LLR treats image channels as contrasts or subspace coefficients and applies
nuclear soft-thresholding to local 2D or 3D blocks. It uses the smaller
channel-channel Gram matrix when channels are fewer than block voxels, and
`block_batch_size` bounds its workspace.

## Host-backed CUDA execution

Large scanner reconstructions can keep k-space, coefficient images, optimizer
state, and the compact transfer in host RAM while executing bounded pieces on
the GPU. Pass one {class}`pulserver.recon.execution.CudaStreaming` policy to
the acquisition physics, subspace composition, and reconstruction algorithm:

```python
execution = recon.execution.CudaStreaming(
    streams=2,
    transfer_chunk_size=1_048_576,
    physics_batch_size=1,
    frame_cache_size=2,
    denoiser_slab_size=32,
    denoiser_halo=8,
    result_device="cpu",
)

base = recon.physics.NonCartesian3D(
    trajectory,
    (256, 256, 256),
    coil_maps=cpu_coil_maps,
    streaming=execution,
)
physics = recon.physics.Toeplitz(
    recon.physics.Subspace(base, basis, streaming=execution),
)
coefficients = recon.pics(
    cpu_kspace,
    physics,
    recon.denoisers.LLR(dimension=3, block_size=8),
    regularization=0.01,
    iterations=30,
    stepsize=estimated_stepsize,
    streaming=execution,
)
```

PICS keeps CG/FISTA iterates on CPU. A subspace adjoint projects each acquired
frame immediately, the forward operator emits frame measurements directly to
host storage, and `frame_cache_size` bounds simultaneously live frame-specific
NUFFT plans.

With the default unindexed `device="cuda"`, all visible GPUs participate.
Each GPU owns `streams` transfer/compute streams; independent Cartesian
batches, Toeplitz coil/batch groups, and denoiser slabs are distributed over
those workers. Use `device="cuda:0"` for one GPU or
`devices=("cuda:0", "cuda:1")` to select an explicit subset. Optimizer state
is intentionally not replicated: expensive physics and denoiser calls fan
out and reduce back into the single CPU iteration.

`spectrum_residency` and `kernel_residency` choose host or device storage;
their `"auto"` modes use the configured device-memory fraction. One CUDA
stream minimizes workspace, while two streams overlap pinned transfers and
computation after plans and allocators are warm.

Non-overlapping LLR is exact when slab boundaries align with its block grid.
TV, TGV, and wavelet proximal maps have global spatial coupling, so bounded
slab execution is an overlap approximation. Increase `denoiser_halo` to
reduce boundary error, or omit streaming when the full proximal fits.

The optimized packed CUDA matvec requires Triton. It JIT-compiles and caches a
device/rank specialization on first use; deployment therefore needs a
compatible NVIDIA driver and a writable Triton cache, but not `nvcc` or a CUDA
toolkit on the scanner.

Pulserver's private CUFINUFFT adapter passes Torch tensors directly through
CUFINUFFT's generic CUDA-array interface. `backend="auto"` selects it whenever
CUDA is available and standard MRI-NUFFT FINUFFT otherwise. CuPy is therefore
not installed by `pulserver[recon-cuda]` and is not part of the public
reconstruction API.
