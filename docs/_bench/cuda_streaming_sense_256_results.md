# Coil-aware subspace normal

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB), Intel Core i7-13700H,
Torch 2.13.0+cu130, Triton 3.7.1. The synthetic operator uses a real,
Hermitian-packed, radially supported rank-5 transfer and eight complex SENSE
maps. Zero-filled values make allocation reproducible without removing FFTs,
transfers, packed multiplication, SENSE factors, or accumulation.

| Execution | Size / rank / coils | Runtime | Peak VRAM | Peak RSS | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Host-backed, two streams | 256³ / 5 / 8 | 11.3–12.3 s | 5.12 GiB allocated; 5.27 GiB reserved | 10.6–11.2 GiB | Measured across isolated/repeated runs |
| Full CUDA | 160³ / 5 / 8 | 1.156 s | 3.16 GiB | 1.65 GiB | Measured |
| Full CUDA | 192³ / 5 / 8 | 1.996 s | 5.46 GiB | 2.40 GiB | Measured |
| Full CUDA | 256³ / 5 / 8 | 4.73 s | about 12.9 GiB | — | Cubic extrapolation from 160³ and 192³ |
| CPU | 256³ / 5 / 8 | about 120 s | — | about 25–30 GiB | Extrapolated from measured CPU controls |

The original generic implementation measured 27.01 s and 5.53 GiB allocated
for host-backed execution, and 11.72 s with 7.87 GiB at 160³ for full CUDA.
The fused packed-Hermitian matvec, fixed stream buffers, support-resident
spectra, SENSE factor fusion, and one-volume FFT workspace therefore provide
a 2.2–2.4× streamed speedup and a 10.1× full-offload speedup. Full-offload VRAM
at 160³ fell by about 60%.

The 256³ streamed result is a complete eight-coil normal, representative of
one CG iteration or one ordinary FISTA gradient step; vector updates add a
small cost and denoising is additional. A degree-`d` polynomial FISTA gradient
evaluates the normal `d + 1` times. It excludes persistent CUFINUFFT plans,
the original k-space, solver vectors beyond the input/output, and denoiser
state.

The full-CUDA measurements scale almost exactly with voxel count:
`1.996 / 1.156 = 1.727`, versus `(192 / 160)³ = 1.728`. The 256³ estimate is
therefore 4.73 s on this laptop GPU, requiring roughly 13 GiB. A GPU with at
least 15% more effective throughput reaches the 4.13 s Julia reference for
the smaller K=5/C=8 MRF problem. The 8 GiB path cannot overlap independent
coil workspaces because one bounded coefficient workspace occupies about
5.1 GiB, so dual streams overlap packed-transfer chunks within a coil rather
than running two complete coils concurrently.

One coefficient-space 256³/K=5 transfer measures 1.14 s with two streams
versus 1.98 s with one stream in the isolated three-repeat control, both at
about 5 GiB peak VRAM. A separate explicit-FP16 run measured 0.996 s versus
1.030 s FP32 under the same GPU state. FP16 storage retains FP32 accumulation
and passed the
reference test at `atol=rtol=2e-3`; it remains opt-in because it changes the
operator and provides only a small speed benefit on this PCIe/GPU combination.
