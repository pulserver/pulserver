# Julia-style resident Toeplitz hot path

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB, 40 W configured power
limit), Torch 2.13.0+cu130, Triton 3.7.1. Each case is a complete eight-coil
SENSE normal with a real Hermitian-packed radial transfer. Four untimed
iterations establish cuFFT plans, Triton autotuning, allocator state, and the
laptop's sustained power state; the table reports the median of four timed
iterations.

| Size / rank / coils | CUDA path | Transfer | Normal (s) | Peak VRAM |
| --- | --- | --- | ---: | ---: |
| 160³ / 5 / 8 | Compact one-volume | FP32 | 1.683 | 3.31 GiB |
| 160³ / 5 / 8 | Resident batched banks | FP32 | 1.393 | 5.39 GiB |
| 96³ / 15 / 8 | Compact one-volume | FP32 | 1.112 | 2.84 GiB |
| 96³ / 15 / 8 | Resident batched banks | FP32 | 0.942 | 4.39 GiB |
| 96³ / 15 / 8 | Resident batched banks | FP16/FP32 accumulate | 0.899 | 3.57 GiB |

The resident FP32 path is 17.3% faster at rank 5 and 15.3% faster at rank 15.
FP16 packed storage provides a further 4.5% at rank 15 and halves the
persistent transfer from 1.67 GiB to 0.84 GiB. It is explicit rather than the
default because it changes the numerical operator.

The resident implementation retains two padded coefficient banks, executes
one batched forward and inverse cuFFT per coil, applies sensitivity factors
directly into/out of those banks, and reads/writes retained frequencies
directly in Triton. The first application benchmarks both a Julia-like
independent-output kernel and a cross-output-reuse kernel, including several
launch configurations; this laptop selected reuse for both ranks. Selection
is cached for subsequent CG/FISTA iterations.

`cuda_mode="auto"` selects resident execution only when both banks, the packed
transfer, the current live allocations, and a conservative extra cuFFT work
bank fit below `cuda_max_device_fraction`. `cuda_mode="resident"` converts an
insufficient-memory choice into a hard `MemoryError`; `"compact"` provides a
reproducible low-memory control.

A cubic projection of the sustained 160³ resident result to 256³ is 5.70 s on
this 40 W laptop configuration. Peak allocated memory projects to roughly
22 GiB. This is not an A100 timing: the 80 GB reference device can retain the
complete problem and has dramatically higher sustained memory bandwidth.
The implementation now supplies the same high-memory schedule needed for a
fair A100 comparison, but Julia parity must be confirmed by running this
benchmark on the target A100 rather than inferred from the laptop.

Reproduce the controls with:

```bash
python docs/_bench/bench_cuda_streaming_sense.py --mode full --size 160 \
  --rank 5 --coils 8 --cuda-mode compact --warmups 4 --repeats 4
python docs/_bench/bench_cuda_streaming_sense.py --mode full --size 160 \
  --rank 5 --coils 8 --cuda-mode resident --warmups 4 --repeats 4
python docs/_bench/bench_cuda_streaming_sense.py --mode full --size 96 \
  --rank 15 --coils 8 --cuda-mode resident \
  --cuda-transfer-precision float16 --warmups 4 --repeats 4
```
