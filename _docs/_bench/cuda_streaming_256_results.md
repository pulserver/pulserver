| Streams | Normal (s) | Peak VRAM (GiB) | Peak RSS (GiB) | Kernel (GiB) | Dense kernel (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.983 | 4.94 | 7.72 | 4.19 | 25.00 |
| 2 | 1.141 | 5.00 | 7.81 | 4.19 | 25.00 |

The 256³, rank-5 packed radial real transfer is 4.19 GiB versus 25.00 GiB for dense complex storage, a 5.97x reduction. Timings are warmed compact-normal applications with the transfer resident in host RAM and support-restricted spectra selected automatically for the GPU. The two-stream path is 42.5% faster than one stream on NVIDIA GeForce RTX 4060 Laptop GPU with Torch 2.13.0+cu130 while using the same 5.00 GiB peak allocated VRAM. These numbers are one coefficient-space transfer application, equivalent to one coil pass; they are not a complete multi-coil SENSE normal.

The transfer values are zero-filled to make the scanner-scale allocation reproducible; FFTs, support gathering/scattering, host transfers, packed matrix multiplication, and output storage are unchanged.
