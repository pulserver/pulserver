# API reference

Conceptual background is in {doc}`../explanations/index` and procedures in the
{doc}`user guide <../user-guide/index>`.

| Page | Module | What it documents |
| --- | --- | --- |
| {doc}`protocol` | `pulserver.protocol` | The entries of a scanner protocol, their presets, and the text blocks that carry them between the design calls and the interpreter. |
| {doc}`design` | `pulserver.design` | {class}`~pulserver.design.ScannerSequence`, which binds a pypulseqpp sequence application to the scanner protocol, and the UI entries it is declared with. |
| {doc}`host` | `pulserver.host` | The design calls of the PSD host processes and the store of the designs they generate. |
| {doc}`ir` | `pulserver.ir` | The segmentation of a sequence chain into the IR cache the interpreter loads. |
| {doc}`vre` | `pulserver.vre` | The reconstruction proxy, the designs it resolves a series to, and the enrichment it applies. |
| {doc}`recon` | `pulserver.recon` | {class}`~pulserver.recon.ReconPlugin`, the context and buffers its hooks receive, and the results they return. |
| {doc}`mrd` | `pulserver.mrd` | MRD acquisitions, header entries and images, and the definitions and readouts a sequence states. |

The C library a scanner interpreter links is documented in its public headers,
[`src/c/include/pulseg/`](https://github.com/pulserver/pulserver/tree/main/src/c/include/pulseg),
starting from `pulseg.h`.

```{toctree}
:hidden:

protocol
design
host
ir
vre
recon
mrd
```
