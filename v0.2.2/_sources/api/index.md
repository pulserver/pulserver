# API reference

Conceptual background is in {doc}`../explanations/index` and procedures in the
{doc}`user guide <../user-guide/index>`.

| Page | Module | Contents |
| --- | --- | --- |
| {doc}`protocol` | `pulserver.protocol` | Protocol entries, presets and the protocol text blocks. |
| {doc}`design` | `pulserver.design` | {class}`~pulserver.design.SequencePlugin` and its UI entries. |
| {doc}`host` | `pulserver.host` | Design calls and the design store. |
| {doc}`ir` | `pulserver.ir` | Segmentation into the IR cache, and its playout. |
| {doc}`proxy` | `pulserver.proxy` | Reconstruction proxy, design lookup and MRD enrichment. |
| {doc}`recon` | `pulserver.recon` | {class}`~pulserver.recon.ReconPlugin`, its context, buffers and results. |
| {doc}`mrd` | `pulserver.mrd` | MRD acquisitions, header entries, images and sequence definitions. |
| {doc}`validate` | `pulserver.validate` | Playout validation against a recording or the IR cache. |
| {doc}`virtual` | `pulserver.virtual` | Phantoms, the Fourier engine, the scan clock and the reconstruction client. |

The C library a scanner interpreter links is documented in its public headers,
[`src/c/include/pulseg/`](https://github.com/pulserver/pulserver/tree/main/src/c/include/pulseg),
starting from `pulseg.h`.

```{toctree}
:hidden:

protocol
design
host
ir
proxy
recon
mrd
validate
virtual
```
