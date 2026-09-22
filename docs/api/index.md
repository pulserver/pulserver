# API reference

| Page | Contents |
| --- | --- |
| {doc}`protocol` | `pulserver.protocol`: protocol parameters, presets, and the text blocks that carry them to the interpreter |
| {doc}`design` | `pulserver.design`: a pypulseqpp application bound to the scanner UI |
| {doc}`ir` | `pulserver.ir`: a sequence segmented into the IR cache the interpreter loads |
| {doc}`host` | `pulserver.host`: design sessions, generated revisions, the host daemon and its client |
| {doc}`mrd` | `pulserver.mrd`: acquisitions, header entries and images, and what a sequence says about its readouts |
| {doc}`recon` | `pulserver.recon`: the contract a reconstruction plugin is written against |
| {doc}`vre` | `pulserver.vre`: the reconstruction proxy, its revisions and the enrichment it attaches |

The C library a scanner interpreter links is documented in its public headers,
[`src/c/include/pulseg/`](https://github.com/pulserver/pulserver/tree/main/src/c/include/pulseg),
starting from `pulseg.h`.

```{toctree}
:hidden:

protocol
design
ir
host
mrd
recon
vre
```
