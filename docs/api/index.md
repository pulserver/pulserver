# API reference

Three interfaces onto one representation. The Python API designs sequences
and reconstructs data; the C API is what a scanner links against; the C++ API
is the same library with RAII types plus the sequence library and the
reconstruction-side reader.

| Interface | What it is |
|---|---|
| {doc}`python/index` | The design toolbox, the drop-in `pypulseq` layer, the reconstruction stack, and the shipped plugins. |
| {doc}`c/index` | The scanner-side library an interpreter links against. C89, no dependencies. |
| {doc}`cpp/index` | The same library with RAII types, plus the sequence library and the reconstruction-side reader. |

```{toctree}
:hidden:
:maxdepth: 2

python/index
c/index
cpp/index
```

## Which one do I want?

| I want to… | Use |
|---|---|
| build a sequence, write a `.seq`, check it | {doc}`python/design`, {doc}`python/pypulseq` |
| run a ready-made protocol | {doc}`python/apps` |
| reconstruct, or write a reconstruction plugin | {doc}`python/recon`, {doc}`python/apps` |
| read a scan's acquisitions, header or stored file | {doc}`python/mrd` |
| declare what the scanner UI shows | {doc}`python/design`, {doc}`python/apps` |
| write an interpreter that plays `.seq` on hardware | {doc}`c/index` |
| add safety checking to an interpreter I already have | {doc}`c/checks`, {doc}`cpp/checks` |
| show the sequence's parameters on a console | {doc}`c/protocol` |
| read or write a `.seq` from C or C++ | {doc}`c/pulseq`, {doc}`cpp/pulseq` |
| feed a reconstruction service from the sequence file | {doc}`cpp/recon` |
