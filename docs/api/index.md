# API reference

Three interfaces onto one representation. The Python API designs sequences
and reconstructs data; the C API is what a scanner links against; the C++ API
is the same library with RAII types plus the reconstruction-side reader.

```{toctree}
:maxdepth: 2

python/index
c/index
cpp/index
```

## Which one do I want?

| I want to… | Use |
|---|---|
| build a sequence, write a `.seq`, check it | {doc}`python/design`, {doc}`python/pypulseq` |
| run a ready-made protocol | {doc}`python/seqzoo` |
| reconstruct, or write a reconstruction plugin | {doc}`python/recon`, {doc}`python/reczoo` |
| declare what the scanner UI shows | {doc}`python/protocol` |
| write an interpreter that plays `.seq` on hardware | {doc}`c/index` |
| add safety checking to an interpreter I already have | {doc}`cpp/pulseg` |
| feed a reconstruction service from the sequence file | {doc}`cpp/recon` |
