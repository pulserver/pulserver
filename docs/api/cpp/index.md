# C++ API

C++17 over the C library. Every name here forwards to the C entry point of the
same name with the prefix dropped, with the arguments in the same order; see
the {doc}`C API <../c/index>` for what each call does.

```{toctree}
:maxdepth: 1

types
protocol
file
checks
cache
generation
playout
pulseq
recon
```

````{only} not doxygen
```{note}
The reference on these pages is generated from the headers by Doxygen, which
is not installed in this build. `apt install doxygen` (or the equivalent) and
rebuild to see it.
```
````

## What this layer adds

| | |
|---|---|
| Lifetime | `Collection`, `CheckPlan`, `ChunkPlan` and `Bridge` own their C handles and release them in a destructor. All are move-only. |
| Errors | A negative code and a `pulseg_diagnostic` become a thrown `pulseg::Error` carrying both. Bridge failures throw `pulseg::BridgeError`, which carries `errno`. |
| Values | What the C API returns through a caller-allocated struct and a matching `_free` is returned by value. |

Nothing else differs. Where a page here is short, the {doc}`C page <../c/index>`
is where the meaning is.

## Pages

`types` through `playout` mirror the C pages of the same name. `pulseq` and
`recon` have no C counterpart: the sequence library and the
reconstruction-side reader are C++ only.

| Page | C counterpart |
|---|---|
| {doc}`types` | {doc}`../c/types` |
| {doc}`protocol` | {doc}`../c/protocol` |
| {doc}`file` | {doc}`../c/file` |
| {doc}`checks` | {doc}`../c/checks` |
| {doc}`cache` | {doc}`../c/cache` |
| {doc}`generation` | {doc}`../c/generation` |
| {doc}`playout` | {doc}`../c/playout` |
| {doc}`pulseq` | none |
| {doc}`recon` | none |

## Headers

`pulseg.hpp` includes all of the above. The individual headers are
`types.hpp`, `error.hpp`, `collection.hpp`, `protocol.hpp`, `file.hpp` and
`chunk.hpp`.

The C headers remain available: they are `extern "C"` and `pulseg.hpp`
includes them, so any C entry point can be called directly, with
`Collection::handle()` supplying the raw pointer.
