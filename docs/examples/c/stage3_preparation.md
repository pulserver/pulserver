# Stage 3 — preparation

The last point at which the scan can be refused, and the only stage that pays
for the whole sequence.

`pulseg_read` follows the `NextSequence` chain, deduplicates the unique
blocks, detects the TR and the segments from block content, and expands the
execution stream. A `TRID` label is never read.

The check plan is created explicitly because several questions are asked of
one sequence. Passing `NULL` gives identical verdicts. What it does and does
not save is on {doc}`../../api/c/checks`.

`pulseg_save_cache` writes beside the `.seq`, deriving the path from
`opts->cache_ext`. The two stages after this read it.

The C++ counterpart is {doc}`../cpp/stage3_preparation`.

```{literalinclude} ../../../examples/c/stage3_preparation.c
:language: c
:caption: examples/c/stage3_preparation.c
```
