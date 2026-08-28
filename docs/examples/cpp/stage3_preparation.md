# Stage 3 — preparation

The last point at which the scan can be refused, and the only stage that pays
for the whole sequence.

The `Collection` constructor follows the chain, deduplicates the unique
blocks, detects the TR and the segments from block content, and expands the
execution stream.

`CheckPlan` is created explicitly here because several questions are asked of
one sequence. Omitting it gives identical verdicts. What it does and does not
save is on {doc}`../../api/c/checks`.

`save_cache` takes only the sequence path: the cache path comes from it and
`Opts::cache_ext`.

The C counterpart is {doc}`../c/stage3_preparation`.

```{literalinclude} ../../../examples/cpp/stage3_preparation.cpp
:language: cpp
:caption: examples/cpp/stage3_preparation.cpp
```
