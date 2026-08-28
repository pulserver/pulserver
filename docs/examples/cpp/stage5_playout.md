# Stage 5 — playout

The scan loop.

A block definition is shared by every occurrence of it; a block instance is
one occurrence, with the amplitudes it plays at and its rotation.

The cursor rests before the first block: `cursor_advance` moves and reports
where it landed, and `PULSEG_CURSOR_DONE` ends the scan.

A segment boundary is where the next prepared segment is armed.

The C counterpart is {doc}`../c/stage5_playout`.

```{literalinclude} ../../../examples/cpp/stage5_playout.cpp
:language: cpp
:caption: examples/cpp/stage5_playout.cpp
```
