# Stage 5 — playout

The scan loop.

A block definition is shared by every occurrence of it; a block instance is one
occurrence, with the amplitudes it plays at, its rotation, its RF shim and
whether it acquires.

The cursor rests before the first block: `pulseg_cursor_advance` moves and
reports where it landed, and `PULSEG_CURSOR_DONE` ends the scan.

A segment is the unit the hardware is pointed at, so a segment boundary is
where the next prepared segment is armed.

`pulseg_cursor_mark` and `pulseg_cursor_reset` are the pair a prescan uses.

The C++ counterpart is {doc}`../cpp/stage5_playout`.

```{literalinclude} ../../../examples/c/stage5_playout.c
:language: c
:caption: examples/c/stage5_playout.c
```
