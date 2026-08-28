# Stage 2 — prescription check

Runs on every parameter change, so it never builds a sequence.

Before a `.seq` exists, `VALIDATE` asks the design host whether the protocol is
playable and how long it would take. Once a file is on disk,
`pulseg_peek_scan_time` reads the `[DEFINITIONS]` section and stops.

The peek is an approximation: dead time between segments is not accounted for.

`pulseg_text_buffer` carries the message storage and its capacity as one
argument.

The C++ counterpart is {doc}`../cpp/stage2_prescription`.

```{literalinclude} ../../../examples/c/stage2_prescription.c
:language: c
:caption: examples/c/stage2_prescription.c
```
