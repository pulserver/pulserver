# Stage 2 — prescription check

Runs on every parameter change, so it never builds a sequence.

Before a `.seq` exists, `Bridge::validate` returns a `ValidateResult`:
playable, a duration, and a message. Once a file is on disk,
`peek_scan_time` reads the `[DEFINITIONS]` section and stops.

The peek is an approximation: dead time between segments is not accounted for.

`BridgeError` and `Error` are caught separately: a broken pipe is not a
sequence-model failure.

The C counterpart is {doc}`../c/stage2_prescription`.

```{literalinclude} ../../../examples/cpp/stage2_prescription.cpp
:language: cpp
:caption: examples/cpp/stage2_prescription.cpp
```
