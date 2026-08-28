# Stage 4 — waveform generation

A scan of a million blocks contains a handful of distinct waveforms, replayed
at different amplitudes and rotations.

`pulseg_plan_chunks` returns one of two modes. `PULSEG_WAVE_RESIDENT` means
every distinct waveform fits in waveform memory at once and nothing is
uploaded mid-scan. `PULSEG_WAVE_STREAMED` means each segment is materialised
while its predecessor plays, which the budget decides and which can be
refused as infeasible.

`pulseg_materialize_wave` renders one axis of one wave.

```{note}
Chunk planning reads the scan-stage cache. The planner needs the execution
stream to know how often each waveform is played and in what order. A stage
that only rendered definitions would take
`pulseg_load_geninstructions_cache` instead.
```

The C++ counterpart is {doc}`../cpp/stage4_waveform_generation`.

```{literalinclude} ../../../examples/c/stage4_waveform_generation.c
:language: c
:caption: examples/c/stage4_waveform_generation.c
```
