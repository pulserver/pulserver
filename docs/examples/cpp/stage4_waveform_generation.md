# Stage 4 — waveform generation

A scan of a million blocks contains a handful of distinct waveforms, replayed
at different amplitudes and rotations.

`ChunkPlan::mode()` is `PULSEG_WAVE_RESIDENT` when they all fit in waveform
memory at once, `PULSEG_WAVE_STREAMED` when each segment has to be
materialised while its predecessor plays.

`materialise()` renders one axis of one wave and returns the points by value.

The collection is loaded with `from_scanloop_cache`, not
`from_geninstructions_cache`: the planner needs the execution stream to know
how often each waveform is played and in what order.

The C counterpart is {doc}`../c/stage4_waveform_generation`.

```{literalinclude} ../../../examples/cpp/stage4_waveform_generation.cpp
:language: cpp
:caption: examples/cpp/stage4_waveform_generation.cpp
```
