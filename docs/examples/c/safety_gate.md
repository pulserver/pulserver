# The checks on their own

An interpreter that already gates amplitude and slew, adding the acoustic and
nerve-stimulation checks and nothing else. No cache, no structure on disk, no
change to how the sequence is played.

The full gate is available from the same call: amplitude, slew, continuity and
raster alignment are checked too, and each is a separate entry point for a
caller that wants only some of them.

RF is absent. SAR limits are vendor property, so the per-pulse summary is
reported through `pulseg_get_rf_stats` and the limit stays with whoever owns
it.

The `plan` argument is `NULL` here, which is right for a caller asking one
question: the checks keep their preprocessing private to the call.

The C++ counterpart is {doc}`../cpp/safety_gate`.

```{literalinclude} ../../../examples/c/safety_gate.c
:language: c
:caption: examples/c/safety_gate.c
```
