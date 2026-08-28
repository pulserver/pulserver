# The checks on their own

An interpreter that already gates amplitude and slew, adding the acoustic and
nerve-stimulation checks and nothing else. No cache, no structure on disk, no
change to how the sequence is played.

`Collection` releases the C handle in its destructor, on the throwing path as
well as the normal one. A violation arrives as a thrown `pulseg::Error`
carrying the code and the diagnostic.

The plan argument is omitted, which is right for a caller asking one question:
the checks keep their preprocessing private to the call.

The C counterpart is {doc}`../c/safety_gate`.

```{literalinclude} ../../../examples/cpp/safety_gate.cpp
:language: cpp
:caption: examples/cpp/safety_gate.cpp
```
