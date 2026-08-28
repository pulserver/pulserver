# Stage 1 — setup

Once per session, before any sequence exists.

`Opts` is stated once and passed everywhere. The gradient and slew ceilings
are derated by sqrt(3) so no physical axis exceeds the amplifier under an
arbitrary rotation.

`Bridge` spawns the `pypulseq_host` child and reaps it in the destructor, so
it lives as long as the object.

`Protocol::keys()` gives the parameter ids; `wire_name()` and `param_type()`
give the console what it needs to build each control.

The C counterpart is {doc}`../c/stage1_setup`.

```{literalinclude} ../../../examples/cpp/stage1_setup.cpp
:language: cpp
:caption: examples/cpp/stage1_setup.cpp
```
