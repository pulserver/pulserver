# Stage 1 — setup

Once per session, before any sequence exists.

`pulseg_opts` carries the limits and the four rasters, and every check and
raster comparison downstream is against it. The gradient and slew ceilings are
derated by sqrt(3) so no physical axis exceeds the amplifier under an
arbitrary rotation.

The bridge spawns `pypulseq_host` and keeps it alive for the session. Its
entry points return -1 with `errno` set, not `PULSEG_ERR_*` codes.

`LIST_PROTOCOL` returns the parameters and their defaults. Each carries a wire
name and a declared type, which is enough to build a console control.

The C++ counterpart is {doc}`../cpp/stage1_setup`.

```{literalinclude} ../../../examples/c/stage1_setup.c
:language: c
:caption: examples/c/stage1_setup.c
```
