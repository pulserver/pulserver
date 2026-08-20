# Protocol and bridge

The other direction: the parameters a console UI shows, and the seam to the
Python host that builds a sequence from them.

````{only} not doxygen
```{note}
The reference below is generated from the headers by Doxygen, which is not
installed in this build. Everything else on this page is unaffected;
`apt install doxygen` (or the equivalent) and rebuild to see it.
```
````

## Protocol

A vendor-neutral parameter table. The IDs mirror the Python `UIParam`
enumeration one for one, so a value set on the console is the value the plugin
reads; no vendor unit, CV name or UI concept appears here. The wire format is
the standard preamble:

```text
[NimPulseqGUI Protocol]
TE: 5.0
TR: 500.0
NSlices: 10
FatSat: 1
[NimPulseqGUI Protocol End]
```

The IDs are `#define` plus `typedef` rather than a C enum, because some EPIC
compilers reject C enums — the same reason the rest of `src/c/` is C89.

````{only} doxygen
```{doxygenfunction} pulseg_protocol_parse
```

```{doxygenfunction} pulseg_protocol_serialize
```

```{doxygenfunction} pulseg_protocol_find
```

```{doxygenfunction} pulseg_param_find
```

```{doxygenfunction} pulseg_param_get_type
```

```{doxygenfunction} pulseg_param_wire_name
```
````

### Reading and writing a value

Typed accessors, one pair per kind. A string list is addressed by index,
because the wire format carries the choice rather than the text.

````{only} doxygen
```{doxygenfunction} pulseg_protocol_get_float
```

```{doxygenfunction} pulseg_protocol_get_int
```

```{doxygenfunction} pulseg_protocol_get_bool
```

```{doxygenfunction} pulseg_protocol_get_stringlist
```

```{doxygenfunction} pulseg_protocol_set_float
```

```{doxygenfunction} pulseg_protocol_set_int
```

```{doxygenfunction} pulseg_protocol_set_bool
```

```{doxygenfunction} pulseg_protocol_set_stringlist
```
````

````{only} doxygen
```{doxygenstruct} pulseg_protocol
:members:
```

```{doxygenstruct} pulseg_protocol_value
:members:
```

```{doxygenstruct} pulseg_param_entry
:members:
```
````

## Bridge

Sequence *generation* is Python; everything downstream of a `.seq` file is
this library. The bridge is the seam: it spawns `pypulseq_host --persistent`,
keeps it alive across a session, and exchanges line-oriented commands over its
pipes.

These entry points return `0` or a non-negative count on success and `-1` with
`errno` set on failure — **not** the `PULSEG_ERR_*` codes the rest of the
library uses. What fails here is a process or a pipe, not a sequence model.

````{only} doxygen
```{doxygenfunction} pulseg_bridge_open_with_opts
```

```{doxygenfunction} pulseg_bridge_close
```

```{doxygenfunction} pulseg_bridge_list_protocol
```

```{doxygenfunction} pulseg_bridge_validate
```

```{doxygenfunction} pulseg_bridge_generate
```

```{doxygenstruct} pulseg_bridge
:members:
```
````

## See also

{doc}`../python/apps` is the Python side of the same contract: the typed
parameters a plugin declares and the `SequencePlugin` the bridge drives.
