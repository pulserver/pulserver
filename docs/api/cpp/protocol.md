# Protocol and bridge

`Protocol` is the parameter set a sequence takes. Parameters are addressed by
id; `Protocol::param_id()` maps a wire name to one, `wire_name()` and
`param_type()` go the other way.

`Bridge` is one live `pypulseq_host` child process. It spawns on construction
and reaps in the destructor. Its three commands are `list_protocol()`,
`validate()` and `generate()`; all three throw `BridgeError` on a process or
pipe failure.

The C counterpart is {doc}`../c/protocol`.

## Reference

````{only} doxygen
```{doxygenclass} pulseg::Protocol
:members:
```

```{doxygenclass} pulseg::Bridge
:members:
```

```{doxygenstruct} pulseg::ValidateResult
:members:
```
````
