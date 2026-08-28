# Types, limits and errors

`Opts` carries the scanner's limits and the four rasters. `to_c()` produces the
`pulseg_opts` every call takes.

`Error` carries the C error code and the diagnostic message. `BridgeError`
carries `errno`, and is thrown only by `Bridge`.

The C counterpart is {doc}`../c/types`.

## Reference

````{only} doxygen
```{doxygenclass} pulseg::Error
:members:
```

```{doxygenclass} pulseg::BridgeError
:members:
```

```{doxygenstruct} pulseg::Opts
:members:
```

```{doxygenstruct} pulseg::ForbiddenBand
:members:
```
````
