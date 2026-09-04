# Reading a sequence

`peek_scan_time` and `peek_sequence_flags` read the `[DEFINITIONS]` section and
stop.

`Collection`'s constructors read and structure in one step: one takes a path
and follows the chain, the other takes buffers already in memory.

`pulseq_read` and `pulseg_convert_collection` are not wrapped separately. A
caller composing them itself, or supplying parsed files from its own reader,
calls the C entry points directly.

The C counterpart is {doc}`../c/file`.

## Reference

````{only} doxygen
```{doxygenfunction} pulseg::peek_scan_time
```

```{doxygenfunction} pulseg::peek_sequence_flags
```

```{doxygenclass} pulseg::Collection
```

```{doxygenfunction} pulseg::Collection::opts
```

```{doxygenfunction} pulseg::Collection::check_consistency
```

```{doxygenfunction} pulseg::Collection::get_scan_time
```

````
