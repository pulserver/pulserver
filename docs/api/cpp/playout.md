# Playout

`cursor_next` advances one block. `cursor_advance` advances and reports where
it landed. `cursor_info` reads the current position without moving.

The cursor rests before the first block: advance, then read.

`cursor_mark` and `cursor_reset` are the pair a prescan uses. `cursor_rewind`
returns to the start.

`get_block_instance` reads the block the cursor rests on;
`get_block_instance_at` reads any position directly.

The C counterpart is {doc}`../c/playout`.

## Reference

````{only} doxygen
```{doxygenfunction} pulseg::Collection::cursor_next
```

```{doxygenfunction} pulseg::Collection::cursor_advance
```

```{doxygenfunction} pulseg::Collection::cursor_info
```

```{doxygenfunction} pulseg::Collection::cursor_rewind
```

```{doxygenfunction} pulseg::Collection::cursor_mark
```

```{doxygenfunction} pulseg::Collection::cursor_reset
```

```{doxygenfunction} pulseg::Collection::get_block_instance
```

```{doxygenfunction} pulseg::Collection::get_block_instance_at
```

```{doxygenstruct} pulseg::BlockInstance
:members:
```

```{doxygenstruct} pulseg::ScanTimeInfo
:members:
```

```{doxygenstruct} pulseg::LabelLimits
:members:
```
````
