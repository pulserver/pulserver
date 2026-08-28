# Playout

Walking the scan in the order it plays. The cursor turns the structural
description back into a stream of block instances, signed amplitudes and
rotations included.

## The cursor

A cursor rests at position -1, so a fresh one must be advanced before it is
read. `pulseg_cursor_next` advances one block; `pulseg_cursor_advance`
advances and reports where it landed.

`pulseg_cursor_mark` and `pulseg_cursor_reset` are the pair a prescan uses.

````{only} doxygen
```{doxygenfunction} pulseg_cursor_next
:project: pulserver_c
```

```{doxygenfunction} pulseg_cursor_advance
:project: pulserver_c
```

```{doxygenfunction} pulseg_cursor_rewind
:project: pulserver_c
```

```{doxygenfunction} pulseg_cursor_mark
:project: pulserver_c
```

```{doxygenfunction} pulseg_cursor_reset
:project: pulserver_c
```

```{doxygenfunction} pulseg_cursor_get_info
:project: pulserver_c
```

```{doxygenstruct} pulseg_cursor_info
:project: pulserver_c
:members:
```
````

## One block instance

A block definition is shared by every occurrence of it. A block instance is
one occurrence: its amplitudes, its rotation, its RF shim, and the labels its
readout carries. `pulseg_get_block_instance` reads the one the cursor rests
on; `pulseg_get_block_instance_at` reads any position directly.

````{only} doxygen
```{doxygenfunction} pulseg_get_block_instance
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_block_instance_at
:project: pulserver_c
```

```{doxygenstruct} pulseg_block_instance
:project: pulserver_c
:members:
```
````

## See also

{doc}`cache` supplies the collection; `SCANLOOP` is the section holding the
execution stream. {doc}`generation` asks about the same structure without a
position in it. {doc}`../cpp/playout` is the C++ counterpart.
