# Playout

Walking the scan in the order it plays. The cursor is what turns the
structural description back into a stream of block instances — the same walk
the interpreter makes, so what it reports is what will be played, signed
amplitudes and rotations included.

This is the surface a scan loop runs on, and the one place where the answer
depends on *where you are* rather than on what the sequence contains.

## The cursor

`pulseg_cursor_next` advances and reports; `pulseg_cursor_advance` moves
without reading, which is what a loop skipping ahead does. A cursor rests at
position -1, so a fresh one must be advanced before it is read.

`pulseg_cursor_mark` and `pulseg_cursor_reset` are the pair a prescan uses: run
a stretch of the stream, then return to where it began and run the real thing
from the same point.

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

A block instance is a block definition plus what this particular occurrence
does with it: its amplitudes, its rotation, its RF shim, the labels its
readout carries. `pulseg_get_block_instance` reads the one the cursor rests
on; `pulseg_get_block_instance_at` reads any position directly, for a loop
that seeks rather than streams.

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

{doc}`cache` supplies the collection a scan loop walks — `SCANLOOP` is the
section that holds the execution stream. {doc}`generation` is the same
structure asked about statically, without a position in it.
