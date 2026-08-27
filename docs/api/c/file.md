# Reading a sequence

Three questions, in increasing order of what they cost: what does the file say
about itself, what is in it, and what does it mean.

```c
pulseg_collection *coll = NULL;
pulseg_diagnostic diag;
pulseg_diagnostic_init(&diag);

int code = pulseg_read(&coll, &diag, "scan.seq", &opts, 1, 1, 1);
if (PULSEG_FAILED(code)) {
    char message[256];
    pulseg_format_error(message, sizeof message, code, &diag);
    return;
}
```

A **collection** is one parsed scan: every subsequence in the chain, its
segments, the unique blocks they are built from, the event libraries behind
those, and the cursor that walks the whole thing in playout order. The
structure is *detected* from block content — a `TRID` label is never asked
for and never trusted — so what comes back is what the file plays.

## Peeking

Before committing to a parse, a console needs two answers off the head of the
chain: how long the scan will take, and what the sequence declares about
itself. Both read the `[DEFINITIONS]` section only.

`pulseg_peek_scan_time` is the file-level counterpart of the design host's
`VALIDATE` (see {doc}`protocol`): the same question asked of a `.seq` that
already exists rather than of a protocol that has not been built yet. Its
answer is an approximation — dead time between segments is not accounted for.

````{only} doxygen
```{doxygenfunction} pulseg_peek_scan_time
:project: pulserver_c
```

```{doxygenfunction} pulseg_peek_sequence_flags
:project: pulserver_c
```

```{doxygenstruct} pulseg_scan_time_info
:project: pulserver_c
:members:
```
````

## Reading

`pulseg_read` takes the head of a chain, follows it, and converts the result in
one call. `pulseg_read_from_buffers` takes the files already in memory, which
is what a design service that never touched a disk has.

````{only} doxygen
```{doxygenfunction} pulseg_read
:project: pulserver_c
```

```{doxygenfunction} pulseg_read_from_buffers
:project: pulserver_c
```

```{doxygenfunction} pulseg_collection_alloc
:project: pulserver_c
```

```{doxygenfunction} pulseg_collection_free
:project: pulserver_c
```
````

## Converting

The two halves of `pulseg_read` are separable, and a caller that already has
parsed `pulseq_file` structures — from {doc}`pulseq` directly, or from its own
reader for another sequence-design language — composes them itself.

`pulseg_convert_collection` is the seam: deduplication of the unique blocks,
TR and segment detection, execution-stream expansion, the label table, and the
cross-subsequence consistency checks. It is where a `.seq` stops being a file
and becomes a structure.

````{only} doxygen
```{doxygenfunction} pulseg_convert_collection
:project: pulserver_c
```

```{doxygenfunction} pulseg_check_consistency
:project: pulserver_c
```
````

## What the scan says about itself

The description a reconstruction wants when k-space alone is not enough: the
event table, the RF definitions behind it, and the parameters the design side
declared.

````{only} doxygen
```{doxygenfunction} pulseg_get_sequence_description
:project: pulserver_c
```

```{doxygenfunction} pulseg_sequence_description_free
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_sequence_parameters
:project: pulserver_c
```

```{doxygenstruct} pulseg_sequence_description
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_sequence_parameters
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_seq_event
:project: pulserver_c
:members:
```
````

## See also

{doc}`pulseq` is the reader underneath this one, usable on its own;
{doc}`checks` is what a collection passes before it is played; {doc}`cache`
stores the result so the later stages do not repeat the work.
