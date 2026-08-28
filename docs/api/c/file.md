# Reading a sequence

Three entry points, in increasing order of cost: what the file declares, what
it contains, and what it means.

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
those, and the cursor that walks them in playout order. The structure is
detected from block content; a `TRID` label is never read.

## Peeking

Both read the `[DEFINITIONS]` section of the head of the chain and stop.

`pulseg_peek_scan_time` is the file-level counterpart of the design host's
`VALIDATE` (see {doc}`protocol`), asked of a `.seq` that already exists. Its
answer is an approximation: dead time between segments is not accounted for.

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

## Parsing, then structuring

`pulseq_read` follows a chain from its head and returns the raw Pulseq model:
blocks, the event libraries they index, shapes, definitions. The full reader —
buffers, the binary format, definitions-only — is on {doc}`pulseq`.

`pulseg_convert_collection` takes those parsed files and produces the
collection: deduplication of the unique blocks, TR and segment detection,
execution-stream expansion, the label table, and the cross-subsequence
consistency checks. A reader for another sequence-design language arrives
here.

````{only} doxygen
```{doxygenfunction} pulseg_convert_collection
:project: pulserver_c
```

```{doxygenfunction} pulseg_check_consistency
:project: pulserver_c
```
````

## Both at once

`pulseg_read` is the two composed, for a caller with a path.
`pulseg_read_from_buffers` is the same for files already in memory.

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

## What the scan says about itself

The event table, the RF definitions behind it, and the parameters the design
side declared.

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

{doc}`pulseq` is the reader underneath. {doc}`checks` is what a collection
passes before it is played. {doc}`cache` stores the result.
{doc}`../cpp/file` is the C++ counterpart.
