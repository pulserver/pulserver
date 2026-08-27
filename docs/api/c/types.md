# Types, limits and errors

What every other page takes as an argument: the scanner's limits, the
diagnostic a failure comes back in, and the two small types that carry a
variable-length argument as one thing.

```c
#include "pulseg/pulseg.h"

pulseg_opts opts;
pulseg_opts_init(
    &opts, 42.576e6f, 3.0f, 42.576e6f * 0.040f, 42.576e6f * 150.0f,
    1.0f, 4.0f, 2.0f, 10.0f);
```

## System limits

`pulseg_opts` is what everything is checked against and what the rasters come
from — and those rasters are the scanner's, not the ones a `.seq` declares:
the file's own are validated against these for playability. `pulseg_opts_init`
takes the limits and the rasters; `pulseg_opts_init_full` additionally takes
the acoustic peak-detection parameters instead of their defaults.

````{only} doxygen
```{doxygenstruct} pulseg_opts
:project: pulserver_c
:members:
```

```{doxygenfunction} pulseg_opts_init
:project: pulserver_c
```

```{doxygenfunction} pulseg_opts_init_full
:project: pulserver_c
```

```{doxygenfunction} pulseg_opts_get_design_raster
:project: pulserver_c
```
````

## Variable-length arguments

A count and the array it measures are one logical argument, so they travel as
one type. That keeps a parameter list from interleaving an output with the
length of an input, and makes a call site impossible to get half right.

Both are borrowed views: neither owns what it points at, and the storage must
outlive the call.

````{only} doxygen
```{doxygenstruct} pulseg_forbidden_band_list
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_forbidden_band
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_text_buffer
:project: pulserver_c
:members:
```
````

Where the knobs of a call outnumber its subject, they are gathered the same
way — see `pulseg_mech_resonances_request` on {doc}`checks`.

## Diagnostics

Every entry point that can fail takes a `pulseg_diagnostic`, and fills it with
what went wrong and where — the block, the axis, the number that was out of
range. `pulseg_format_error` turns the pair of code and diagnostic into the
line a user should see.

````{only} doxygen
```{doxygenstruct} pulseg_diagnostic
:project: pulserver_c
:members:
```

```{doxygenfunction} pulseg_diagnostic_init
:project: pulserver_c
```

```{doxygenfunction} pulseg_format_error
:project: pulserver_c
```
````

## Errors

Every code has a message and, where a caller can act on it, a hint — the two
that a `pulseg_diagnostic` is formatted around.

Branch on `PULSEG_FAILED()` and surface the message, rather than matching a
value: the numbers are grouped by what failed and the group is the stable
part, not the offset within it.

| Range | What failed |
|---|---|
| -1 to -9 | an argument, or an allocation |
| -10 to -19 | reading or parsing the file |
| -50 to -59 | building the unique-block library |
| -100 to -199 | detecting the TR |
| -200 to -299 | detecting the segments |
| -250 to -259 | planning the chunks |
| -400 to -449 | acoustic resonance |
| -450 to -499 | PNS |
| -500 to -559 | the collection, and the gradient limits |
| -560 to -569 | consistency between the structure and the cache |

````{only} doxygen
```{doxygenfunction} pulseg_get_error_message
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_error_hint
:project: pulserver_c
```

```{doxygenfile} pulseg_errors.h
:project: pulserver_c
:sections: define
```
````

## See also

{doc}`file` reads a sequence against these limits, {doc}`checks` judges one
against them.
