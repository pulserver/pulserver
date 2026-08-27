# pulseq (C)

The standalone `.seq` reader. It depends on nothing else in this repository
and can be linked on its own: it owns the raw file model, the parser, the
shape codec and the label tables. `pulseg` links it and never the reverse.

```c
#include "pulseq/pulseq.h"

pulseq_file seq;
pulseq_file_init(&seq, NULL);
if (pulseq_read(&seq, "scan.seq") != PULSEQ_SUCCESS) { /* ... */ }
pulseq_file_free(&seq);
```

The file model is `float` end to end, deliberately: a `.seq` stores what it
stores, and single precision is what a scanner plays. The C++ side
instantiates the same sources a second time at double precision when an
analysis needs it, scoped inside `pulseq::raw64`.

````{only} not doxygen
```{note}
The reference below is generated from the headers by Doxygen, which is not
installed in this build. Everything else on this page is unaffected;
`apt install doxygen` (or the equivalent) and rebuild to see it.
```
````

## Reading

Text and binary are the same model, read by different front ends;
`pulseq_file_is_binary` says which a path is. A chain is read by
`pulseq_file_set_read`, which follows the `NextSequence` definition to the
end. `pulseq_read_definitions_only` is what a consumer that wants the header
and nothing else pays for.

````{only} doxygen
```{doxygenfunction} pulseq_file_init
:project: pulserver_c
```

```{doxygenfunction} pulseq_file_free
:project: pulserver_c
```

```{doxygenfunction} pulseq_read
:project: pulserver_c
```

```{doxygenfunction} pulseq_read_from_buffer
:project: pulserver_c
```

```{doxygenfunction} pulseq_read_definitions_only
:project: pulserver_c
```

```{doxygenfunction} pulseq_file_is_binary
:project: pulserver_c
```

```{doxygenfunction} pulseq_read_binary
:project: pulserver_c
```

```{doxygenfunction} pulseq_read_binary_from_buffer
:project: pulserver_c
```

```{doxygenfunction} pulseq_read_binary_definitions_only
:project: pulserver_c
```

```{doxygenfunction} pulseq_file_set_read
:project: pulserver_c
```

```{doxygenfunction} pulseq_file_set_free
:project: pulserver_c
```

```{doxygenfunction} pulseq_verify_signature
:project: pulserver_c
```
````

## The file model

````{only} doxygen
```{doxygenstruct} pulseq_file
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_file_set
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_raster
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_definition
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_reserved_definitions
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_section_offsets
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_shape
:project: pulserver_c
:members:
```
````

## Blocks and extensions

````{only} doxygen
```{doxygenfunction} pulseq_get_raw_block_content_ids
:project: pulserver_c
```

```{doxygenfunction} pulseq_get_raw_extension
:project: pulserver_c
```

```{doxygenstruct} pulseq_raw_block
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_raw_extension
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_rf_shim_entry
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_trigger_event
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_label_event
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_flag_event
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseq_label_limit
:project: pulserver_c
:members:
```
````

## Shapes

A `.seq` stores a waveform compressed, and the scheme is chosen for what MR
waveforms look like: run-length encoding of the derivative, so a constant run
and a linear ramp both cost a handful of numbers.

````{only} doxygen
```{doxygenfunction} pulseq_decompress_shape
:project: pulserver_c
```
````

## Labels and hints

Label names resolve to ids, and an unknown name registers rather than fails —
a sequence may carry a counter this library has never heard of, and a reader
that refused it would be refusing a valid file.

````{only} doxygen
```{doxygenfunction} pulseq_label_id_for_name
:project: pulserver_c
```

```{doxygenfunction} pulseq_label_register_name
:project: pulserver_c
```

```{doxygenfunction} pulseq_hint_id_for_name
:project: pulserver_c
```

```{doxygenfunction} pulseq_label_name_for_id
:project: pulserver_c
```

```{doxygenfunction} pulseq_hint_name_for_id
:project: pulserver_c
```
````

## RF spectra

A slice-selective pulse played on a gradient excites a slab whose thickness is
its spectral bandwidth over that gradient's amplitude, and nothing in a `.seq`
records the thickness — so anything that wants it takes the transform of the
pulse.

It is a *plan* because the grid depends on the raster and the wanted
resolution alone, never on the pulse: 10 Hz over a 1 µs raster is a hundred
thousand points, and a variable-flip train has hundreds of distinct pulses
sharing one grid. Create the plan once, run every pulse through it.

The recipe is `mr.calcRfBandwidth`'s, defaults included: resample onto a
uniform grid centred on the pulse centre, transform, and take the outermost
frequencies at which the magnitude first exceeds `cutoff` times its peak.

````{only} doxygen
```{doxygenfunction} pulseq_rf_spectrum_create
:project: pulserver_c
```

```{doxygenfunction} pulseq_rf_spectrum_run
:project: pulserver_c
```

```{doxygenfunction} pulseq_rf_spectrum_size
:project: pulserver_c
```

```{doxygenfunction} pulseq_rf_spectrum_free
:project: pulserver_c
```

```{doxygenfunction} pulseq_rf_spectrum_freq
:project: pulserver_c
```

```{doxygenfunction} pulseq_rf_spectrum_re
:project: pulserver_c
```

```{doxygenfunction} pulseq_rf_spectrum_im
:project: pulserver_c
```

```{doxygenfunction} pulseq_rf_bandwidth
:project: pulserver_c
```
````

## Paths

A chain names its successor by file name, and the reader has to resolve it
against the directory the lead file came from.

````{only} doxygen
```{doxygenfunction} pulseq_path_dirname
:project: pulserver_c
```

```{doxygenfunction} pulseq_path_join
:project: pulserver_c
```
````

## See also

{doc}`../cpp/pulseq` is the C++ sequence library over the same format —
writing as well as reading, k-space, moments and label derivation.
