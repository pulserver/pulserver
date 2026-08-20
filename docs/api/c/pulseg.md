# pulseg

Reading a `.seq` chain into a collection, and asking that collection what it
holds. This is the library an interpreter integrates.

```c
#include "pulseg/pulseg.h"

pulseg_opts opts;
pulseg_opts_init(
    &opts, 42.576e6f, 3.0f, 42.576e6f * 0.040f, 42.576e6f * 150.0f,
    1.0f, 4.0f, 2.0f, 10.0f);

pulseg_collection *coll = NULL;
pulseg_diagnostic diag;
pulseg_diagnostic_init(&diag);

int code = pulseg_read(&coll, &diag, "scan.seq", &opts, 1, 1, 1, 1);
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

````{only} not doxygen
```{note}
The reference below is generated from the headers by Doxygen, which is not
installed in this build. Everything else on this page is unaffected;
`apt install doxygen` (or the equivalent) and rebuild to see it.
```
````

## System limits

`pulseg_opts` is what everything is checked against and what the rasters come
from — and those rasters are the scanner's, not the ones a `.seq` declares:
the file's own are validated against these for playability. `pulseg_opts_init`
takes the limits and the rasters; `pulseg_opts_init_full` additionally takes
the acoustic peak-detection parameters instead of their defaults.

````{only} doxygen
```{doxygenstruct} pulseg_opts
:members:
```

```{doxygenfunction} pulseg_opts_init
```

```{doxygenfunction} pulseg_opts_init_full
```

```{doxygenfunction} pulseg_opts_get_design_raster
```
````

## Reading

`pulseg_read` takes the head of a chain and follows it; `pulseg_read_from_buffers`
takes the files already in memory, which is what a design service that never
touched a disk has. `pulseg_peek_scan_time` answers the one question a UI asks
before committing to a parse.

````{only} doxygen
```{doxygenfunction} pulseg_read
```

```{doxygenfunction} pulseg_read_from_buffers
```

```{doxygenfunction} pulseg_peek_scan_time
```

```{doxygenfunction} pulseg_collection_alloc
```

```{doxygenfunction} pulseg_collection_free
```

```{doxygenfunction} pulseg_check_consistency
```
````

### Diagnostics

Every entry point that can fail takes a `pulseg_diagnostic`, and fills it with
what went wrong and where — the block, the axis, the number that was out of
range. `pulseg_format_error` turns the pair of code and diagnostic into the
line a user should see.

````{only} doxygen
```{doxygenstruct} pulseg_diagnostic
:members:
```

```{doxygenfunction} pulseg_diagnostic_init
```

```{doxygenfunction} pulseg_format_error
```
````

## Structure

What the scan is made of, from the whole down to one block. A subsequence is
one file of the chain; a segment is the interpreter's unit of playout; a block
is one row of the block table, and the same block definition is shared by
every instance that plays it.

````{only} doxygen
```{doxygenfunction} pulseg_get_collection_info
```

```{doxygenfunction} pulseg_get_subseq_info
```

```{doxygenfunction} pulseg_get_segment_info
```

```{doxygenfunction} pulseg_get_block_info
```

```{doxygenfunction} pulseg_get_num_unique_blocks
```

```{doxygenfunction} pulseg_get_num_unique_segments
```

```{doxygenfunction} pulseg_get_unique_block_id
```

```{doxygenfunction} pulseg_get_segment_block_def_indices
```

```{doxygenfunction} pulseg_segment_has_grad
```

```{doxygenfunction} pulseg_get_canonical_segment_sequence
```

```{doxygenfunction} pulseg_get_tr_groups
```

```{doxygenfunction} pulseg_get_scan_time
```
````

A segment's layout is the positions it occupies and the blocks that sit in
them, which is what an interpreter that loads a segment at a time asks for.

````{only} doxygen
```{doxygenfunction} pulseg_get_subseq_segment_layout
```

```{doxygenfunction} pulseg_get_subseq_segment_block_indices
```

```{doxygenstruct} pulseg_segment_layout
:members:
```
````

````{only} doxygen
```{doxygenstruct} pulseg_collection_info
:members:
```

```{doxygenstruct} pulseg_subseq_info
:members:
```

```{doxygenstruct} pulseg_segment_info
:members:
```

```{doxygenstruct} pulseg_block_info
:members:
```

```{doxygenstruct} pulseg_scan_time_info
:members:
```

```{doxygenstruct} pulseg_tr_group
:members:
```
````

## The cursor

Walking the scan in the order it plays. The cursor is what turns the
structural description back into a stream of block instances — the same walk
the interpreter makes, so what it reports is what will be played, signed
amplitudes and rotations included.

````{only} doxygen
```{doxygenfunction} pulseg_cursor_next
```

```{doxygenfunction} pulseg_cursor_advance
```

```{doxygenfunction} pulseg_cursor_rewind
```

```{doxygenfunction} pulseg_cursor_mark
```

```{doxygenfunction} pulseg_cursor_reset
```

```{doxygenfunction} pulseg_cursor_get_info
```

```{doxygenfunction} pulseg_get_block_instance
```

```{doxygenfunction} pulseg_get_block_instance_at
```

```{doxygenstruct} pulseg_cursor_info
:members:
```

```{doxygenstruct} pulseg_block_instance
:members:
```
````

## Events

The RF pulses, gradients, ADC windows and labels a block carries. The shape
arrays come back normalised to a peak of about one, with the physical scale
reported separately — which is what lets one waveform serve every instance
that plays it at a different amplitude.

````{only} doxygen
```{doxygenfunction} pulseg_get_rf_stats
```

```{doxygenfunction} pulseg_get_rf_array
```

```{doxygenfunction} pulseg_get_rf_event_array
```

```{doxygenfunction} pulseg_get_rf_magnitude
```

```{doxygenfunction} pulseg_get_rf_phase
```

```{doxygenfunction} pulseg_get_rf_time_us
```

```{doxygenfunction} pulseg_get_rf_initial_amplitude_hz
```

```{doxygenfunction} pulseg_get_rf_max_amplitude_hz
```

```{doxygenfunction} pulseg_get_rf_isocenter_us
```

```{doxygenfunction} pulseg_get_rf_def_magnitude
```

```{doxygenfunction} pulseg_get_rf_def_phase
```

```{doxygenfunction} pulseg_get_rf_def_time
```

```{doxygenfunction} pulseg_get_tr_rf_ids
```

```{doxygenfunction} pulseg_get_num_rf_shims
```

```{doxygenfunction} pulseg_get_rf_shim_def
```
````

````{only} doxygen
```{doxygenfunction} pulseg_get_grad_amplitude
```

```{doxygenfunction} pulseg_get_grad_time_us
```

```{doxygenfunction} pulseg_get_grad_initial_amplitude_hz_per_m
```

```{doxygenfunction} pulseg_get_grad_initial_shape_id
```

```{doxygenfunction} pulseg_get_grad_max_amplitude_hz_per_m
```
````

````{only} doxygen
```{doxygenfunction} pulseg_get_adc_def
```

```{doxygenfunction} pulseg_get_adc_label
```

```{doxygenfunction} pulseg_get_label_limits
```

```{doxygenstruct} pulseg_adc_def
:members:
```

```{doxygenstruct} pulseg_rf_stats
:members:
```

```{doxygenstruct} pulseg_label_limits
:members:
```

```{doxygenstruct} pulseg_rf_event
:members:
```

```{doxygenstruct} pulseg_rf_view
:members:
```

```{doxygenstruct} pulseg_rf_shim_def
:members:
```
````

## Waveforms

One canonical TR, resolved into the samples the amplifiers see. This is what
the safety checks run against and what a plot draws; `pulseg_get_tr_waveforms`
adds the RF, the ADC events and the block boundaries to the gradients.

````{only} doxygen
```{doxygenfunction} pulseg_get_tr_gradient_waveforms
```

```{doxygenfunction} pulseg_tr_gradient_waveforms_free
```

```{doxygenfunction} pulseg_get_tr_waveforms
```

```{doxygenfunction} pulseg_tr_waveforms_free
```

```{doxygenstruct} pulseg_tr_gradient_waveforms
:members:
```

```{doxygenstruct} pulseg_tr_waveforms
:members:
```

```{doxygenstruct} pulseg_grad_axis_waveform
:members:
```

```{doxygenstruct} pulseg_channel_waveform
:members:
```

```{doxygenstruct} pulseg_adc_event
:members:
```

```{doxygenstruct} pulseg_tr_block_descriptor
:members:
```
````

## Chunking

Most scans do not need this: a handful of base waveforms replayed at different
amplitudes and rotations fits in waveform memory at once, and the planner says
so. Streaming is for what does not fit — individually optimised trajectories,
where the library is measured in gigabytes — and then the unit is a segment,
because a segment is what the interpreter points the hardware at.

````{only} doxygen
```{doxygenfunction} pulseg_plan_chunks
```

```{doxygenfunction} pulseg_free_chunk_plan
```

```{doxygenfunction} pulseg_materialize_wave
```

```{doxygenstruct} pulseg_chunk_plan
:members:
```

```{doxygenstruct} pulseg_chunk
:members:
```

```{doxygenstruct} pulseg_chunk_budget
:members:
```

```{doxygenstruct} pulseg_wave_key
:members:
```

```{doxygenenum} pulseg_wave_mode
```
````

## Cache

A binary sidecar holding what is expensive to recompute — the deduplication
tables, the segmentation, the execution stream. The per-section loaders exist
so a consumer pays only for what it reads: the pulse-generation pass takes
`COMMON` and `SHAPES`, the scan loop additionally takes `INSTANCES` and
`SCANLOOP`. The reconstruction side reads the `.seq` directly and never
touches this.

````{only} doxygen
```{doxygenfunction} pulseg_save_cache
```

```{doxygenfunction} pulseg_load_cache
```

```{doxygenfunction} pulseg_load_geninstructions_cache
```

```{doxygenfunction} pulseg_load_scanloop_cache
```

```{doxygenfunction} pulseg_clear_cache
```

```{doxygenfunction} pulseg_write_vendor_cache_section
```

```{doxygenfunction} pulseg_read_vendor_cache_section
```
````

## Sequence description

What the scan says about itself, for a reconstruction that wants more than
k-space: the event table, the RF definitions behind it, and the parameters the
design side declared.

````{only} doxygen
```{doxygenfunction} pulseg_convert_collection
```

```{doxygenfunction} pulseg_get_sequence_description
```

```{doxygenfunction} pulseg_sequence_description_free
```

```{doxygenfunction} pulseg_get_sequence_parameters
```

```{doxygenstruct} pulseg_sequence_description
:members:
```

```{doxygenstruct} pulseg_sequence_parameters
:members:
```

```{doxygenstruct} pulseg_seq_event
:members:
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
```

```{doxygenfunction} pulseg_get_error_hint
```

```{doxygenfile} pulseg_errors.h
:sections: define
```
````

## See also

{doc}`safety` is the gate a collection passes before it is played,
{doc}`protocol` the parameters a console shows, and {doc}`../cpp/pulseg` the
same library with RAII types over it.
