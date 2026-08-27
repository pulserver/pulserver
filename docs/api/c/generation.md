# Structure, events and waveform generation

What the scan is made of, the events behind it, and the samples an amplifier
or a transmit chain is actually handed. This is the surface a
waveform-generation pass works through, and the same getters answer the
questions a preparation pass has — echo-filter inputs, RF statistics, the
corner points a gradient-heating model integrates.

## Structure

From the whole scan down to one block. A subsequence is one file of the chain;
a segment is the interpreter's unit of playout; a block is one row of the
block table, and the same block definition is shared by every instance that
plays it.

````{only} doxygen
```{doxygenfunction} pulseg_get_collection_info
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_subseq_info
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_segment_info
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_block_info
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_num_unique_blocks
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_num_unique_segments
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_unique_block_id
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_segment_block_def_indices
:project: pulserver_c
```

```{doxygenfunction} pulseg_segment_has_grad
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_canonical_segment_sequence
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_tr_groups
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_scan_time
:project: pulserver_c
```
````

A segment's layout is the positions it occupies and the blocks that sit in
them, which is what an interpreter that loads a segment at a time asks for.

````{only} doxygen
```{doxygenfunction} pulseg_get_subseq_segment_layout
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_subseq_segment_block_indices
:project: pulserver_c
```

```{doxygenstruct} pulseg_segment_layout
:project: pulserver_c
:members:
```
````

````{only} doxygen
```{doxygenstruct} pulseg_collection_info
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_subseq_info
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_segment_info
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_block_info
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_tr_group
:project: pulserver_c
:members:
```
````

## Events

The RF pulses, gradients, ADC windows and labels a block carries. The shape
arrays come back normalised to a peak of about one, with the physical scale
reported separately — which is what lets one waveform serve every instance
that plays it at a different amplitude.

`pulseg_get_rf_stats` is also the RF summary a vendor applies its own SAR
limits to; the checks on {doc}`checks` deliberately do not.

````{only} doxygen
```{doxygenfunction} pulseg_get_rf_stats
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_array
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_event_array
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_magnitude
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_phase
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_time_us
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_initial_amplitude_hz
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_max_amplitude_hz
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_isocenter_us
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_def_magnitude
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_def_phase
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_def_time
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_tr_rf_ids
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_num_rf_shims
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_rf_shim_def
:project: pulserver_c
```
````

````{only} doxygen
```{doxygenfunction} pulseg_get_grad_amplitude
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_grad_time_us
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_grad_initial_amplitude_hz_per_m
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_grad_initial_shape_id
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_grad_max_amplitude_hz_per_m
:project: pulserver_c
```
````

````{only} doxygen
```{doxygenfunction} pulseg_get_adc_def
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_adc_label
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_label_limits
:project: pulserver_c
```

```{doxygenstruct} pulseg_adc_def
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_rf_stats
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_label_limits
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_rf_event
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_rf_view
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_rf_shim_def
:project: pulserver_c
:members:
```
````

## Waveforms

One canonical TR, resolved into the samples the amplifiers see. This is what
the checks run against and what a plot draws; `pulseg_get_tr_waveforms` adds
the RF, the ADC events and the block boundaries to the gradients.

The corner points are the same TR as a piecewise-linear vertex list rather
than a sampled waveform, which is the form a gradient-heating or slew integral
wants.

````{only} doxygen
```{doxygenfunction} pulseg_get_tr_gradient_waveforms
:project: pulserver_c
```

```{doxygenfunction} pulseg_tr_gradient_waveforms_free
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_tr_waveforms
:project: pulserver_c
```

```{doxygenfunction} pulseg_tr_waveforms_free
:project: pulserver_c
```

```{doxygenfunction} pulseg_get_tr_corner_points
:project: pulserver_c
```

```{doxygenfunction} pulseg_corner_point_stream_free
:project: pulserver_c
```

```{doxygenstruct} pulseg_tr_gradient_waveforms
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_tr_waveforms
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_grad_axis_waveform
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_channel_waveform
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_adc_event
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_tr_block_descriptor
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_corner_point_stream
:project: pulserver_c
:members:
```
````

## Materialising and chunking

`pulseg_materialize_wave` renders one distinct waveform on demand — the call a
generation pass makes once per entry in the wave table rather than once per
block.

Most scans do not need a chunk plan: a handful of base waveforms replayed at
different amplitudes and rotations fits in waveform memory at once, and the
planner says so. Streaming is for what does not fit — individually optimised
trajectories, where the library is measured in gigabytes — and then the unit
is a segment, because a segment is what the interpreter points the hardware
at.

````{only} doxygen
```{doxygenfunction} pulseg_materialize_wave
:project: pulserver_c
```

```{doxygenfunction} pulseg_plan_chunks
:project: pulserver_c
```

```{doxygenfunction} pulseg_free_chunk_plan
:project: pulserver_c
```

```{doxygenstruct} pulseg_chunk_plan
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_chunk
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_chunk_budget
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_wave_key
:project: pulserver_c
:members:
```

```{doxygenenum} pulseg_wave_mode
:project: pulserver_c
```
````

## See also

{doc}`cache` is where a generation pass gets its collection from without
re-parsing; {doc}`playout` walks the same structure in the order it plays.
