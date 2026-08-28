# Structure, events and waveform generation

The structure and event getters are members of `Collection` and return values.

`ChunkPlan` owns a `pulseg_chunk_plan`. `mode()` is `PULSEG_WAVE_RESIDENT` or
`PULSEG_WAVE_STREAMED`; `materialise()` renders one axis of one wave and
returns the points and the peak.

The collection a `ChunkPlan` is built from must carry the execution stream.

The C counterpart is {doc}`../c/generation`.

## Reference

````{only} doxygen
```{doxygenfunction} pulseg::Collection::collection_info
```

```{doxygenfunction} pulseg::Collection::subseq_info
```

```{doxygenfunction} pulseg::Collection::segment_info
```

```{doxygenfunction} pulseg::Collection::block_info
```

```{doxygenfunction} pulseg::Collection::segments
```

```{doxygenfunction} pulseg::Collection::segment_block_def_indices
```

```{doxygenfunction} pulseg::Collection::num_unique_blocks
```

```{doxygenfunction} pulseg::Collection::unique_block_id
```

```{doxygenfunction} pulseg::Collection::adc_def
```

```{doxygenfunction} pulseg::Collection::rf_shim_def
```

```{doxygenfunction} pulseg::Collection::num_rf_shims
```

```{doxygenfunction} pulseg::Collection::get_rf_stats
```

```{doxygenfunction} pulseg::Collection::get_rf_definition
```

```{doxygenfunction} pulseg::Collection::tr_rf_ids
```

```{doxygenfunction} pulseg::Collection::grad_initial_amplitude
```

```{doxygenfunction} pulseg::Collection::grad_initial_shape_id
```

```{doxygenfunction} pulseg::Collection::get_label_limits
```

```{doxygenfunction} pulseg::Collection::get_adc_label
```

```{doxygenfunction} pulseg::Collection::get_tr_gradient_waveforms
```

```{doxygenfunction} pulseg::Collection::get_tr_waveforms
```

```{doxygenclass} pulseg::ChunkPlan
:members:
```

```{doxygenstruct} pulseg::MaterialisedWave
:members:
```

```{doxygenstruct} pulseg::TrGradientWaveforms
:members:
```

```{doxygenstruct} pulseg::TrWaveforms
:members:
```

```{doxygenstruct} pulseg::RfStats
:members:
```

```{doxygenstruct} pulseg::SegmentLayout
:members:
```

```{doxygenstruct} pulseg::SequenceDescription
:members:
```

```{doxygenstruct} pulseg::RfDefinitionShapes
:members:
```
````
