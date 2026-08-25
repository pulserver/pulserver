# pulserver.pypulseq

The event layer: the `Sequence` object, the factories that build the blocks
it holds, the analysis that reads a built sequence back, and the files either
side.

```python
import pulserver.pypulseq as pp

seq = pp.Sequence(pp.Opts())
seq.add_block(pp.make_delay(1e-3))
```

The complete public upstream PyPulseq namespace is re-exported unchanged, then
Pulserver's replacements are layered on top, so a plugin needs one import for
the whole event layer and `import pypulseq` alongside it is never necessary.
Only the names in `OVERRIDES` differ from upstream; everything else *is*
upstream.

This namespace is the event layer only. The factories that build whole
sequence modules — an excitation with its rephaser, one readout TR — live in
{doc}`design`.

```{eval-rst}
.. currentmodule:: pulserver.pypulseq
```

## The sequence

One object, built block by block and read back through the analysis methods on
it: `calculate_kspace` for where the samples are, `calculate_pns` and
`calculate_gradient_spectrum` for what the hardware will make of them,
`evaluate_labels` for the counters, and `plot`, `plot_kspace` and
`plot_rf` for a picture.
`Opts` is the limits everything is designed against.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/pypulseq
   :template: autosummary/class.rst

   Sequence
   Opts
   TransformFOV
```

## Events

The factories that build what goes in a block. These are upstream's, wrapped
so the event comes back with its fields in slots rather than in a dictionary:
same validation, same defaults.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/pypulseq

   make_adc
   make_delay
   make_soft_delay
   make_trapezoid
   make_extended_trapezoid
   make_arbitrary_grad
   make_block_pulse
   make_sinc_pulse
   make_gauss_pulse
   make_adiabatic_pulse
   make_arbitrary_rf
   make_trigger
   make_digital_output_pulse
```

### Extensions

Rotation and RF-shim extensions, and the labels an interpreter and a
reconstruction read the encoding counters off.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/pypulseq

   make_rotation
   make_rf_shim
   make_label
   canonical_label
   get_supported_labels
   block_to_events
   tile
```

The label vocabulary itself is documented by `get_supported_labels`, whose
tables say what every name means and which of them reach a reconstruction. The
same splits are reachable as plain tuples and dictionaries, for a plugin that
iterates them rather than reads them: `COUNTER_LABELS`, `ENCODING_COUNTERS` and
`FRAME_COUNTERS` for the counters, `FLAG_LABELS`, `SCANNER_FLAGS` and
`STICKY_FLAGS` for the flags, and `MRD_COUNTERS` and `MRD_FLAGS` for the map
onto ISMRMRD's own names.

## Pulses, gradients and timing

Factories that return an event or a plain array rather than a whole module,
which is what puts them here and not in {doc}`design`. `make_slr_pulse` is the
filter-design pulse everything selective is built from; `traj_to_grad` turns a
k-space path into the gradient that walks it.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/pypulseq

   make_slr_pulse
   make_sms_pulse
   make_spsp_pulse
   make_2d_selective_pulse
   make_sigpy_pulse
   make_crusher
   make_phase_encoding
   make_phase_blip
   concatenate_gradients
   traj_to_grad
   calc_adc_timing
```

## Sampling

Undersampling masks, view orderings and projection angles: plain arrays a scan
loop indexes with, so they belong here rather than with the modules.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/pypulseq

   make_uniform_mask
   make_random_mask
   make_poisson_disc_mask
   make_caipirinha_mask
   calc_calibration_lines
   calc_sampled_lines
   calc_sampled_pairs
   make_linear_order
   make_centric_order
   make_radial_order
   make_radial_adaptive_order
   make_shuffling_order
   calc_traversal_order
   calc_epi_order
   calc_uniform_angles
   calc_golden_angles
   calc_tiny_golden_angles
   calc_raga_angles
   calc_projection_shell
```

## System limits and schedules

Derating, raster rounding, and the per-repetition flip and phase lists a scan
loop indexes. All of them answer from an `Opts` and a count rather than from a
sequence.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/pypulseq

   apply_system_derates
   cap_system
   ceil_to_raster
   round_to_raster
   quantize_readout_timing
   make_rf_spoiling_schedule
   make_phase_cycling_schedule
   make_traps_schedule
```

## Simulation

What a pulse does to the magnetisation, before any of it reaches a scanner.
`sim_rf` is MATLAB Pulseq's `simRf`; `bloch` is the integrator underneath it.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/pypulseq

   sim_rf
   bloch
   calc_rf_bandwidth
   calc_rf_power
```

## Results

What the analysis methods return under `compat=False`: the named form of an
answer whose upstream spelling is a tuple. Exported so a caller can annotate
or `isinstance`-check what it was handed.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/pypulseq
   :template: autosummary/class.rst

   KSpace
   Pns
   GradientSpectrum
   Waveforms
   WaveformsAndTimes
   AdcTimes
   RfTimes
   RfPower
   RfResponse
   BTensor
   DiffusionTable
   SoftDelay
```

## Files

Reading and writing `.seq` files, and the vendor band tables the mechanical
resonance check is run against. {meth}`Sequence.write` writes to a path;
`pulserver.pypulseq.write` also returns the payload, which is what a design service
sends over a socket.

```{eval-rst}
.. currentmodule:: pulserver.pypulseq

.. autosummary::
   :toctree: ../generated/pypulseq

   read
   write
   read_asc_bands
   read_esp_bands
   bands_to_hz_per_m
```

## MATLAB parity

```{eval-rst}
.. currentmodule:: pulserver.pypulseq

.. autosummary::
   :toctree: ../generated/pypulseq

   rotate3D
   get_supported_rf_use
   make_hexagon_gradient_area
   restore_additional_shape_samples
   verify_file_signature
```

## Upstream, re-exported

Imported here unchanged from PyPulseq. Their documentation is upstream's.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/pypulseq

   add_gradients
   align
   scale_grad
   rotate
   split_gradient
   split_gradient_at
   make_extended_trapezoid_area
   points_to_waveform
   calc_duration
   calc_ramp
   calc_rf_center
   calc_adc_segments
   calc_SAR
   round_half_up
   enable_trace
   disable_trace
```

## Sets

The membership tests the API page and its contract test are both written
against.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/pypulseq

   OVERRIDES
   UPSTREAM
   RESULTS
   MATLAB_PARITY
   BASE_FACTORIES
   SAMPLING
   SYSTEM
   SLOTTED
```
