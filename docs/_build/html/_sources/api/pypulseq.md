# `pulserver.pypulseq`

This is the drop-in replacement for `pypulseq`. It re-exports upstream
PyPulseq while replacing `Sequence` and selected helpers with Pulserver-aware
implementations. The sections below are organised by use, not private package
layout; Pulserver extensions and their upstream counterparts appear together.

## Types

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.Sequence
   pulserver.pypulseq.Opts
   pulserver.pypulseq.SAR
   pulserver.pypulseq.SigpyPulseOpts
   pulserver.pypulseq.SamplingPattern
   pulserver.pypulseq.SliceGroup
```

## RF pulses

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_adiabatic_pulse
   pulserver.pypulseq.make_arbitrary_rf
   pulserver.pypulseq.make_block_pulse
   pulserver.pypulseq.make_gauss_pulse
   pulserver.pypulseq.make_hard_pulse
   pulserver.pypulseq.make_sinc_pulse
   pulserver.pypulseq.make_sigpy_pulse
   pulserver.pypulseq.make_slr_pulse
   pulserver.pypulseq.make_frequency_selective_pulse
   pulserver.pypulseq.make_slice_selective_pulse
   pulserver.pypulseq.make_spsp_pulse
   pulserver.pypulseq.make_spatially_selective_pulse
   pulserver.pypulseq.make_spiral_selective_pulse
   pulserver.pypulseq.make_2d_selective_pulse
   pulserver.pypulseq.make_3d_selective_pulse
   pulserver.pypulseq.make_inversion_pulse
   pulserver.pypulseq.make_refocusing_pulse
   pulserver.pypulseq.make_mt_pulse
   pulserver.pypulseq.make_ihmt_pulse
   pulserver.pypulseq.make_bloch_siegert_pulse
   pulserver.pypulseq.make_t2prep_pulse
   pulserver.pypulseq.make_t1t2_prep_pulse
   pulserver.pypulseq.make_diffusion_prep
   pulserver.pypulseq.make_fat_saturation_pulse
   pulserver.pypulseq.make_rf_shim
   pulserver.pypulseq.calc_SAR
```

## Encoding

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_line_readout
   pulserver.pypulseq.make_epi_readout
   pulserver.pypulseq.make_fse_readout
   pulserver.pypulseq.make_radial_readout
   pulserver.pypulseq.make_spiral_readout
   pulserver.pypulseq.make_rosette_readout
   pulserver.pypulseq.make_noncartesian_3d_readout
   pulserver.pypulseq.make_zte_readout
   pulserver.pypulseq.chunk_indices
   pulserver.pypulseq.linear_order
   pulserver.pypulseq.outer_inner_order
   pulserver.pypulseq.outer_product
   pulserver.pypulseq.sampled_lines
   pulserver.pypulseq.fse_linear_order
   pulserver.pypulseq.fse_radial_order
   pulserver.pypulseq.fse_radial_adaptive_order
   pulserver.pypulseq.fse_shuffling_order
   pulserver.pypulseq.random_mask
   pulserver.pypulseq.caipirinha_mask
   pulserver.pypulseq.poisson_disc_mask
   pulserver.pypulseq.from_mask
   pulserver.pypulseq.from_relative_shifts
   pulserver.pypulseq.skipped_caipi
   pulserver.pypulseq.radial_2d
   pulserver.pypulseq.golden_angles
   pulserver.pypulseq.uniform_angles
   pulserver.pypulseq.golden_means_3d
   pulserver.pypulseq.spiral_phyllotaxis
   pulserver.pypulseq.directions_to_rotations
   pulserver.pypulseq.slice_groups
```

## Gradients

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_arbitrary_grad
   pulserver.pypulseq.make_trapezoid
   pulserver.pypulseq.make_extended_trapezoid
   pulserver.pypulseq.make_extended_trapezoid_area
   pulserver.pypulseq.make_crusher
   pulserver.pypulseq.make_spoiler
   pulserver.pypulseq.make_phase_encoding
   pulserver.pypulseq.make_phase_blip
   pulserver.pypulseq.add_gradients
   pulserver.pypulseq.scale_grad
   pulserver.pypulseq.split_gradient
   pulserver.pypulseq.split_gradient_at
   pulserver.pypulseq.points_to_waveform
   pulserver.pypulseq.traj_to_grad
   pulserver.pypulseq.traj2grad
```

## Timing

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.align
   pulserver.pypulseq.calc_adc_segments
   pulserver.pypulseq.calc_adc_timing
   pulserver.pypulseq.calc_duration
   pulserver.pypulseq.calc_ramp
   pulserver.pypulseq.calc_rf_bandwidth
   pulserver.pypulseq.calc_rf_center
   pulserver.pypulseq.check_timing
   pulserver.pypulseq.make_adc
   pulserver.pypulseq.make_delay
   pulserver.pypulseq.make_soft_delay
   pulserver.pypulseq.make_trigger
   pulserver.pypulseq.make_digital_output_pulse
```

## Miscellaneous

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_label
   pulserver.pypulseq.make_rotation
   pulserver.pypulseq.get_supported_labels
   pulserver.pypulseq.rotate
   pulserver.pypulseq.convert
   pulserver.pypulseq.compress_shape
   pulserver.pypulseq.decompress_shape
   pulserver.pypulseq.round_half_up
   pulserver.pypulseq.enable_trace
   pulserver.pypulseq.disable_trace
   pulserver.pypulseq.make_rf_spoiling_schedule
   pulserver.pypulseq.make_phase_cycling_schedule
   pulserver.pypulseq.make_traps_schedule
```
