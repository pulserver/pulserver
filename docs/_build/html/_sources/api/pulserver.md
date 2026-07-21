# `pulserver`

`pulserver` contains plugin contracts, protocol types, and convenient
re-exports for sequence authoring. Authoring helpers are also available in the
more focused {doc}`pulserver.pypulseq <pypulseq>` namespace.

## Types

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.Sequence
   pulserver.PulseqSequence
   pulserver.Module
   pulserver.UIParam
   pulserver.Validate
   pulserver.ParamKind
   pulserver.InputMode
   pulserver.FloatKey
   pulserver.IntKey
   pulserver.BoolKey
   pulserver.EnumKey
   pulserver.SequenceType
   pulserver.ImagingMode
   pulserver.PreparationType
   pulserver.TriggerType
   pulserver.TypeinFloatParam
   pulserver.DropdownFloatParam
   pulserver.TypeinIntParam
   pulserver.DropdownIntParam
   pulserver.BoolParam
   pulserver.StringListParam
   pulserver.Description
```

`Protocol` and `ProtocolValue` are the mapping and value type aliases used by
the protocol helpers below.

## Protocol handling

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.expected_param_kind
   pulserver.enum_options
   pulserver.make_enum_param
   pulserver.validate_protocol_entry
   pulserver.validate_protocol
   pulserver.param_to_dict
   pulserver.dict_to_param
   pulserver.protocol_to_dict
   pulserver.dict_to_protocol
   pulserver.set_protocol_value
   pulserver.run_cli
```

## RF pulses and preparation

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.make_hard_pulse
   pulserver.make_adiabatic_pulse
   pulserver.make_sigpy_pulse
   pulserver.make_slr_pulse
   pulserver.make_frequency_selective_pulse
   pulserver.make_slice_selective_pulse
   pulserver.make_spsp_pulse
   pulserver.make_spatially_selective_pulse
   pulserver.make_spiral_selective_pulse
   pulserver.make_2d_selective_pulse
   pulserver.make_3d_selective_pulse
   pulserver.make_inversion_pulse
   pulserver.make_refocusing_pulse
   pulserver.make_mt_pulse
   pulserver.make_ihmt_pulse
   pulserver.make_bloch_siegert_pulse
   pulserver.make_t2prep_pulse
   pulserver.make_t1t2_prep_pulse
   pulserver.make_diffusion_prep
   pulserver.make_fat_saturation_pulse
```

## Encoding, gradients, and timing

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.make_line_readout
   pulserver.make_epi_readout
   pulserver.make_fse_readout
   pulserver.make_radial_readout
   pulserver.make_spiral_readout
   pulserver.make_rosette_readout
   pulserver.make_noncartesian_3d_readout
   pulserver.make_zte_readout
   pulserver.make_crusher
   pulserver.make_phase_encoding
   pulserver.make_phase_blip
   pulserver.make_spoiler
   pulserver.calc_adc_timing
```

## Sampling and schedules

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.chunk_indices
   pulserver.linear_order
   pulserver.outer_inner_order
   pulserver.outer_product
   pulserver.sampled_lines
   pulserver.fse_linear_order
   pulserver.fse_radial_order
   pulserver.fse_radial_adaptive_order
   pulserver.fse_shuffling_order
   pulserver.random_mask
   pulserver.caipirinha_mask
   pulserver.poisson_disc_mask
   pulserver.from_mask
   pulserver.from_relative_shifts
   pulserver.skipped_caipi
   pulserver.radial_2d
   pulserver.golden_angles
   pulserver.uniform_angles
   pulserver.golden_means_3d
   pulserver.spiral_phyllotaxis
   pulserver.directions_to_rotations
   pulserver.slice_groups
   pulserver.make_rf_spoiling_schedule
   pulserver.make_phase_cycling_schedule
   pulserver.make_traps_schedule
```
