# `pulserver.pypulseq`

The drop-in replacement for `pypulseq`, and the single namespace for
everything that produces a waveform, an event, or an acquisition plan. The
complete upstream PyPulseq namespace is re-exported, then Pulserver's
`Sequence` and its extensions are layered on top, so one import covers both:

```python
import pulserver.pypulseq as pp

seq = pp.Sequence()
excitation = pp.make_slice_selective_pulse(0.35, 5e-3, system=pp.Opts())
readout = pp.make_line_readout(pp.Opts(), (0.22, 0.22), (128, 128))
```

Sections below run from the smallest unit outwards: single events, then the
multi-block modules built from them, then the plans that drive those modules.
Within specialized sections the entries follow the design workflow.

The `make_*_pulse` and `make_*_readout` factories return a
{class}`~pulserver.SequenceModule` — a stateful, reusable fragment designed
once and re-indexed per shot. High-level sampling factories return an
{class}`~pulserver.AcquisitionPlan`; lower-level masks, orderings and tilts
remain independently composable through {class}`~pulserver.SamplingPattern`.
These types are documented in {doc}`pulserver <pulserver>`.

## Base PyPulseq

Single events, timing arithmetic, shape utilities and per-block metadata:
the upstream namespace, plus the handful of Pulserver replacements that
operate at the same event level. `Sequence` subclasses upstream's with a
lower per-block overhead; `make_label` accepts user-defined label strings;
`make_rotation` and `make_rf_shim` are Pulserver extension events;
`traj2grad` supersedes upstream `traj_to_grad`, re-parameterising a k-space
path in time instead of differentiating whatever sampling it arrived with.
Pulserver's `Opts` uses zero dead/ringdown times and vendor-neutral 2 us
RF/ADC and 10 us gradient/block rasters. Scanner-specific code should still
pass explicit limits.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.Opts
   pulserver.pypulseq.Sequence
   pulserver.pypulseq.add_gradients
   pulserver.pypulseq.align
   pulserver.pypulseq.calc_adc_segments
   pulserver.pypulseq.calc_adc_timing
   pulserver.pypulseq.calc_duration
   pulserver.pypulseq.calc_ramp
   pulserver.pypulseq.calc_rf_bandwidth
   pulserver.pypulseq.calc_rf_center
   pulserver.pypulseq.compress_shape
   pulserver.pypulseq.convert
   pulserver.pypulseq.decompress_shape
   pulserver.pypulseq.disable_trace
   pulserver.pypulseq.enable_trace
   pulserver.pypulseq.get_supported_labels
   pulserver.pypulseq.make_adc
   pulserver.pypulseq.make_arbitrary_grad
   pulserver.pypulseq.make_arbitrary_rf
   pulserver.pypulseq.make_block_pulse
   pulserver.pypulseq.make_delay
   pulserver.pypulseq.make_digital_output_pulse
   pulserver.pypulseq.make_extended_trapezoid
   pulserver.pypulseq.make_extended_trapezoid_area
   pulserver.pypulseq.make_gauss_pulse
   pulserver.pypulseq.make_label
   pulserver.pypulseq.make_rf_shim
   pulserver.pypulseq.make_rotation
   pulserver.pypulseq.make_sinc_pulse
   pulserver.pypulseq.make_soft_delay
   pulserver.pypulseq.make_trapezoid
   pulserver.pypulseq.make_trigger
   pulserver.pypulseq.points_to_waveform
   pulserver.pypulseq.rotate
   pulserver.pypulseq.round_half_up
   pulserver.pypulseq.scale_grad
   pulserver.pypulseq.split_gradient
   pulserver.pypulseq.split_gradient_at
   pulserver.pypulseq.traj2grad
```

Timing is validated with `Sequence.check_timing()`. Unlike upstream,
`compress_shape`, `decompress_shape` and `convert` resolve directly to
callables in this namespace.

## Gradients

The gradient events Pulserver adds, expressed in imaging terms — dephasing
cycles, FOV, matrix — rather than in area.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_crusher
   pulserver.pypulseq.make_phase_blip
   pulserver.pypulseq.make_phase_encoding
```

## RF pulses — excitation and refocusing

Slice-, slab- and frequency-selective pulses, and the non-selective ones. Each
returns a module carrying the RF event together with its selection and
rephasing gradients, so a whole excitation is appended in one call and
re-offset per slice. `make_slr_pulse` designs its FIR filter in-package, so
SigPy is not a dependency; it is also exported as `make_sigpy_pulse` for
compatibility with the pypulseq factory it replaces.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_hard_pulse
   pulserver.pypulseq.make_frequency_selective_pulse
   pulserver.pypulseq.make_slice_selective_pulse
   pulserver.pypulseq.make_refocusing_pulse
   pulserver.pypulseq.make_slr_pulse
   pulserver.pypulseq.make_multiband_frequency_selective_pulse
   pulserver.pypulseq.make_pins_slice_selective_pulse
   pulserver.pypulseq.make_multiband
   pulserver.pypulseq.make_pins
```

## RF pulses — multidimensional and spectral-spatial

The supported plane-selective spiral design and alternating-gradient
spectral-spatial design. Generic caller-supplied and unvalidated 3D variants
are intentionally kept out of the public API.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_plane_selective_pulse
   pulserver.pypulseq.make_spsp_pulse
```

## Magnetization preparation

Multi-block preparation modules — each bundles its pulses, inter-pulse delays
and terminating spoiler into one appendable unit, with the scope labels that
keep it out of the imaging FOV transform.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_bloch_siegert_pulse
   pulserver.pypulseq.make_diffusion_prep
   pulserver.pypulseq.make_fat_saturation_pulse
   pulserver.pypulseq.make_ihmt_pulse
   pulserver.pypulseq.make_inversion_pulse
   pulserver.pypulseq.make_mt_pulse
   pulserver.pypulseq.make_t1t2_prep_pulse
   pulserver.pypulseq.make_t2prep_pulse
```

## Readouts

One factory per readout family. Each returns a module holding a whole shot —
prewinders, echo train, ADCs, labels, rewinders — that is re-indexed per shot
via `set_state` or by calling it. For the Cartesian families the
dimensionality is inferred from `matrix`.

The non-Cartesian families come in three coverages of the same base waveform:
the plain factory rotates it *in plane* (2D imaging), `*_projection_*` steers
it over the sphere (kooshball radial, spiral projection), and `*_stack_*`
rotates it in plane while encoding kz conventionally (stack of stars, stack of
spirals). `make_noncartesian_2d_readout` and `make_noncartesian_3d_readout`
take a k-space path that is *not* a rotated copy of a base waveform and design
a gradient for it directly.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_epi_readout
   pulserver.pypulseq.make_fse_readout
   pulserver.pypulseq.make_line_readout
   pulserver.pypulseq.make_noncartesian_2d_readout
   pulserver.pypulseq.make_noncartesian_3d_readout
   pulserver.pypulseq.make_radial_projection_readout
   pulserver.pypulseq.make_radial_readout
   pulserver.pypulseq.make_radial_stack_readout
   pulserver.pypulseq.make_rosette_projection_readout
   pulserver.pypulseq.make_rosette_readout
   pulserver.pypulseq.make_rosette_stack_readout
   pulserver.pypulseq.make_spiral_projection_readout
   pulserver.pypulseq.make_spiral_readout
   pulserver.pypulseq.make_spiral_stack_readout
   pulserver.pypulseq.make_zte_readout
```

## Sampling

High-level plans accept plain UI-facing values such as matrix,
acceleration, ETL/segment length, slice count, SMS factor and frame count.
They do not require a protocol object and remain sequence-independent.
Ordering functions and
mask/tilt generators form the lower-level layer. Construct a
`SamplingPattern` directly from custom support/order, or use its `from_mask`
and `from_relative_shifts` constructors. See the
{doc}`sampling reference <../reference/sampling>` for the full model.

### High-level acquisition sampling

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_cartesian_sampling
   pulserver.pypulseq.make_epi_sampling
   pulserver.pypulseq.make_noncartesian_2d_sampling
   pulserver.pypulseq.make_noncartesian_stack_sampling
   pulserver.pypulseq.make_noncartesian_projection_sampling
```

### Low-level orderings

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_skipped_caipi_order
   pulserver.pypulseq.make_centric_order
   pulserver.pypulseq.make_linear_order
   pulserver.pypulseq.make_radial_adaptive_order
   pulserver.pypulseq.make_radial_order
   pulserver.pypulseq.make_shuffling_order
```

### Low-level masks and tilts

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_caipirinha_mask
   pulserver.pypulseq.make_poisson_disc_mask
   pulserver.pypulseq.make_random_mask
   pulserver.pypulseq.make_radial_tilt
   pulserver.pypulseq.make_golden_means_3d_tilt
   pulserver.pypulseq.make_spiral_phyllotaxis_tilt
   pulserver.pypulseq.calc_sampled_lines
   pulserver.pypulseq.calc_golden_angles
   pulserver.pypulseq.calc_raga_angles
   pulserver.pypulseq.calc_tiny_golden_angles
   pulserver.pypulseq.calc_uniform_angles
```

### Independent outer-loop utilities

These remain useful when directly combining a `SamplingPattern` with slice,
frame, average or contrast loops. The high-level 2D factories already
incorporate slice/SMS grouping and expose the resulting frequency table.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.calc_chunk_indices
   pulserver.pypulseq.make_outer_inner_order
   pulserver.pypulseq.make_outer_product
   pulserver.pypulseq.make_slice_sampling
```

## RF phase and flip-angle schedules

Per-repetition phase and flip-angle lists: quadratic RF spoiling, arbitrary
phase cycling, and Alsop-style variable refocusing trains.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_phase_cycling_schedule
   pulserver.pypulseq.make_rf_spoiling_schedule
   pulserver.pypulseq.make_traps_schedule
```
