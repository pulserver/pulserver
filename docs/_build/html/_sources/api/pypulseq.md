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

Sections below are organised by role, not by package layout; Pulserver
extensions and their upstream counterparts appear side by side.

## Sequence and system

The sequence container and the scanner limits every factory is designed
against. Pulserver's `Sequence` subclasses upstream's with a lower per-block
overhead and support for custom labels and rotation/RF-shim extensions.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.Sequence
   pulserver.pypulseq.Opts
   pulserver.pypulseq.SigpyPulseOpts
```

## RF pulses — excitation and refocusing

Slice-, slab-, frequency-selective and non-selective pulses. The Pulserver
factories (`make_hard_pulse` and below) return a stateful
{class}`~pulserver.Module` carrying the RF event together with its selection
and rephasing gradients, so a whole excitation is appended in one call and
re-offset per slice. The upstream factories return bare events.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_hard_pulse
   pulserver.pypulseq.make_slr_pulse
   pulserver.pypulseq.make_sigpy_pulse
   pulserver.pypulseq.make_frequency_selective_pulse
   pulserver.pypulseq.make_slice_selective_pulse
   pulserver.pypulseq.make_refocusing_pulse
   pulserver.pypulseq.make_inversion_pulse
   pulserver.pypulseq.make_adiabatic_pulse
   pulserver.pypulseq.make_block_pulse
   pulserver.pypulseq.make_sinc_pulse
   pulserver.pypulseq.make_gauss_pulse
   pulserver.pypulseq.make_arbitrary_rf
```

## RF pulses — multidimensional and spectral-spatial

Pulses whose selectivity comes from playing RF concurrently with a gradient
trajectory: small-tip 2D/3D excitation on a spiral or EPI path, and
alternating-gradient spectral-spatial design.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_spsp_pulse
   pulserver.pypulseq.make_spatially_selective_pulse
   pulserver.pypulseq.make_spiral_selective_pulse
   pulserver.pypulseq.make_2d_selective_pulse
   pulserver.pypulseq.make_3d_selective_pulse
```

## Magnetization preparation

Multi-block preparation modules — each bundles its pulses, inter-pulse delays
and terminating spoiler into one appendable unit, with the scope labels that
keep it out of the imaging FOV transform.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_fat_saturation_pulse
   pulserver.pypulseq.make_mt_pulse
   pulserver.pypulseq.make_ihmt_pulse
   pulserver.pypulseq.make_t2prep_pulse
   pulserver.pypulseq.make_t1t2_prep_pulse
   pulserver.pypulseq.make_diffusion_prep
   pulserver.pypulseq.make_bloch_siegert_pulse
```

## Readout modules

One factory per readout family. Each returns a stateful
{class}`~pulserver.Module` holding a whole shot — prewinders, echo train, ADCs,
labels and rewinders — that is re-indexed per shot via `set_state` or by
calling it. Dimensionality is inferred from `matrix`.

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
```

## Gradients

Single gradient events and the algebra over them. `make_crusher`,
`make_spoiler`, `make_phase_encoding` and `make_phase_blip` express the
gradient in imaging terms — dephasing cycles, FOV, matrix — rather than area.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_trapezoid
   pulserver.pypulseq.make_arbitrary_grad
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
   pulserver.pypulseq.traj2grad
```

## Arbitrary-gradient design

`pulserver.pypulseq.arbgrad` designs slew- and amplitude-limited **base
waveforms** for non-Cartesian trajectories (spiral, rosette, arbitrary
k-space paths) with a vendored MRArbGrad C++ core, and reports how many shots
cover k-space. Shot ordering and rotation stay in Python.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.arbgrad
```

## ADC and timing

ADC event creation, raster-feasible readout timing, and the duration/centre
queries used to solve TE and TR budgets.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_adc
   pulserver.pypulseq.calc_adc_timing
   pulserver.pypulseq.calc_adc_segments
   pulserver.pypulseq.make_delay
   pulserver.pypulseq.make_soft_delay
   pulserver.pypulseq.calc_duration
   pulserver.pypulseq.calc_rf_center
   pulserver.pypulseq.calc_rf_bandwidth
   pulserver.pypulseq.calc_ramp
   pulserver.pypulseq.align
```

Timing is validated with `Sequence.check_timing()`.

## Sampling patterns

Which k-space locations are acquired, and in which order — kept numeric and
sequence-independent. A `SamplingPattern` pairs the sampled `support` with
per-shot index arrays; readout modules consume its coordinates. See the
{doc}`sampling reference <../reference/sampling>` for the full model.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.SamplingPattern
```

### Cartesian masks and echo-train ordering

Undersampling masks with a calibration region, and the orderings that split a
(ky, kz) point set into FSE echo trains or MPRAGE segments.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.sampled_lines
   pulserver.pypulseq.random_mask
   pulserver.pypulseq.caipirinha_mask
   pulserver.pypulseq.poisson_disc_mask
   pulserver.pypulseq.from_mask
   pulserver.pypulseq.fse_linear_order
   pulserver.pypulseq.fse_radial_order
   pulserver.pypulseq.fse_radial_adaptive_order
   pulserver.pypulseq.fse_shuffling_order
```

### EPI and relative-shift trains

Plans expressed as a per-shot start plus a list of blip increments — the
natural form for EPI, segmented blipped-CAIPI, and EPTI-like schedules.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.from_relative_shifts
   pulserver.pypulseq.skipped_caipi
```

### Non-Cartesian tilts

Spoke and interleaf directions for radial and 3D centre-out acquisitions, and
their conversion to rotation matrices for a rotated base waveform.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.radial_2d
   pulserver.pypulseq.golden_angles
   pulserver.pypulseq.uniform_angles
   pulserver.pypulseq.golden_means_3d
   pulserver.pypulseq.spiral_phyllotaxis
   pulserver.pypulseq.directions_to_rotations
```

### Slice order and outer loops

Slice/SMS grouping with per-band frequency offsets, and the generic helpers
for enumerating the loops outside the readout.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.slice_groups
   pulserver.pypulseq.SliceGroup
   pulserver.pypulseq.outer_product
   pulserver.pypulseq.outer_inner_order
   pulserver.pypulseq.linear_order
   pulserver.pypulseq.chunk_indices
```

## RF phase and flip-angle schedules

Per-repetition phase and flip-angle lists: quadratic RF spoiling, arbitrary
phase cycling, and Alsop-style variable refocusing trains.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_rf_spoiling_schedule
   pulserver.pypulseq.make_phase_cycling_schedule
   pulserver.pypulseq.make_traps_schedule
```

## Labels, rotations, and extensions

Per-block metadata: counter labels consumed by the reconstruction, block
rotations, RF shim vectors, and hardware triggers. Pulserver's `make_label`
accepts user-defined label strings in addition to the built-in set.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_label
   pulserver.pypulseq.get_supported_labels
   pulserver.pypulseq.make_rotation
   pulserver.pypulseq.rotate
   pulserver.pypulseq.make_rf_shim
   pulserver.pypulseq.make_trigger
   pulserver.pypulseq.make_digital_output_pulse
```

## Shapes and utilities

Pulseq shape compression, unit conversion, and tracing helpers. Upstream binds
`compress_shape`, `decompress_shape` and `convert` to their *modules* rather
than to the functions, so `pp.compress_shape` is a module; the callables are
documented below at their canonical location and are reached as
`pp.compress_shape.compress_shape`.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pypulseq.compress_shape.compress_shape
   pypulseq.decompress_shape.decompress_shape
   pypulseq.convert.convert
   pulserver.pypulseq.round_half_up
   pulserver.pypulseq.enable_trace
   pulserver.pypulseq.disable_trace
```
