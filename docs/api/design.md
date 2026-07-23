# `pulserver.design`

The design toolbox: one factory per thing Pulserver knows how to build. Every
entry on this page returns one of the two authoring types the plugin contract
exchanges — a {class}`~pulserver.SequenceModule` (a stateful, reusable
multi-block fragment, designed once and re-indexed per shot) or a
{class}`~pulserver.ScanLoop` (a table of encoding positions grouped into
shots). Both types are documented in {doc}`pulserver <pulserver>`.

```python
import pulserver.design as design
import pulserver.pypulseq as pp

system = pp.Opts()
excitation = design.make_slice_selective_pulse(0.35, 5e-3, system=system)
readout = design.make_line_readout(system, (0.22, 0.22), (128, 128))
loop = design.make_cartesian_sampling((128, 128), acceleration=2)

seq = pp.Sequence(system)
for shot in loop:
    for block in excitation:
        seq.add_block(*block)
    readout.set_state(lin_idx=int(shot[0, 0]))
    for block in readout:
        seq.add_block(*block)
```

`pulserver.design` sits on top of {doc}`pulserver.pypulseq <pypulseq>`, which
is the event layer: `Sequence`, `Opts`, `make_trapezoid` and the rest of
PyPulseq. The split is by role, and the two namespaces share no names.

Nothing here iterates the sequence for you: the loop nesting stays the
plugin's own `for` statements. Sections run from the smallest unit outwards —
single gradient events, the modules built from them, then the loops that drive
those modules.

## Gradients

The gradient events Pulserver adds, expressed in imaging terms — dephasing
cycles, FOV, matrix — rather than in area.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.make_crusher
   pulserver.design.make_phase_blip
   pulserver.design.make_phase_encoding
```

## RF pulses — excitation and refocusing

Slice-, slab- and frequency-selective pulses, and the non-selective ones. Each
returns a module carrying the RF event together with its selection and
rephasing gradients, so a whole excitation is appended in one call and
re-offset per slice. `make_slr_pulse` designs its FIR filter in-package, so
SigPy is not a dependency; it is also exported as `make_sigpy_pulse` for
compatibility with the PyPulseq factory it replaces.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.make_hard_pulse
   pulserver.design.make_frequency_selective_pulse
   pulserver.design.make_slice_selective_pulse
   pulserver.design.make_refocusing_pulse
   pulserver.design.make_slr_pulse
   pulserver.design.make_multiband_frequency_selective_pulse
   pulserver.design.make_pins_slice_selective_pulse
   pulserver.design.make_multiband
   pulserver.design.make_pins
```

## RF pulses — multidimensional and spectral-spatial

The supported plane-selective spiral design and alternating-gradient
spectral-spatial design. Generic caller-supplied and unvalidated 3D variants
are intentionally kept out of the public API.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.make_plane_selective_pulse
   pulserver.design.make_spsp_pulse
```

## Magnetization preparation

Multi-block preparation modules — each bundles its pulses, inter-pulse delays
and terminating spoiler into one appendable unit, with the scope labels that
keep it out of the imaging FOV transform.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.make_bloch_siegert_pulse
   pulserver.design.make_diffusion_prep
   pulserver.design.make_fat_saturation_pulse
   pulserver.design.make_ihmt_pulse
   pulserver.design.make_inversion_pulse
   pulserver.design.make_mt_pulse
   pulserver.design.make_t1t2_prep_pulse
   pulserver.design.make_t2prep_pulse
```

## Readouts

One factory per readout family. Each returns a module holding a whole shot —
prewinders, echo train, ADCs, labels, rewinders — that is re-indexed per shot
via `set_state`. For the Cartesian families the dimensionality is inferred
from `matrix`.

Most of them also take a `slice_rephasing` argument, which folds the
excitation's rephaser into the prewinder instead of paying for it as its own
block; see {doc}`../how-to/write_a_plugin_with_modules` for how to use it.

The non-Cartesian families come in three coverages of the same base waveform:
the plain factory rotates it *in plane* (2D imaging), `*_projection_*` steers
it over the sphere (kooshball radial, spiral projection), and `*_stack_*`
rotates it in plane while encoding kz conventionally (stack of stars, stack of
spirals). `make_noncartesian_2d_readout` and `make_noncartesian_3d_readout`
take a k-space path that is *not* a rotated copy of a base waveform and design
a gradient for it directly.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.make_epi_readout
   pulserver.design.make_bssfp_readout
   pulserver.design.make_fse_readout
   pulserver.design.make_line_readout
   pulserver.design.make_noncartesian_2d_readout
   pulserver.design.make_noncartesian_3d_readout
   pulserver.design.make_radial_projection_readout
   pulserver.design.make_radial_readout
   pulserver.design.make_radial_stack_readout
   pulserver.design.make_rosette_projection_readout
   pulserver.design.make_rosette_readout
   pulserver.design.make_rosette_stack_readout
   pulserver.design.make_spiral_projection_readout
   pulserver.design.make_spiral_readout
   pulserver.design.make_spiral_stack_readout
   pulserver.design.make_zte_readout
```

## Scan loops

Matrix-driven factories accept plain UI-facing values — matrix, acceleration,
ETL/segment length, view count — and return a `ScanLoop` covering k-space.
The slice loop is an ordinary `ScanLoop` whose positions are in metres:
`make_slice_loop` returns the excitation order and the physical positions
{meth}`~pulserver.ScanLoop.to_frequencies` converts to RF offsets.
`make_counter_loop` is the same object for every other dimension — frames on
`REP`, contrasts on `SET` or `PHS`, averages on `AVG` — either as a bare count
or as the table of values it schedules.

`calc_encoding_scales` turns a loop's integer positions into the gradient
scale factors a readout's `set_state` consumes. See the {doc}`scan-loop
reference <../reference/sampling>` for the full model.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.make_cartesian_sampling
   pulserver.design.make_epi_sampling
   pulserver.design.make_noncartesian_2d_sampling
   pulserver.design.make_noncartesian_projection_sampling
   pulserver.design.make_slice_loop
   pulserver.design.make_counter_loop
   pulserver.design.calc_encoding_scales
```

## RF phase and flip-angle schedules

Per-repetition phase and flip-angle lists: quadratic RF spoiling, arbitrary
phase cycling, and Alsop-style variable refocusing trains.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.make_phase_cycling_schedule
   pulserver.design.make_rf_spoiling_schedule
   pulserver.design.make_traps_schedule
```

## Waveform and timing helpers

`traj2grad` supersedes upstream `traj_to_grad`: it re-parameterises a k-space
path in time so the result honours `max_grad` and `max_slew` by construction,
instead of differentiating whatever sampling the path arrived with.
`calc_adc_timing` solves for a dwell time and duration feasible on both the
ADC and gradient rasters at once.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.calc_adc_timing
   pulserver.design.traj2grad
```
