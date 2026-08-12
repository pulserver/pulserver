# `pulserver.design`

Reusable, composable sequence modules. Every class on this page is a
{class}`~pulserver.SequenceModule`: a handful of Pulseq blocks that always
travel together — an excitation with its slice-select and rephaser, a
preparation with its spoiler, one whole readout TR. It designs its waveforms
once, at construction, and publishes them under the names its constructor gave
them.

```python
import pulserver.design as design
import pulserver.pypulseq as pp

system = pp.Opts()
slab = design.SpatialSelectiveExcitation(system, 8.0, 0.12, is_slab=True)
readout = design.LineReadout3D(
    system, slab.rf, slab.gz_slab,
    fov_m=(0.22, 0.22, 0.12), matrix=(128, 128, 64),
    te=4e-3, tr=10e-3, spoiling_cycles=4.0,
)

phases = pp.make_rf_spoiling_schedule(lines * partitions)
seq = pp.Sequence(system)
for shot, (line, partition) in enumerate(plan):
    readout.rf.phase_offset = readout.adc.phase_offset = phases[shot]
    seq.add_block(readout.rf, readout.gz_select)
    seq.add_block(readout.wait_te)
    seq.add_block(
        readout.gx_pre,
        pp.scale_grad(readout.gy_phase, ky[line]),
        pp.scale_grad(readout.gz_phase, kz[partition]),
    )
    seq.add_block(readout.gx_read, readout.adc, *readout.adc_labels)
    seq.add_block(readout.gx_rew, readout.gy_rew, readout.gz_rew)
    seq.add_block(readout.wait_tr)
```

Two things about that loop are the whole design.

**A module never writes a scan loop, and is never required.** It owns the
design — solving gradients, budgeting TE and TR, landing the ADC on both
rasters — and hands back named events. The `for` statements, the encoding
plan and the order of the shots stay the plugin's, and a plugin that would
rather build its events by hand loses nothing but the convenience.

**Per-shot variation is ordinary PyPulseq.** There is no state to set: a
phase is an attribute write, an encode is {func}`~pypulseq.scale_grad`, an
orientation is a rotation event. The events a module publishes are the very
objects its blocks hold, so a write shows through immediately.

`pulserver.design` sits on top of {doc}`pulserver.pypulseq <pypulseq>`, which
is the event layer: `Sequence`, `Opts`, `make_trapezoid`, and also the masks,
orderings, angle generators and phase schedules an encoding plan is built
from — those return plain arrays, so they belong there rather than here.

## Excitation

What tips the magnetization, and where. Each carries the RF event together
with any selection and rephasing gradients, so a whole excitation is one
object.

`SpatialSelectiveExcitation` uses an SLR pulse, whose slice profile is a
filter design problem rather than a windowed sinc, and its `is_slab` flag
decides how the rephaser is delivered: separable for a 2D slice, merged into
one gradient for a 3D slab.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.NonSelectiveExcitation
   pulserver.design.SpatialSelectiveExcitation
   pulserver.design.Inversion
```

## Magnetization preparation

Multi-block preparations: the pulses and the spoiler that terminates them.
The recovery time is deliberately absent — it is the gap between the
preparation and the readout that samples it, so it belongs to the loop that
plays both.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.InversionPreparation
```

## Readouts

One class per geometry, each spanning a **whole repetition**: the pulse, the
prewinder, the acquisition, the rewinder, and whatever TE and TR were asked
for. The pulse arrives as an *event* rather than as a module, so the same
class serves an excitation — giving a gradient echo — or a refocusing pulse,
giving the second half of a spin echo.

The geometry is what earns a class, because the geometry is what changes the
blocks. Everything else is an argument: `direction` runs a spiral outward,
inward or in-out; `num_echoes` and `flyback` choose an echo train; `density`
sets a spiral's pitch profile; `spoiling_cycles` with `spoiling_position`
picks between balanced, SSFP-FID and SSFP-Echo.

Spoiling is always stated as cycles of dephasing across a voxel rather than as
an area, so the same numbers mean the same thing at any resolution.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.LineReadout2D
   pulserver.design.LineReadout3D
   pulserver.design.RadialReadout2D
   pulserver.design.RadialStackReadout
   pulserver.design.RadialProjectionReadout
   pulserver.design.SpiralReadout2D
   pulserver.design.SpiralStackReadout
   pulserver.design.SpiralProjectionReadout
   pulserver.design.RosetteReadout2D
   pulserver.design.RosetteStackReadout
   pulserver.design.RosetteProjectionReadout
```

The three coverages of one base interleave: the `*2D*` classes rotate it in
plane, `*Stack*` rotates it in plane while encoding kz conventionally (stack
of stars, stack of spirals), and `*Projection*` steers it over the sphere
(kooshball radial, spiral projection). All three lay out the interleave once
and leave the orientation to the loop; `explicit=True` with `angles=` writes
every rotated arm out instead, for an interpreter that cannot apply a rotation
at runtime.

## Base classes

Subclass one of these only to add a family this package does not ship; see
{doc}`../how-to/write_a_new_module`.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.RfModule
   pulserver.design.NonCartesianReadout
```
