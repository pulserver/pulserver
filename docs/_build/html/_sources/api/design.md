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
    system, slab.rf, slab.gz,
    fov=(0.22, 0.22, 0.12), matrix=(128, 128, 64),
    te=4e-3, tr=10e-3, spoiling_cycles=4.0,
)

phases = pp.make_rf_spoiling_schedule(lines * partitions)
seq = pp.Sequence(system)
for shot, (line, partition) in enumerate(plan):
    readout.rf.phase_offset = readout.adc.phase_offset = phases[shot]
    seq.add_block(readout.rf, readout.gz)
    seq.add_block(readout.wait_te)
    seq.add_block(
        readout.gx_pre,
        pp.scale_grad(readout.gy_pre, ky[line]),
        pp.scale_grad(readout.gz_pre, kz[partition]),
    )
    seq.add_block(readout.gx, readout.adc, *readout.adc_labels)
    seq.add_block(readout.gx_spoil, readout.gy_rew, readout.gz_rew)
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

The refocusing pulses are their spin-echo counterparts: a nominal 180 phased a
quarter turn away, between crushers stated in the same cycles-per-voxel
vocabulary as everything else. They are events like any other, so a readout
given one instead of an excitation is a spin echo rather than a gradient echo.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.NonSelectiveExcitation
   pulserver.design.SpatialSelectiveExcitation
   pulserver.design.NonSelectiveRefocusing
   pulserver.design.SpatialSelectiveRefocusing
   pulserver.design.Inversion
```

## Magnetization preparation

Multi-block preparations: the pulses and the spoiler that terminates them.
The recovery time is deliberately absent — it is the gap between the
preparation and the readout that samples it, so it belongs to the loop that
plays both.

The T2 family stores weighting on the longitudinal axis, so the contrast is
decoupled from the readout that follows: a bSSFP or a short-TE gradient echo
can carry T2 contrast it could never generate itself. Every pulse is
adiabatic, and the refocusing pulses come in pairs — one adiabatic full
passage leaves a transmit-dependent phase that only a second one undoes, so an
odd count is refused rather than corrected.

The off-resonance family — magnetization transfer, its inhomogeneous
difference, and the Bloch–Siegert pulse — shares one shape: a strong pulse
placed away from water, played a few times, and usually spoiled. Only the
envelope and the offset differ, which is all
{class}`~pulserver.design.OffResonanceSaturation` leaves to a subclass.
`IhMtPreparation` is the one place the multiband wrapper appears: its two
saturation bands are one envelope modulated into two, matched to the
single-offset arm on **power** rather than amplitude, so the ihMT difference is
a difference in inhomogeneous saturation and not in deposited energy.

`DiffusionPreparation` is designed once, along z and at its largest b-value; a
direction is a rotation the loop applies and a lower b-value is a square-root
scaling of the same waveform. Its b-value is integrated from the rasterized
waveform rather than taken from the δ/Δ formula, which the ramps put a few
percent out.

`FatSaturation` is the one module that positions itself. A band pointed
somewhere other than the imaging prescription applies its own
{class}`~pulserver.pypulseq.TransformFOV` at design time, then declares
`NOPOS` and `NOROT` on the block that carries the pulse — so a transform
applied to the finished scan moves the imaging volume and leaves the band
where it was put. The flags are cleared on the module's last block, because
Pulseq labels are sticky.

```{eval-rst}
.. autosummary::
   :toctree: generated/design
   :nosignatures:

   pulserver.design.InversionPreparation
   pulserver.design.T2Preparation
   pulserver.design.T1T2Preparation
   pulserver.design.MtPreparation
   pulserver.design.IhMtPreparation
   pulserver.design.BlochSiegertPreparation
   pulserver.design.DiffusionPreparation
   pulserver.design.FatSaturation
```

## Readouts

One class per geometry, each spanning a **whole repetition**: the pulse, the
prewinder, the acquisition, the rewinder, and whatever TE and TR were asked
for. The pulse arrives as an *event* rather than as a module, so the same
class serves an excitation — giving a gradient echo — or a refocusing pulse,
giving the second half of a spin echo.

The geometry is what earns a class, because the geometry is what changes the
blocks. Everything else is an argument: `direction` runs a spiral outward,
inward or in-out; `n_echoes` and `flyback` choose an echo train; `density`
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
   pulserver.design.OffResonanceSaturation
```
