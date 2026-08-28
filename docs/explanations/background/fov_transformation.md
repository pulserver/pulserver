# Repositioning a written sequence

A sequence is designed in its own **logical** frame — readout, phase and
slice — with the imaging volume at the origin. A prescription moves it: an
operator angles the slab, slides it off isocentre, changes the field of view.
Pulseq does not ask the designer to rebuild the sequence for that. `mr.TransformFOV`
takes a scan that already exists and applies a rotation and a translation to
it.

The two halves are not alike, and the difference is what everything on this
page follows from.

## Rotation is a change of axes

A rotation mixes the three gradient channels. It can be applied two ways, and
the object takes either: `mr.rotate3D` bakes it into the waveforms, producing
new gradients; or, under `use_rotation_extension`, the rotation is attached to
each block as a `ROTATIONS` quaternion and left for whatever plays the
sequence to resolve.

The second costs nothing in waveforms. A radial scan of a thousand spokes
that bakes its rotations holds a thousand readout waveforms; the same scan
carrying an extension holds one, and a thousand quaternions.

## Translation is a phase

Shifting the imaging volume by $\Delta r$ multiplies the signal by
$e^{-i 2\pi\, \Delta r \cdot k(t)}$, so a translation never touches a
gradient. It is a phase, and where that phase can be put depends on how
$k(t)$ behaves while an event is running.

```{figure} ../assets/pulseq/transform_fov_walk.png
What `mr.TransformFOV` does to one block. The gradient is integrated to get
$k$; whether the result is two scalars or a vector depends on the gradient
being flat across the event, and either way the accumulated phase is carried
into the next block.
```

Per block, per axis with a shift on it, the gradient is turned into a
piecewise polynomial and integrated analytically. That integral is $k$, and
what happens next is a case split:

**The gradient is constant across the event.** Then $k$ advances linearly, so
the phase is linear in time — which is exactly a frequency offset plus a
constant phase offset. Two scalars go onto the RF event or the ADC, and no
consumer downstream has to know anything happened. A Cartesian readout, whose
readout gradient is flat while the ADC is open, is this case.

**The gradient is not constant.** Then no frequency offset describes the
phase, and it has to be given sample by sample. For an RF pulse that is easy:
the phase is multiplied into `rf.signal` and the pulse is still one waveform.
For an ADC there is nowhere to put it — the samples do not exist yet — so it
goes into `adc.phaseModulation`, an array of one phase per sample, and the
reconstruction is obliged to undo it. Spirals, cones and any readout that
ramps while acquiring are this case.

## The phase accumulates across blocks

$k$ at the start of a block is $k$ at the end of the one before it, so
`applyToBlock` carries a running `prior_phase_cycle` from block to block.
That is the state that makes the transformation a sequential walk: block $n$
cannot be transformed without having transformed every block before it.

Excitation and refocusing pulses are what reset it — an excitation puts
magnetisation back at $k = 0$, a refocusing pulse negates it — so the
accumulated value is not a running total of everything played, but the real
k-space position of the magnetisation the readout is about to measure.

## Three labels exempt a block

`NOPOS`, `NOROT` and `NOSCL` are sticky labels that switch off translation,
rotation and scaling for the blocks they cover. They exist because not every
block belongs to the prescribed volume: a fat-saturation slab, a navigator
aimed elsewhere, a spoiler whose direction is arbitrary. A block already
positioned by its own design says so, and the transformation leaves it alone.

## What this costs

`applyToSeq` walks the sequence with `getBlock` and rebuilds it with
`addBlock`. Every event is decoded into a struct, modified and re-registered,
so a scan of a million blocks is a million round trips through the event
libraries — and the running phase means they cannot be done in any other
order. On demonstration-sized sequences that is unnoticeable; at protocol
sizes it is the dominant cost of a prescription change.

Pulserver keeps the object and its semantics, including the three labels, and
changes how the walk is executed and what a non-constant readout stores. See
{doc}`../performance/transform_fov`.
