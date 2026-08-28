# Applying a prescription

An operator angles the slab, slides it off isocentre and changes the field of
view, and the console expects the answer before it redraws. What the
prescription means for a written sequence is
{doc}`Pulseq's own transformation <../background/fov_transformation>`: a
rotation, a translation that is a phase, and a scaling. This page is what it
costs here.

Three properties of the {doc}`representation <../sequence_model/pulseg_representation>`
carry the whole page, and none of them changes what the transformation
computes.

## The running phase is one pass, not a walk

The translation needs to know where absolute $k$ stands at the start of every
block. In the reference implementation that is a scalar carried by hand from
block to block, which is what forces the sequential walk.

The engine already tracks the same quantity for the whole sequence in one
pass: `pulseq::block_k_origins` returns $k$ at the start of every block,
reset by an excitation and negated by a refocusing pulse. That *is* the
running phase, written as a k-space position rather than as an accumulated
scalar. With it in hand, applying a shift needs nothing from Python but the
block range to apply it over — no `getBlock`, no `addBlock`, and no event
decoded into a struct and registered again.

The labels work the same way. `NOSCL`, `NOPOS` and `NOROT` are read as
**runs** — `label_gate_runs` returns the block ranges each label leaves alone
— so a scan with no exemptions is one call per transformation, and a scan
with a fat-saturation module is three or four. Nothing here is ever a Python
loop over the blocks of a scan.

## Deduplication comes first

`apply_to_sequence` calls `remove_duplicates` before it does anything else.
A scan arrives holding one gradient row per use; everything below — the
trajectory the shift integrates, the memo that keys its work — then costs the
size of the *libraries* rather than the length of the scan. It is a no-op on
a sequence already deduplicated, so a caller that does it itself pays nothing
twice.

## A scaling and a rotation touch no samples

A gradient is stored as a normalised shape beside a scalar amplitude, so
scaling multiplies the amplitude and leaves the waveform alone. No shape is
registered, and two gradients that differed only in amplitude before still
share their shape afterwards.

Rotation is attached as a `ROTATIONS` extension rather than baked into new
gradients, so one waveform serves every orientation that uses it. Only
`use_rotation_extension=True` is supported, and that is the point: baking
rotations is what makes a large non-Cartesian scan unaffordable, and the
exemption labels exist precisely so a module that must *not* be rotated again
downstream can say so.

## What a non-Cartesian readout stores

The remaining cost is the one the reference implementation cannot avoid.
Where the gradient is not flat across the ADC window, the shift is not two
scalars, and the phase has to be given per sample — one array of
`num_samples` in `adc.phaseModulation`, per readout. A stack of spirals with
a few hundred arms stores a few hundred such arrays, and no two are equal,
because each carries its own angle and its own amplitude.

```{figure} ../assets/transform_fov/base_trajectory.png
Six readouts that differ only by rotation. Stored as a phase per sample they
are six unequal arrays; stored as the trajectory before the angle and the
amplitude are applied, they are one row that all six point at.
```

Under `compat=False` the readout stores its **base trajectory** instead:
the k-space curve the gradient waveforms span *before* their per-instance
amplitudes and rotation are applied, normalised by the ADC window duration.
A consumer of ours recovers the readout's k with what the ADC row already
carries,

$$
k_a(t) = A_a \cdot n_\text{samples} \cdot \Delta t \cdot b_a(t),
$$

then applies the block's rotation. What collapses is the storage: readouts
that differ only by rotation share one row outright, and the ones that also
scale share it too, because the amplitude is a relative factor that survives
the unit conversion. One row per *distinct trajectory* replaces one array per
*readout*.

Three details make that affordable and are worth stating, because each of
them is a decision that could have gone the other way:

**Three axes, always.** An axis flat across the window is stored flat rather
than omitted, so nothing has to be inferred from a length — a two-column
shape cannot distinguish $(x, y)$ from $(x, z)$.

**Axis-major, not interleaved.** The shape codec run-length encodes a
*derivative*. Laid out as all of $x$, then all of $y$, then all of $z$, a flat
axis costs three numbers; interleaved, three axes alternate every sample and
compress to nothing. That compression is the entire reason storing three axes
is affordable.

**Normalised by the window.** `compress_shape` quantises onto an absolute
$10^{-7}$ grid. Left in seconds — order $10^{-4}$, stepping by $10^{-5}$ —
a readout would have about a hundred levels across it. Dividing by
$n_\text{samples} \Delta t$ puts the samples at order one, where the grid is
not a limitation.

## What the overload costs

`phase_modulation` is the only field of the right shape, and putting a
trajectory in it is an overload of a field the format defines as a phase. A
toolbox that applies it literally gets a plausible wrong image rather than an
error, which is the worst kind of failure — so a file that carries base
trajectories *says* so, in `[DEFINITIONS] PhaseModulationMode`. Unknown
definitions are ignorable, so the marker costs nothing to a reader that does
not care, and a reader that does care needs one lookup.

The trade is worth naming plainly. Compatibility with a stock Pulseq
reconstruction is lost for exactly those readouts, and it is recovered two
ways: the overload is opt-out, so `compat=True` bakes every shift the
way the reference does and the file needs nothing downstream; and a consumer
that wants both can read the marker and rescale, which is the same arithmetic
it would have done anyway to turn a phase back into a position.

There is one thing the overload buys that the phase never could. A base
trajectory is not only how a deferred shift is finished — it is the readout's
$k$, which a non-Cartesian reconstruction needs whether or not the volume was
moved. It is written whenever `compat` is cleared, shift or no shift, and a
scan prescribed at isocentre gets it too.
