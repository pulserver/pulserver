# Gradient amplitude, slew, continuity and timing

The four checks that need nothing but the sequence and the system, and
therefore always run. They are also the four that catch most real mistakes,
because they are the ones a designer can violate without noticing.

```{figure} ../assets/gradient_slew/vector_and_seam.png
The two mistakes these checks exist to catch. (a) Every axis inside its own
limit is not the same statement as the vector inside the limit. (b) Blocks run
back to back, so the value one ends at and the value the next starts at are
adjacent samples of one waveform.
```

## Amplitude: a vector limit

A gradient amplifier limits each axis, and the *combination* of axes. A
sequence playing 35 mT/m on x and 35 mT/m on y is not playing 35 mT/m: it is
playing 49 mT/m in the direction $(1,1)/\sqrt2$, and a 40 mT/m system will
refuse it.

So the check is on the root-sum-square across axes at every sample, not per
axis. The per-axis test is the special case where the other axes are idle,
which is why an oblique or non-Cartesian sequence can fail a limit that every
individual waveform respects. The same applies after a rotation: a
`ROTATIONS` extension mixes the axes, so a shot that is legal unrotated can be
illegal at some angle, and the check evaluates the rotated waveform.

## Slew, including across block boundaries

Slew is $dG/dt$, checked the same way — root-sum-square, after rotation. The
part that catches people is that a block boundary is not a boundary for the
gradient: blocks run back to back with no gap, so a waveform ending at
20 mT/m followed by a block starting at 0 is a step, and a step is infinite
slew.

Pulserver checks across boundaries for this reason, and reports the pair of
blocks rather than one of them.

## Continuity: a separate statement

A discontinuity is not always a slew violation — it can be a *design* error
that happens to be within the limit. An extended-trapezoid gradient that
starts at an amplitude the previous block did not end at means the waveform
that plays is not the waveform that was designed: the hardware will connect
them, and the moment the sequence delivers is not the moment it computed.

So continuity is checked as itself: every gradient's first sample must meet
its predecessor's last, per axis, and a mismatch is reported as a
discontinuity rather than as a slew number. A sequence built from Pulserver's
own modules cannot produce one; a sequence assembled by hand, or one whose
rewinders were re-solved per rotation angle rather than materialised from a
base waveform, can.

```{note}
Trapezoids and simple arbitrary gradients start and end at zero, so the check
is trivially satisfied for most Cartesian sequences. It earns its place on
continuous-gradient families — ZTE, spirals with bridges, anything where the
readout does not return to zero between blocks.
```

## The step limit

Some systems additionally limit how much the gradient may change *between
successive playouts of a segment* — a view-to-view step, which is what a
rotating non-Cartesian trajectory does at its readout start. Where such a
limit is configured, it is checked over the instance table rather than over
the waveform: a trajectory whose successive views start further apart than the
amplifier tolerates is refused with the pair of views that did it.

## Timing: every event on an addressable grid

The three checks above are about amplitude. The fourth is about *when*: a
sequencer starts and stops events on a raster, and a time that is not an
integer multiple of it cannot be played.

```{figure} ../assets/gradient_slew/raster_alignment.png
An event may start where the raster does. A 14 µs start on a 4 µs grid is not
a rounding question — there is no instruction that begins there.
```

The rasters a file declares are not the answer. The file's rasters are
compared with the scanner's when the collection is built, and that comparison
accepts either direction: a sequence laid out on a raster *finer* than the
scanner's passes it, while still holding times the hardware cannot address. A
15 µs event on a declared 5 µs grid is legal by that test and unplayable on a
10 µs one, which is the case this check exists for.

Each time goes against the raster it is played on:

| Time | Judged against |
|---|---|
| RF and ADC start times | the RF raster |
| ADC dwell | the ADC raster |
| gradient delays, trapezoid rise, flat and fall | the gradient raster |
| block durations | the block-duration raster |
| an arbitrary event's own time shape | its raster, sample by sample |

An event that ends after its block does is reported here too. A block *longer*
than its events is legal and silent — that is a delay, not an error.

Like the amplitude checks, this one is a pass over the *definitions*: every
time field lives in a deduplicated definition, so it costs the number of
distinct events in the scan rather than the number of blocks.

The two sides check slightly different sets, and the difference is deliberate.
`check_timing()` at design time also judges RF dead time and ringdown, ADC
dead time and soft-delay agreement, and reports them entry for entry as
upstream PyPulseq does. The interpreter's pass drops those three: they bound
what a transmit chain can *do* rather than what the sequencer can *address*,
so they belong where the sequence is written, not where it is played.

## Diagnostics

Every refusal names the check, the location, and the margin:

```
gmax exceeded (|g| = 48.2 mT/m > 40.0 mT/m, axis=xy, block 1247)
slew exceeded (|dG/dt| = 214 T/m/s > 180 T/m/s, blocks 88 -> 89)
gradient discontinuity (x: 12.4 -> 0.0 mT/m, blocks 4 -> 5)
raster misalignment (adc delay 15 us, raster 10 us, block 33)
```

The block indices are indices into the file, so they can be looked at:

```python
ok, message = seq.check_hardware_limits()
if not ok:
    print(message)
    seq.plot(time_range=[0.0, 0.05])
```

## Where the checks run

Both sides, from the same code: `check_hardware_limits()` and
`check_timing()` in Python while the sequence is being written, and
`check_safety()` in the interpreter before download — see
{doc}`../../examples/c/safety_gate`. The cost is a walk over the instance
table with the waveform library resident, which is a fraction of the parse
that preceded it.
