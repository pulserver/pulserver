# Gradient amplitude, slew, continuity and timing

```{admonition} TL;DR
:class: tip

- **Amplitude is a vector limit.** 35 mT/m on x and 35 on y is 49 mT/m along
  the diagonal, and a 40 mT/m system refuses it. Root-sum-square across axes,
  after rotation.
- **Slew and continuity are one criterion**, $|\Delta G|/\Delta t \le$ `max_slew`,
  applied to two different sample pairs: consecutive samples *inside* a waveform,
  and the last sample of one waveform against the first of the next. A block
  boundary is not a boundary for the gradient.
- That is why they cost differently. Interior slew is the normalised shape's
  slew times an amplitude, so it is a **shape** question; the seam needs both
  endpoints at their own instance amplitude and rotation, so it is an
  **instance** question.
- **Timing** goes against the raster each event is *played* on, not the raster
  the file declares. A file laid out on a finer raster than the scanner's passes
  the raster comparison while holding times the hardware cannot address.
- All four run over the *definitions*, so they cost the number of distinct
  events rather than the number of blocks.
```

The four checks that need nothing but the sequence and the system, and therefore
always run. They are also the four that catch most real mistakes.

```{figure} ../assets/gradient_slew/vector_and_seam.png
(a) Every axis inside its own limit is not the same statement as the vector
inside the limit. (b) The same slew criterion broken in both places it can be
broken, drawn as the waveform and as the quantity actually checked. Block *n*
was designed against a system allowing 200 T/m/s and is played on one allowing
170, so its own ramp is already illegal — an **interior** failure, inside a
single waveform, with no block boundary near it. It then ends at 6 mT/m where
its neighbour starts at 0: the **seam** failure, the same inequality across one
raster tick, which puts it far off the scale. One dashed line judges both.
```

## Amplitude: a vector limit

A gradient amplifier limits each axis, and the *combination* of axes. A
sequence playing 35 mT/m on x and 35 mT/m on y is not playing 35 mT/m: it is
playing 49 mT/m in the direction $(1,1)/\sqrt2$, and a 40 mT/m system will
refuse it.

So the check is the root-sum-square across axes at every sample. The per-axis
test is the special case where the other axes are idle, which is why an oblique
or non-Cartesian sequence can fail a limit that every individual waveform
respects. The same applies after a rotation: a `ROTATIONS` extension mixes the
axes, so a shot legal unrotated can be illegal at some angle, and the check
evaluates the rotated waveform.

## Slew and continuity: one criterion, two sample pairs

Slew is $dG/dt$, checked the same way — root-sum-square, after rotation. A block
boundary is not a boundary for the gradient: blocks run back to back with no
gap, so a waveform ending at 20 mT/m followed by a block starting at 0 is a step
across one raster tick, and the hardware has to slew it like any other.

Continuity is not a second physical statement. It is the same inequality asked
about the pair of samples that straddles the seam, and the tolerance is written
that way — a step passes exactly when

$$|\Delta G| \;\le\; \texttt{max\_slew} \times \texttt{grad\_raster},$$

which is $|\Delta G|/\Delta t \le$ `max_slew` with the raster as $\Delta t$.
A "small" discontinuity is not a legal design error; it is a legal *slew*, and a
large one is refused by the same number that refuses a steep ramp.

What differs is the arithmetic each pair needs, and that is what makes them two
passes rather than one:

| | the pair | what it needs |
|---|---|---|
| **interior** | consecutive samples inside one waveform | the normalised shape's own slew, computed once per shape, times this instance's amplitude |
| **seam** | last sample of one waveform, first of the next | both endpoints, each scaled by *its own* instance amplitude and turned by *its own* rotation |

So the interior question is answered per **shape** and reused by every playout of
it, while the seam question can only be answered where two neighbours meet, at
the amplitudes they actually run — which is a walk over instances.
{doc}`../performance/gradient_checks` measures the difference that makes.

A mismatch is reported as a discontinuity, with the step in mT/m and the pair of
blocks, rather than as a slew number: the same violation, named the way that
tells the author what to fix.

```{note}
Trapezoids and simple arbitrary gradients start and end at zero, so the seam is
trivially satisfied for most Cartesian sequences. It earns its place on
continuous-gradient families — ZTE, spirals with bridges, anything where the
readout does not return to zero between blocks — and on a sequence whose
rewinders were re-solved per rotation angle rather than materialised from a base
waveform.
```

## The step limit

Some systems additionally limit how much the gradient may change *between
successive playouts of a segment* — a view-to-view step, which is what a
rotating non-Cartesian trajectory does at its readout start. Where such a
limit is configured, it is checked over the instance table rather than over
the waveform: a trajectory whose successive views start further apart than the
amplifier tolerates is refused with the pair of views that did it.

## Timing: every event on an addressable grid

A sequencer starts and stops events on a raster, and a time that is not an
integer multiple of it cannot be played.

```{figure} ../assets/gradient_slew/raster_alignment.png
(a) An event may start where the raster does; a 14 µs start on a 4 µs grid is
not a rounding question, because no instruction begins there. (b) The case the
check exists for: a time that is a multiple of the raster the *file* declares
and not of the one it will be *played* on.
```

The rasters a file declares are not the answer. The file's rasters are compared
with the scanner's when the collection is built, and that comparison accepts
either direction — a sequence laid out on a *finer* raster passes it while still
holding times the hardware cannot address. A 15 µs event on a declared 5 µs grid
is legal by that test and unplayable on a 10 µs one, which is the case this
check exists for.

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

The two sides check slightly different sets, deliberately. `check_timing()` at
design time also judges RF dead time and ringdown, ADC dead time and soft-delay
agreement, reporting them entry for entry as upstream PyPulseq does. The
interpreter's pass drops those three: they bound what a transmit chain can *do*
rather than what the sequencer can *address*.

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

Both sides, from the same code: `check_hardware_limits()` and `check_timing()`
in Python while the sequence is being written, `check_safety()` in the
interpreter before download — see {doc}`../../examples/c/safety_gate`.
