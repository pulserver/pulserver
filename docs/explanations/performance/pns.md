# Peripheral nerve stimulation

A gradient that switches fast enough depolarises peripheral nerve membranes.
{doc}`The safety page <../safety/pns>` sets out the physics and fixes the
quantity to measure: the slew rate of every gradient axis, run through the
coil's nerve model, combined across axes and taken at its peak,

$$r(t) \;=\; \bigl\lVert\, (k * \dot G)(t) \,\bigr\rVert_2 ,
\qquad \text{verdict} \;=\; \max_t r(t),$$

as a percentage of the model's own threshold.

Unlike {doc}`mechanical resonance <mechanical_resonance>`, **nothing here has
to be calibrated**. A forbidden band states a frequency range and usually no
amplitude, so what counts as too much had to be measured against a corpus. A
nerve model states its own threshold: the response *is* a percentage of it,
100 % is the limit, and 80 % is the margin a console leaves. There is no
policy to derive and no gap to place a number in.

That leaves one thing standing between the definition and a verdict.

**How to compute it in the time predownload can spend.** $\dot G$ is the slew
of the whole scan — minutes of waveform on a microsecond raster — and $k$ is
tens of chronaxie constants of history, 721 samples on the raster these
figures are drawn on. Written directly that is $3NK$ multiply-adds over a
scan that grows with the prescription, and it has to hold for every
repetition, not just the first one.

This page is that made affordable. Everything on it is the safety page's rule
evaluated in a cheaper order, and every shortcut is drawn against the
calculation it replaces. The model throughout is **Irnich
rheobase/chronaxie** — the one the scanner-side gate runs, and the only one of
the two that publishes a kernel at all.

---

## 1. What the analysis is given

A scan is $M$ repetitions of one structural TR — the smallest block period
over which the {doc}`normalized structure repeats
<../sequence_model/tr_and_segmentation>`, derived from the block content
rather than annotated.

One TR is an ordered list of **base block** ids. A base block is the
definition-level identity of a block: the tuple of event *definitions* it
plays and its duration, so two blocks share one whenever they play the same
definitions, whatever amplitudes they play them at ({doc}`the PulSeg split
<../background/pulseg>`). An event definition is the part of an event that is
fixed for the whole scan — its structural properties, with the per-playout
numbers held separately.

For the gradients, that leaves a TR as an ordered list of $(g_x, g_y, g_z)$
definition tuples, where each entry is one of two things:

| | identified by | plus, per playout |
|---|---|---|
| trapezoid | delay, rise, flat, fall | amplitude, rotation |
| arbitrary | delay, time shape, sample count | waveform shape, amplitude, rotation |

That table is the whole cost model. Position in the TR fixes a
start time, so a shift handles *when* a block plays; a definition fixes a
shape, so one convolution serves every playout of it; and amplitude is a
per-playout number that scales a response already computed. What is left to
actually convolve is the set of distinct shapes, which for a Cartesian
sequence is a handful and never grows with the matrix.

Note where the waveform shape sits in the arbitrary row. Two spiral arms
written out shot by shot have the *same* definition — same delay, same
sampling, same length — and different shapes. They are one entry in the
sequence's definition library and two distinct shapes to this analysis, which
keys on the shape as well.

---

## 2. The canonical TR

The analysis runs on a single window and the verdict is taken to hold for the
scan. That is sound only if no repetition can stimulate harder than the
window — and repetitions are not all the same waveform. Position by position,
three situations arise:

**A — nothing varies.** Excitation, slice select, spoiler, an unrotated
readout: the same definition at the same amplitude in every repetition. The
window plays exactly what the scan plays.

**B — an amplitude varies.** A phase encode steps; a partition moves. One
shape, several amplitudes, and the window takes the largest of them at that
position. This is sound because the same shape driven harder is a larger
response on that axis at every instant, and the combination across axes only
grows with each of them.

(pns-shape-groups)=
**C — the waveform varies.** A multishot readout written out shot by shot:
different shapes at the same position. There is no bound to take here. No
amplitude makes one spiral arm's shape cover another's, and the instant of the
peak is a property of the whole window rather than of the position, so a
position-by-position maximum would not be a maximum of anything. The
repetitions are **grouped by the definitions they play** instead: one window
is built per group, at that group's own shapes and its own amplitude maxima,
and the worst of the groups is the verdict.

A sequence whose repetitions differ only in amplitude has one group, and the
check is the single envelope evaluation it has always been. A four-arm spiral
written out as its own waveforms has four. Past
`PULSEG__MAX_SHAPE_GROUPS` (64) the sweep **fails closed** rather than
guessing — the diagnostic tells the author to write the repeated waveform
once and turn it with a `ROTATIONS` extension, which collapses the groups back
to one.

The figure switches each kind of variation on alone, on the same controlled 3D
stack-of-spirals scan {doc}`the acoustic page <mechanical_resonance>` uses:
one shipped repetition played four times, same excitation, same spiral, same
partition encode, same 10 ms period, on the weaker of the two gradient
systems. Four repetitions rather than sixteen so the scan can be drawn
straight through, which is how the check sees it — the response of one
repetition running into the next.

```{figure} ../assets/pns_performance/canonical_tr.png
Four repetitions played straight through, with each kind of variation
switched on alone.
```

The reference is the **whole scan**: every repetition as it actually plays,
concatenated, differentiated and convolved in one pass, with no knowledge that
the scan has a period. Both sides are pulled through the same extraction, so
the comparison is of the analysis and not of two renderers.

The four panels are four genuinely different scans — the repetitions'
gradients differ by up to 16 mT/m when the encode sweeps and 28 mT/m when the
arms do — and they return the same verdict, with the window's peak landing
*exactly* on the worst repetition's:

| | repetitions differ by | judged | worst repetition |
|---|---:|---:|---:|
| A nothing varies | — | 95.611 % | 95.611 % |
| B the encode amplitude varies | 16 mT/m | 95.616 % | 95.616 % |
| C the readout waveform varies | 28 mT/m | 95.611 % | 95.611 % |
| D both vary | 28 mT/m | 95.616 % | 95.616 % |

That the window is not merely a bound here but an equality is worth taking
apart, because each case gets there for its own reason.

**A** is trivial: nothing varies, so the window *is* the repetition.

**B** is the per-position amplitude maximum meeting a scan that plays it. The
window takes the largest amplitude each position reaches; a real encode table
sweeps to its own extreme, so some repetition plays that maximum and the
window coincides with it. The other three repetitions come in fractionally
lower — 95.598 % at the smallest encode — which is the whole visible effect of
a 16 mT/m swing.

**C and D reproduce A and B exactly.** This is the one that is not a
coincidence of the prescription. The chronaxie model applies *one* kernel to
every axis, so it commutes with the rotation that turns one arm into the next,
and the verdict is a root-sum-square, which a rotation leaves alone. Turning a
spiral moves stimulation between $G_x$ and $G_y$; it does not change how much
of it there is. Every shape group returns the same number, so sweeping them
costs time and changes nothing.

Which is also why the same scan gives the same answer whichever way its arms
reach the scanner:

```{figure} ../assets/pns_performance/multishot_envelope.png
Spiral gradient echo: four interleaves, and the window built over each.
```

Four arms at one block position, the window each group is judged by, and every
repetition's stimulation against the worst window's. The peak is
122.2063 %, and it is the same 122.2063 % whether the arms arrive as four
written-out waveforms or as one arm plus a `ROTATIONS` extension — with the
repetitions agreeing with the window to $1.2\times10^{-5}$ % of threshold.

**Where the window would be strictly conservative**, and this scan is not, is
when two positions vary *independently* and no repetition plays both extremes
at once. The maximum is taken per position, so the window then assembles a
repetition nobody plays. A single varying position, or a table that sweeps its
own corners, gives back the equality above.

A model with per-axis coefficients has no such symmetry to lean on. SAFE
carries a separate coefficient set per axis, and a rotation extension is
applied below this analysis rather than inside it, so for that model a rotated
arm is not what the check sees. The grouping is the honest route there, and it
is the route taken.

---

## 3. One response per distinct waveform

The window is read once, and what is read is the gradient columns of the block
table rather than any rendered waveform.

```{figure} ../assets/pns_performance/memoization.png
The block table's gradient columns, gathered by position across the TR
instances and deduplicated on the `(gx, gy, gz)` tuple each position plays.
A position that takes one tuple varies only in amplitude; a position that
takes several is what forces the instances to be split into groups.
```

**The tuple, not the axis, is the unit of the walk.** A position is the same
position in two instances when all three of its gradient columns match, so
what is counted per position is the number of distinct `(gx, gy, gz)` tuples,
and a column is a pair: the gradient definition, which fixes the timing
skeleton, and the shape id, which fixes the samples. A phase encode changes an
amplitude and leaves the tuple alone. A written-out spiral arm changes two of
the three shape ids and gives the position a second tuple.

**Positions with one tuple** are bounded by their largest amplitude — cases A
and B of the previous section.

**Positions with several** — case C — send the instances themselves to be
{ref}`classified <pns-shape-groups>`: an instance's signature is the whole
sequence of its positions' tuples, instances sharing a signature form a group,
and one window is built per group.

**Inside a window, the convolution is memoized per axis.** Each block's slice
is keyed on (gradient definition · shape id · block duration) — the identity
the representation already carries. Nothing is re-derived to decide that two
blocks play the same thing, and nothing is compared sample by sample. Each
distinct key is convolved with the model kernel once; every occurrence takes
the stored response out of the table, scales it by its amplitude ratio, shifts
it to its start time, and adds into an accumulator. The three axes are
combined and the peak read off at the end.

That identity does not carry **rotation**, and the slices are cut from the
materialised, already-rotated waveform. What keeps that sound is a
precondition rather than a comparison: a slice is only ever matched against
the logical gradient of the same axis index, and if any physical axis carries
a slice with no logical gradient of its own — which is exactly what a rotation
leaking one logical axis onto three physical ones produces — the assembly
declines and the whole window takes the exact convolution instead. Builds
configured with `PULSEG_DEBUG_MEMO` additionally compare the samples of every
accepted match against the template it was matched to.

Compare that with the alternative — materialise the slew over the whole scan
and convolve it:

```{figure} ../assets/pns_performance/assembly_cost.png
What the stimulation check's cost actually depends on.
```

**Scan length: gone.** A convolution of the timeline goes from 5.8 ms to
182 ms between 3 and 144 repetitions of one EPI shot, as a convolution of the
scan must. The window stays at 1.5–1.9 ms throughout. Fourteen thousand blocks
is a small protocol; a clinical prescription is two orders of magnitude
longer, and the left-hand column keeps growing for as long as the scan does.

**Window length: traded for shape count.** Convolving one window whole costs
$3NK$; assembling it costs one convolution per distinct shape plus two placed
adds per occurrence. Across five families:

| | window | distinct shapes | occurrences | convolved whole | assembled | |
|---|---:|---:|---:|---:|---:|---:|
| spiral GRE | 7.9 ms | 7 | 7 | 1.7 M | 0.97 M | 1.8× |
| 2D GRE | 15 ms | 7 | 7 | 3.3 M | 1.03 M | 3.1× |
| EPI | 51 ms | 10 | 134 | 11.0 M | 0.96 M | 11.5× |
| FSE | 2.0 s | 8 | 51 | 433 M | 1.25 M | 347× |
| MPRAGE | 2.0 s | 9 | 501 | 433 M | 1.64 M | 263× |

A short window with few repeats buys almost nothing; a two-second inversion
train buys two orders of magnitude, because its length grew and its shape count
did not. The assembly is taken only where it is expected to win by at least
4×, and only on windows over 512 samples — below that the simpler exact route
is kept.

---

## 4. Computing one response

A nerve model that publishes a kernel is asserting that it is a linear filter:

$$r(t) = \bigl(k * \dot G\bigr)(t).$$

A filter does not care when its input arrives. In the frequency domain it is a
multiplication, which distributes over a sum, and a delay is a phase factor
that comes back out unchanged — which in the time domain is the statement that
convolution is linear and shift-invariant. The window is already written as a
sum of delayed shapes, because that is what a block list is:

$$G(t) = \sum_i a_i\, g_{c(i)}(t - t_i),$$

block $i$ playing shape $g_{c(i)}$ at amplitude $a_i$ from time $t_i$.
Differentiation and convolution both distribute over that sum, so

$$r_\text{ax}(t) = \sum_i a_i \,\bigl(k * \dot g_{c(i)}\bigr)(t - t_i).$$

Each *distinct* shape is convolved once and stored; every occurrence of it
becomes a scaled, shifted add of the stored result. Nothing has been
truncated, resampled or approximated on the way — the two expressions are the
same sum, regrouped.

The convolution itself is done in the **time domain**, deliberately: a
template is one block long against a kernel spanning tens of chronaxie times,
so a transform of the padded template would be mostly padding, and a direct
linear convolution has no wraparound to guard against in the first place.

Two details make the algebra true of the implementation and not only of the
mathematics. **At the seams**, each block's slice contributes its slew
zero-extended on both sides: it opens with the step up from zero into the
block and closes with the step back down out of it, so where two blocks meet,
the closing step of one and the opening step of the next add to exactly the
difference a forward difference across the seam would have subtracted — the
same two floats, the same subtraction. **Around the window**, the
occurrences are placed again one window later, and again for as many windows
as the padding spans, which is the same warmed-up history that wrapping the
waveform round gives the direct route. How far that padding reaches is set by
the model's memory and not by the sequence, so a repetition shorter than that
memory is replayed more than once.

**Then the axes combine.** Per-axis responses are percentages of the same
threshold, and they are combined by root-sum-square at every instant before
the peak is taken. This is the one step that does *not* decompose: a
root-sum-square of sums is not a sum of root-sum-squares, so the per-shape
assembly runs per axis and the combination happens once, at the end, over the
assembled traces.

Held against the direct route — pad, differentiate, convolve the whole window,
written straight from the published kernel in double precision:

```{figure} ../assets/pns_performance/assembly_equivalence.png
EPI, one canonical TR: assembled per shape against convolved whole.
```

One EPI window: 1601 samples, 25 blocks, 45 slices, 11 distinct shapes,
5 020 323 multiply-adds convolved whole against 658 847 assembled. All three
routes report the same peak to seven digits — **124.4607 %** convolved whole,
assembled, and returned by the library. The bottom panel separates the two
error terms: regrouping the sum in double precision moves the answer by
$4\times10^{-14}$ % of threshold, which is the algebra being exact, and the
library sits $2\times10^{-5}$ % away, which is the float32 it computes in.

The fast route exists only where there is a kernel to publish. **SAFE** is not
a convolution — its three branches rectify on both sides of their lowpass, so
the model is nonlinear and no impulse response describes it. It takes the
direct route always, over the same canonical window and the same groups.

---

## 5. The verdict

One number: the peak of the combined response over the window, against the
model's threshold.

```{figure} ../assets/pns_performance/epi_verdict.png
An echo train's stimulation against the 80 % margin and the 100 % threshold.
```

A single-shot echo train is the instructive case, and not for the reason it
looks like. The train shows up as a run of near-identical ~50 % teeth, one per
gradient reversal — and they do not stack, because the echo spacing is longer
than the 360 µs chronaxie, so each response has decayed before the next edge
arrives. The verdict, **112 %**, is set instead by the slice-select rewinder
at 3.3 ms: one large excursion on a single axis, nowhere near the readout.

Which is why the check evaluates a whole window with its history rather than
scoring events one at a time. Shorten the echo spacing, or move to a coil with
a longer chronaxie, and those teeth start landing on each other's tails; at
that point the train sets the verdict and nothing about any single reversal
predicts it. Whether a repetitive train accumulates is a property of the
interval between its edges against the nerve's own time constant, and neither
is visible in the events alone.

```python
ok, norm, components, t = seq.calculate_pns(
    {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333}, tr="worst_case",
)
ok            # the verdict the predownload gate will reach
norm.max()    # the peak, as a fraction of threshold
```

---

## Equivalence tests

Each shortcut is a claim that two calculations agree, and each has a test that
computes both:

| | held against | |
|---|---|---|
| the assembled response | the same window convolved whole, from the published kernel in double precision | the same sum regrouped: $4\times10^{-14}$ % of threshold in double, $2\times10^{-5}$ % as the library computes it |
| the identity two occurrences are matched on | the materialised samples | asserted sample by sample in `PULSEG_DEBUG_MEMO` builds; in a shipped build, a slice with no gradient of its own on that axis sends the whole window to the exact route instead |
| the assembled response, sample by sample | the same window convolved whole | every sample of all three axes, the wrapped history included, over every fixture the assembly is taken on |
| the canonical window | every repetition it stands for | the peak of the window is at least the peak of any repetition, over all four kinds of variation — and on the family drawn above, exactly equal to it |
| one window per shape group | the encoding that needs none | the same spiral as four written-out arms and as one arm turned by a rotation: 122.2063 % either way |
| a one-repetition sequence | the plain convolution | the same waveform under the same model, evaluated two ways |
| the wrapped history | the scan played back to back | the window's peak is the steady-state peak, boundaries included |

**End to end.** This is a predownload cost, not a UI one: `validate_protocol`
returns before any gradient exists to differentiate. What the interpreter pays
once the finished file comes back in — gradient continuity and slew, this
check, and the acoustic analysis together — is what the {doc}`full benchmark
<full_benchmark>` reports as *Safety*.

None of this machinery is allowed to be an approximation of the rule on the
{doc}`safety page <../safety/pns>`. It is the same rule in a cheaper order,
and the tests above are what holds it there.
