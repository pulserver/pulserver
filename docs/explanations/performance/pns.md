# Peripheral nerve stimulation

```{admonition} TL;DR
:class: tip

- **What is judged.** The slew rate of every gradient axis through the coil's
  nerve model, combined across axes by root-sum-square, at its peak — as a
  percentage of the model's own threshold. The rule is
  {doc}`the safety page's <../safety/pns>`; nothing is calibrated.
- **What is *not* evaluated.** The scan. A scan is thousands of repetitions
  of one structural TR, and the repetitions differ — in amplitude, in
  rotation, in waveform. The check evaluates **one built repetition per
  group of repetitions that play the same waveforms**, the *positional-
  maximum envelope*, and the worst group's peak is the verdict.
- **What that guarantees.** The envelope drives at least as hard as every
  repetition it stands for at every sample, exactly. Its *peak* is held to
  be at least every repetition's by test: on the corpus it lands on the worst
  repetition's to seven digits. It is conservative only where two positions
  vary independently.
- **Where it is blind.** SAFE has a coefficient set per axis, so a rotation
  changes its answer; the window carries one instance's rotation, and the
  prescription rotation is composed in only when the caller supplies it,
  which the interpreter does at predownload. The Irnich model is immune
  (one kernel, RSS).
- **Past 64 groups there is no window, and no bound either.** Every block's
  own exact response is placed on the scan's timeline and the peak of the
  combined response is read off: the same quantity the scan convolved whole
  gives, chunk by chunk on every core. 131 072 distinct arms gate in seconds.
```

{doc}`The safety page <../safety/pns>` fixes the quantity: the slew rate of
every gradient axis, run through the coil's nerve model, combined across axes
and taken at its peak,

$$r(t) \;=\; \bigl\lVert\, (k * \dot G)(t) \,\bigr\rVert_2 ,
\qquad \text{verdict} \;=\; \max_t r(t).$$

Unlike {doc}`mechanical resonance <mechanical_resonance>`, nothing here has to
be calibrated. A nerve model states its own threshold: the response *is* a
percentage of it, 100 % is the limit, 80 % is the margin. The model
throughout is **Irnich rheobase/chronaxie**, the one the scanner-side gate
runs and the only one of the two that publishes a kernel; SAFE is taken up
where it differs.

What stands between the rule and a verdict is that $\dot G$ is minutes of
waveform on a microsecond raster, $k$ is tens of chronaxie constants of
history — 721 samples on the raster these figures are drawn on — and the
scan is not one waveform but thousands of repetitions that are almost, and
not quite, the same. This page is about that *almost*: how one repetition is
built to stand for all of them, what that construction guarantees, where it
is conservative, where it is blind, and what it costs.

---

## 1. A scan is one TR, repeated — approximately

The interpreter detects the **structural TR** from block content alone: the
shortest run of base blocks after which the block sequence repeats, with
every position matching in duration ({doc}`TR and segmentation
<../sequence_model/tr_and_segmentation>`). So every repetition has the same
timing skeleton, position by position. What a position plays across the
repetitions is one of three things:

| | at that position, across repetitions | example |
|---|---|---|
| **A** | nothing varies | excitation, slice select, spoiler, an unrotated readout |
| **B** | the **amplitude** varies, or the **rotation** | a phase encode; a radial spoke turned by a `ROTATIONS` extension |
| **C** | the **waveform itself** varies | a multishot readout written out arm by arm; a SPARKLING-style acquisition |

Amplitude is a low-frequency modulation of the same shape and enters every
check the same way. Rotation and shape do not: a rotation moves drive between
axes without changing the vector, a new shape changes the drive itself. Which
of these a check can see through depends on its criterion:

| the criterion | amplitude (B) | rotation (B) | shape (C) |
|---|---|---|---|
| Irnich: one kernel per axis, root-sum-square, peak | bounded by the per-position maximum | **invariant** — the kernel commutes with a rotation and the RSS is unchanged by it | needs one window per distinct set of shapes |
| SAFE: per-axis coefficients, rectifying stages, peak | bounded by the per-position maximum | **not invariant** — drive moved onto a stiffer axis reads higher | needs one window per distinct set of shapes |

The rest of the page is the construction that follows from that table.

---

## 2. The canonical TR: the positional-maximum envelope, one per shape group

The construction is {doc}`construction 1 of the canonical-TR page
<../safety/canonical_tr>`. Step by step, from the block table and nothing
rendered:

1. **Label every repetition by the waveforms it plays.** A repetition's
   signature is, per position, the base block id, the block duration, and
   the three gradient *definition* ids together with the three gradient
   *shape* ids — the shape id is what tells two spiral arms apart when they
   share a definition (same delay, same sampling, same length). Amplitude
   and rotation are deliberately absent from the signature. Repetitions with
   the same signature form a **shape group**. A Cartesian scan has one
   group; a four-arm spiral written out has four; a scan with a distinct arm
   in every repetition has one group per repetition, which is the case
   section 5 is for.
2. **Per group, take the positional maximum.** At each position, over the
   repetitions of the group, the amplitude of largest magnitude, its sign
   kept. This is read off the amplitude table; no waveform is touched.
3. **Render one repetition per group** with each position at that amplitude
   and the group's own shapes. Rotation is that of the group's representative
   repetition: the window is one instance's rotation, not a bound over
   rotations.
4. **Evaluate the window** as if the scan were that repetition played
   without end — the occurrences are placed again one window later for as
   many windows as the kernel's memory spans, which is the warmed-up history
   an infinite repetition has — and read the peak of the RSS response.
5. **The worst group's peak is the verdict.**

The construction does not produce the verdict; it produces a waveform, which
is then evaluated by the same convolution any single repetition would be.

**What the envelope guarantees, exactly.** At every sample of every axis the
envelope's magnitude is at least the magnitude any repetition of the group
plays there: the position's shape is the same and $\lvert a_\max\rvert \ge
\lvert a_k\rvert$. That is pointwise dominance, and it is asserted directly
(`tests/ctests/test_acoustic_window.c` renders every repetition of seven
shipped families and compares).

**What it does not prove, and how it is held.** The verdict is the *peak* of
a signed sum through a kernel, and a pointwise-larger drive can produce a
smaller peak where terms that reinforced now cancel. So "the window's peak
bounds every repetition's" is a statement verified by test rather than
proved — over all four kinds of variation on the corpus, and exactly on the
family below. This is also why mechanical resonance takes a different
construction: a magnitude at one frequency *can* be bounded by a sum of
magnitudes, and a peak in time cannot.

**Where it is conservative.** The maximum is taken per position. When two
positions vary independently and no repetition plays both extremes at once —
a phase encode and an independent partition encode — the window assembles a
repetition nobody plays, and the verdict is higher than any repetition's.
That is a false positive by construction, never a false negative.

**Where it is blind.** For SAFE, step 3 is a gap: a rotated shot in the same
shape group is seen at the representative's rotation. The prescription
rotation (an oblique slice) is composed into every rotation the check sees
when the caller supplies it, which the interpreter does at predownload; a
design-side call runs in the design frame unless given the matrix. The Irnich
model the scanner gate runs does not care; SAFE does, and the hardware
monitor is what stands behind it. The same page-2 table says why: SAFE's per-axis
coefficients are exactly what a rotation moves drive across.

### How tight it is

The figure switches each kind of variation on alone, on a controlled 3D
stack-of-spirals scan — one shipped repetition played four times, same
excitation, same spiral, same partition encode, same 10 ms period. The
reference is the **whole scan**: every repetition as it plays, concatenated,
differentiated and convolved in one pass with no knowledge that the scan has
a period. Both sides run through the same extraction.

```{figure} ../assets/pns_performance/canonical_tr.png
Four repetitions played straight through, with each kind of variation switched
on alone.
```

| | repetitions differ by | judged | worst repetition |
|---|---:|---:|---:|
| A nothing varies | — | 95.611 % | 95.611 % |
| B the encode amplitude varies | 16 mT/m | 95.616 % | 95.616 % |
| C the readout waveform varies | 28 mT/m | 95.611 % | 95.611 % |
| D both vary | 28 mT/m | 95.616 % | 95.616 % |

**A** is trivial: nothing varies, so the window *is* the repetition. **B** is
the per-position amplitude maximum meeting a scan that plays it — a real
encode table sweeps to its own extreme, so some repetition plays that
maximum; the others come in fractionally lower, 95.598 % at the smallest
encode, which is the whole visible effect of a 16 mT/m swing. **C and D
reproduce A and B exactly**, and that is structural: the Irnich kernel
commutes with the rotation that turns one arm into the next, and the RSS is
invariant, so turning a spiral moves stimulation between $G_x$ and $G_y$
without changing how much of it there is.

```{figure} ../assets/pns_performance/multishot_envelope.png
Spiral gradient echo: four interleaves, and the window built over each.
```

The peak is 122.2063 %, and it is the same 122.2063 % whether the arms arrive
as four written-out waveforms (four shape groups) or as one arm plus a
`ROTATIONS` extension (one group) — the repetitions agreeing with the window
to $1.2\times10^{-5}$ % of threshold.

---

## 3. Evaluating a window: one convolution per distinct waveform

A window is read from the gradient columns of the block table, never from a
rendered scan. Each position names a definition, a shape, an amplitude and a
start time, so the window is already a sum of delayed, scaled shapes, and
convolution is linear and shift-invariant:

$$G(t) = \sum_i a_i\, g_{c(i)}(t - t_i)
\qquad\Longrightarrow\qquad
r_\text{ax}(t) = \sum_i a_i \,\bigl(k * \dot g_{c(i)}\bigr)(t - t_i).$$

Each distinct shape is convolved once; every occurrence is a scaled, shifted
add. Nothing is truncated, resampled or approximated — the two expressions
are the same sum, regrouped.

```{figure} ../assets/pns_performance/memoization.png
The block table's gradient columns, gathered by position across the TR
instances and deduplicated on the `(gx, gy, gz)` tuple each position plays.
```

**The identity a slice is keyed on** is (gradient definition · shape id ·
block duration) — what the representation already carries, so nothing is
compared sample by sample. That identity does not carry rotation, and the
slices are cut from the already-rotated waveform; what keeps that sound is a
precondition: a slice is only matched against the logical gradient of the
same axis index, and if any physical axis carries a slice with no logical
gradient of its own — what a rotation leaking one logical axis onto three
physical ones produces — the assembly declines and the window takes the exact
convolution instead. `PULSEG_DEBUG_MEMO` builds additionally compare the
samples of every accepted match against its template.

**At the seams**, each block's slice contributes its slew zero-extended on
both sides, so where two blocks meet the closing step of one and the opening
step of the next add to exactly the forward difference across the seam.
**Around the window**, the occurrences are placed again one window later, as
many times as the kernel's memory reaches — the warmed-up history of an
infinite repetition, so a repetition shorter than that memory is replayed
more than once.

**Then the axes combine.** Per-axis responses are percentages of the same
threshold and are combined by root-sum-square at every instant before the
peak is taken. This is the one step that does not decompose, so the assembly
runs per axis and the combination happens once.

### How one response is computed

Whichever path a scan takes, a response is the same four steps on one
gradient waveform:

1. **Render on a uniform raster.** A gradient is stored as corner points — a
   trapezoid's four, an extended trapezoid's vertices at their own times, an
   arbitrary waveform's samples at the centres of the gradient raster's
   cells. The corner points are linearly interpolated onto a uniform raster,
   which is where the slew is defined: the window path at half the gradient
   raster, the exact scan at the gradient raster. The two agree to the FFT
   allowance below because the kernel is bin-integrated, which is what the
   raster-halving invariant test holds.
2. **Differentiate.** The slew is the forward difference of the samples,
   divided by γ and the raster, in T/m/s. The step a block makes at its
   start — its first sample against the previous block's last — is one
   slew sample like any other; the window path takes it from the rendered
   scan, the exact path prices it per block as one kernel tap of that size,
   so a gradient that runs on across blocks is charged its slew and not a
   fictitious step at every seam.
3. **Convolve with the model's kernel.** The Irnich model publishes its
   kernel — the chronaxie response, bin-integrated onto the raster, tens of
   chronaxie constants long. Inside a window the convolution is done in the
   time domain, one template per distinct shape: a template is one block
   long against a kernel that spans tens of chronaxie times, so a transform
   of the padded template would be mostly padding, and a direct linear
   convolution has no wraparound to guard against. On the exact scan the
   convolution is done by FFT, one forward and one inverse real transform
   per block per axis, zero-padded to the block plus the kernel's memory so
   nothing wraps; the single-precision transform is allowed
   $2\times10^{-3}$ of the peak. SAFE publishes no kernel — its three
   branches rectify between their lowpass stages — so it is never convolved:
   the rendered slew is handed to the model's own evaluator.
4. **Combine and take the peak.** Root-sum-square across the three physical
   axes at every instant — a `ROTATIONS` extension turns each block's
   response into the physical frame first — and the maximum over time.

```{figure} ../assets/pns_performance/assembly_equivalence.png
EPI, one canonical TR: assembled per shape against convolved whole.
```

One EPI window: 1601 samples, 25 blocks, 45 slices, 11 distinct shapes,
5 020 323 multiply-adds convolved whole against 658 847 assembled. All three
routes report **124.4607 %** to seven digits; regrouping the sum in double
precision moves the answer by $4\times10^{-14}$ % of threshold, and the
library sits $2\times10^{-5}$ % away, which is the float32 it computes in.

**SAFE** is not a convolution — its three branches rectify on both sides of
their lowpass — so it takes the direct route always, over the same window and
the same groups.

### What the cost depends on

```{figure} ../assets/pns_performance/assembly_cost.png
What the stimulation check's cost actually depends on.
```

**Scan length: gone.** A convolution of the timeline goes from 5.8 ms to
182 ms between 3 and 144 repetitions of one EPI shot. The window stays at
1.5–1.9 ms throughout, and a clinical prescription is two orders of magnitude
longer than the 14 000 blocks measured here.

**Window length: traded for shape count.** Convolving one window whole costs
$3NK$; assembling it costs one convolution per distinct shape plus two placed
adds per occurrence.

| | window | distinct shapes | occurrences | convolved whole | assembled | |
|---|---:|---:|---:|---:|---:|---:|
| spiral GRE | 7.9 ms | 7 | 7 | 1.7 M | 0.97 M | 1.8× |
| 2D GRE | 15 ms | 7 | 7 | 3.3 M | 1.03 M | 3.1× |
| EPI | 51 ms | 10 | 134 | 11.0 M | 0.96 M | 11.5× |
| FSE | 2.0 s | 8 | 51 | 433 M | 1.25 M | 347× |
| MPRAGE | 2.0 s | 9 | 501 | 433 M | 1.64 M | 263× |

The assembly is taken only where it is expected to win by at least 4×, and
only on windows over 512 samples. **Shape groups: linear.** Each group is one
window; a four-arm spiral costs four windows.

---

## 4. The verdict

```{figure} ../assets/pns_performance/epi_verdict.png
An echo train's stimulation against the 80 % margin and the 100 % threshold.
```

A single-shot echo train is the instructive case. The train shows up as ~50 %
teeth, one per gradient reversal, and they do not stack: the echo spacing is
longer than the 360 µs chronaxie. The verdict, **112 %**, is set instead by
the slice-select rewinder at 3.3 ms — one large excursion on a single axis,
nowhere near the readout.

```python
ok, norm, components, t = seq.calculate_pns(
    {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333}, tr="worst_case",
)
ok            # the verdict the predownload gate will reach
norm.max()    # the peak, as a fraction of threshold
```

`tr="worst_case"` is the worst shape group's window; `tr=k` is repetition
`k` as it plays; `tr=None` is upstream PyPulseq's calculation over the
timeline, to the bit.

---

## 5. Past the group cap: the exact scan

A SPARKLING-style acquisition plays a distinct optimised arm in every
readout. Section 2's grouping gives one group per repetition — there is no
envelope for that, since no amplitude makes one arm's shape cover another's.
Past `PULSEG__MAX_SHAPE_GROUPS` (64) the check leaves windows behind and
evaluates the scan as what it is. The nerve model is linear, so the
response of the scan is the sum of the responses of its blocks placed at
their start times,

$$R(t) = \sum_i r_i(t - t_i), \qquad r_i = k * \dot G_i ,$$

and that sum is formed, not bounded. The scan's timeline is cut into chunks
of consecutive blocks; each chunk accumulates, per physical axis, the
responses of its own blocks and of the earlier blocks still ringing into it
(the kernel's memory), reads the peak of the root-sum-square over its own
span, and the largest chunk peak is the scan's. Chunks are independent, so
they run on every core the host offers (`pulseg_opts.parallel_for_fn`, or
the library's own POSIX hook where it was built with one), and the result
does not depend on how they were dealt out.

**Each block's response is exact for the block on its own** — section 3's
four steps: rendered on the raster, sliced to its interior slew, convolved
with the kernel by FFT, its opening step against the previous block's end
priced as one kernel tap, rotated into the physical frame. The scan's own
closing step lands on its last block. The peak carries the single-precision
FFT allowance ($2\times10^{-3}$) and nothing else: no bound, no envelope,
no stretch to re-evaluate. Against the whole scan rendered and convolved by
the model's own evaluator it agrees to that allowance on every corpus
fixture and on the written-out-arms fixture (the equivalence table below).

**A model without a kernel** (SAFE) is not a convolution. For it the same
chunks are rendered — opened enough blocks early to carry a warmed-up
history — and handed to the model's evaluator, judged over the chunk's own
span; that route runs sequentially, because the rendering goes through the
check plan the caller holds.

**A repetition's own curve.** `calculate_pns(hw, tr=k)` past the cap is
repetition `k` played as it stands, and `tr="worst_case"` is the repetition
holding the scan's peak, evaluated exactly — a witness, not an envelope,
which the diagnostic says. On the written-out ladder every rung agrees with
its `ROTATIONS` twin on both sides of the threshold.

## 6. Cases and questions

**Between one and 64 shape groups.** Each group gets its own envelope
window and is evaluated alone, warmed up by its own previous copy; the
verdict is the worst group's peak. The order in which groups follow each
other in the real scan is not modelled: the tail of one group's repetition
ringing into the start of another's is replaced by the group's own tail.
The kernel's memory is a few milliseconds against repetitions of at least
that, and the corpus comparison against the whole scan holds it; it is a
tested statement, not a proved one.

**Why the exact scan is not used everywhere.** Cost only. The exact path
prices every block: 0.56 s at 8192 blocks and about a minute on a
million-block MPRAGE, where the envelope reads one repetition per group in
milliseconds and agrees with the whole scan to a fraction of a percent on
the family above. Past the cap there is no envelope to build, so the exact
path is the only one.

**One repetition, or the same events as one long repetition.** With a single
structural repetition the envelope *is* the scan and the reading is exact up
to the warm-up, which places the scan once more before itself so its end
rings into its start. Represented as $N$ repetitions the same events give
the positional-maximum envelope, which reads the same or more.

**Three representative scans.**

- *EPI.* One shape group; the envelope carries the largest phase-encode blip
  at every blip position, which some repetition really plays, and the
  readout train unchanged.
- *Stack of spirals.* One shape group whether the interleaves arrive as one
  arm plus `ROTATIONS` or written out; the partition encode enters at its
  largest amplitude. The Irnich model is frame-free, so turning the arm moves
  stimulation between axes without changing its size, and the four
  written-out arms read the same 122.2063 % as the rotated one.
- *Stack of SPARKLING.* One group per repetition, so the exact scan: every
  block's own response, the fixed excitation, slab select and spoiler
  included, placed on the timeline and summed. Nothing is bounded.

---

## Equivalence tests

Each shortcut is a claim that two calculations agree, and each has a test
that computes both:

| | held against | |
|---|---|---|
| the canonical window | every repetition it stands for | pointwise, every sample of every axis; and the window's peak is at least any repetition's over all four kinds of variation — exactly equal on the family drawn above |
| one window per shape group | the encoding that needs none | the same spiral as four written-out arms and as one arm turned by a rotation: 122.2063 % either way |
| the assembled response | the same window convolved whole, from the published kernel in double precision | $4\times10^{-14}$ % of threshold in double, $2\times10^{-5}$ % as the library computes it |
| the identity two occurrences are matched on | the materialised samples | asserted sample by sample in `PULSEG_DEBUG_MEMO` builds; in a shipped build a slice with no gradient of its own on that axis sends the window to the exact route |
| the assembled response, sample by sample | the same window convolved whole | every sample of all three axes, wrapped history included, over every fixture the assembly is taken on |
| a one-repetition sequence | the plain convolution | the same waveform under the same model, evaluated two ways |
| the wrapped history | the scan played back to back | the window's peak is the steady-state peak, boundaries included |
| the exact scan | the scan rendered whole at half the raster and run through the model's own evaluator | the same peak within the FFT allowance on GRE, EPI, FSE, spiral, radial, ZTE and the written-out arms |
| the chunks dealt in parallel | the same chunks in order | the same peak and the same block, bit for bit |
| the written-out ladder | the rotated twin | the same verdict on both sides of the threshold at 72, 96 and 256 arms |
| `tr=None` | upstream PyPulseq | to the bit |

This is a predownload cost, not a UI one: `validate_protocol` returns before
any gradient exists to differentiate. What the interpreter pays once the
finished file comes back in — gradient continuity and slew, this check, and
the mechanical-resonance analysis together — is what the {doc}`full benchmark
<full_benchmark>` reports as *Safety*, and what the {doc}`pipeline budget
<pipeline_budget>` holds to its line.
