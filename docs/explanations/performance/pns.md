# Peripheral nerve stimulation

```{admonition} TL;DR
:class: tip

- The rule is {doc}`the safety page's <../safety/pns>`, evaluated in a cheaper
  order: $\max_t \lVert (k * \dot G)(t)\rVert_2$ over one window, as a
  percentage of the model's own threshold. Nothing is calibrated and nothing
  is approximated.
- **The window** is the positional-maximum envelope, one per group of
  repetitions playing the same waveforms. On the corpus its peak lands
  *exactly* on the worst repetition's.
- **Scan length drops out.** A convolution of the timeline grows from 5.8 ms to
  182 ms between 3 and 144 repetitions; the window stays at 1.5–1.9 ms.
- **Window length is traded for shape count.** One convolution per distinct
  gradient shape, then a scaled shifted add per occurrence: 1.8× on a spiral
  GRE, 347× on an FSE train.
- The regrouping is exact — $4\times10^{-14}$ % of threshold in double
  precision, $2\times10^{-5}$ % as the library computes it in float32.
- **Past 64 shape groups there is no window.** The scan is priced by a bound
  built from each block's own response — exact per element, priced by its
  neighbours' tails at the gap they really have — within 1.1–1.4× of the
  scan's true peak, and the stretches the bound cannot clear are evaluated
  exactly. 131 072 distinct arms gate in seconds.
```

{doc}`The safety page <../safety/pns>` fixes the quantity: the slew rate of
every gradient axis, run through the coil's nerve model, combined across axes
and taken at its peak,

$$r(t) \;=\; \bigl\lVert\, (k * \dot G)(t) \,\bigr\rVert_2 ,
\qquad \text{verdict} \;=\; \max_t r(t).$$

Unlike {doc}`mechanical resonance <mechanical_resonance>`, **nothing here has
to be calibrated**. A nerve model states its own threshold: the response *is* a
percentage of it, 100 % is the limit, 80 % is the margin. There is no policy to
derive.

That leaves the cost. $\dot G$ is minutes of waveform on a microsecond raster
and $k$ is tens of chronaxie constants of history — 721 samples on the raster
these figures are drawn on — so written directly the check is $3NK$
multiply-adds over a scan that grows with the prescription, and it has to hold
for every repetition. The model throughout is **Irnich rheobase/chronaxie**,
the one the scanner-side gate runs and the only one of the two that publishes a
kernel.

---

## 1. What the analysis is given

A scan is $M$ repetitions of one structural TR. One TR is an ordered list of
**base block** ids, and a base block is the tuple of event *definitions* it
plays plus its duration ({doc}`the PulSeg split <../background/pulseg>`). For
the gradients that leaves an ordered list of $(g_x, g_y, g_z)$ definition
tuples, each entry one of two things:

| | identified by | plus, per playout |
|---|---|---|
| trapezoid | delay, rise, flat, fall | amplitude, rotation |
| arbitrary | delay, time shape, sample count | waveform shape, amplitude, rotation |

That table is the whole cost model. Position fixes a start time, so a shift
handles *when* a block plays; a definition fixes a shape, so one convolution
serves every playout of it; and amplitude scales a response already computed.
What is left to convolve is the set of distinct shapes, which for a Cartesian
sequence is a handful and never grows with the matrix.

Note where the waveform shape sits in the arbitrary row. Two spiral arms
written out shot by shot have the *same* definition — same delay, same
sampling, same length — and different shapes. One entry in the sequence's
library, two distinct shapes here.

---

## 2. The canonical TR

The construction is {doc}`the positional-maximum envelope <../safety/canonical_tr>`:
each position at the largest amplitude it reaches, repetitions grouped by the
waveforms they play, one window per group, worst group wins. What this page
adds is how tight it is.

The figure switches each kind of variation on alone, on a controlled 3D
stack-of-spirals scan — one shipped repetition played four times, same
excitation, same spiral, same partition encode, same 10 ms period. The
reference is the **whole scan**: every repetition as it plays, concatenated,
differentiated and convolved in one pass with no knowledge that the scan has a
period. Both sides run through the same extraction.

```{figure} ../assets/pns_performance/canonical_tr.png
Four repetitions played straight through, with each kind of variation switched
on alone.
```

The four panels are four genuinely different scans — the repetitions' gradients
differ by up to 16 mT/m when the encode sweeps and 28 mT/m when the arms do —
and they return the same verdict, with the window's peak landing exactly on the
worst repetition's:

| | repetitions differ by | judged | worst repetition |
|---|---:|---:|---:|
| A nothing varies | — | 95.611 % | 95.611 % |
| B the encode amplitude varies | 16 mT/m | 95.616 % | 95.616 % |
| C the readout waveform varies | 28 mT/m | 95.611 % | 95.611 % |
| D both vary | 28 mT/m | 95.616 % | 95.616 % |

**A** is trivial: nothing varies, so the window *is* the repetition. **B** is
the per-position amplitude maximum meeting a scan that plays it — a real encode
table sweeps to its own extreme, so some repetition plays that maximum. The
other three come in fractionally lower, 95.598 % at the smallest encode, which
is the whole visible effect of a 16 mT/m swing.

**C and D reproduce A and B exactly**, and that is structural rather than a
coincidence of the prescription. The chronaxie model applies one kernel to
every axis, so it commutes with the rotation that turns one arm into the next,
and the verdict is a root-sum-square, which a rotation leaves alone. Turning a
spiral moves stimulation between $G_x$ and $G_y$; it does not change how much
of it there is.

```{figure} ../assets/pns_performance/multishot_envelope.png
Spiral gradient echo: four interleaves, and the window built over each.
```

The peak is 122.2063 %, and it is the same 122.2063 % whether the arms arrive
as four written-out waveforms or as one arm plus a `ROTATIONS` extension — the
repetitions agreeing with the window to $1.2\times10^{-5}$ % of threshold.

**Where the window would be strictly conservative**, and this scan is not, is
when two positions vary *independently* and no repetition plays both extremes
at once. The maximum is per position, so the window then assembles a repetition
nobody plays.

A model with per-axis coefficients has no such symmetry to lean on. SAFE
carries a separate coefficient set per axis, and a rotation extension is
applied below this analysis rather than inside it, so for that model a rotated
arm is not what the check sees. The grouping is the honest route there.

---

## 3. One response per distinct waveform

The window is read once, from the gradient columns of the block table rather
than from any rendered waveform.

```{figure} ../assets/pns_performance/memoization.png
The block table's gradient columns, gathered by position across the TR
instances and deduplicated on the `(gx, gy, gz)` tuple each position plays. A
position taking one tuple varies only in amplitude; a position taking several
is what forces the instances into groups.
```

**The tuple, not the axis, is the unit of the walk.** A position is the same
position in two instances when all three gradient columns match, and a column
is a pair: the definition, which fixes the timing skeleton, and the shape id,
which fixes the samples. A phase encode changes an amplitude and leaves the
tuple alone; a written-out spiral arm changes two shape ids and gives the
position a second tuple. Positions with one tuple are bounded by their largest
amplitude; positions with several send the instances to be grouped.

**Inside a window, the convolution is memoized per axis.** Each block's slice
is keyed on (gradient definition · shape id · block duration) — the identity
the representation already carries, so nothing is compared sample by sample.
Each distinct key is convolved with the kernel once; every occurrence takes the
stored response, scales it by its amplitude ratio, shifts it to its start time
and adds into an accumulator. The axes are combined and the peak read off at
the end.

That identity does not carry **rotation**, and the slices are cut from the
already-rotated waveform. What keeps that sound is a precondition rather than a
comparison: a slice is only matched against the logical gradient of the same
axis index, and if any physical axis carries a slice with no logical gradient
of its own — which is what a rotation leaking one logical axis onto three
physical ones produces — the assembly declines and the window takes the exact
convolution instead. `PULSEG_DEBUG_MEMO` builds additionally compare the
samples of every accepted match against its template.

```{figure} ../assets/pns_performance/assembly_cost.png
What the stimulation check's cost actually depends on.
```

**Scan length: gone.** A convolution of the timeline goes from 5.8 ms to 182 ms
between 3 and 144 repetitions of one EPI shot, as a convolution of the scan
must. The window stays at 1.5–1.9 ms throughout, and a clinical prescription is
two orders of magnitude longer than the 14 000 blocks measured here.

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

A short window with few repeats buys almost nothing; a two-second inversion
train buys two orders of magnitude, because its length grew and its shape count
did not. The assembly is taken only where it is expected to win by at least 4×,
and only on windows over 512 samples.

---

## 4. Computing one response

A nerve model that publishes a kernel is asserting it is a linear filter,
$r(t) = (k * \dot G)(t)$. Convolution is linear and shift-invariant, and the
window is already written as a sum of delayed shapes because that is what a
block list is:

$$G(t) = \sum_i a_i\, g_{c(i)}(t - t_i)
\qquad\Longrightarrow\qquad
r_\text{ax}(t) = \sum_i a_i \,\bigl(k * \dot g_{c(i)}\bigr)(t - t_i),$$

block $i$ playing shape $g_{c(i)}$ at amplitude $a_i$ from time $t_i$. Each
distinct shape is convolved once; every occurrence is a scaled, shifted add.
Nothing is truncated, resampled or approximated — the two expressions are the
same sum, regrouped.

The convolution is done in the **time domain**: a template is one block long
against a kernel spanning tens of chronaxie times, so a transform of the padded
template would be mostly padding, and a direct linear convolution has no
wraparound to guard against.

Two details make the algebra true of the implementation. **At the seams**, each
block's slice contributes its slew zero-extended on both sides — it opens with
the step up from zero into the block and closes with the step back down — so
where two blocks meet, the closing step of one and the opening step of the next
add to exactly the difference a forward difference across the seam would have
subtracted. **Around the window**, the occurrences are placed again one window
later, and again for as many windows as the padding spans, which is the same
warmed-up history that wrapping the waveform round gives. How far the padding
reaches is set by the model's memory, so a repetition shorter than that memory
is replayed more than once.

**Then the axes combine.** Per-axis responses are percentages of the same
threshold and are combined by root-sum-square at every instant before the peak
is taken. This is the one step that does *not* decompose — a root-sum-square of
sums is not a sum of root-sum-squares — so the assembly runs per axis and the
combination happens once, at the end.

```{figure} ../assets/pns_performance/assembly_equivalence.png
EPI, one canonical TR: assembled per shape against convolved whole.
```

One EPI window: 1601 samples, 25 blocks, 45 slices, 11 distinct shapes,
5 020 323 multiply-adds convolved whole against 658 847 assembled. All three
routes report **124.4607 %** to seven digits. The bottom panel separates the
two error terms: regrouping the sum in double precision moves the answer by
$4\times10^{-14}$ % of threshold — the algebra being exact — and the library
sits $2\times10^{-5}$ % away, which is the float32 it computes in.

The fast route exists only where there is a kernel to publish. **SAFE** is not
a convolution — its three branches rectify on both sides of their lowpass — so
it takes the direct route always, over the same window and the same groups.

---

## 5. The verdict

```{figure} ../assets/pns_performance/epi_verdict.png
An echo train's stimulation against the 80 % margin and the 100 % threshold.
```

A single-shot echo train is the instructive case, and not for the reason it
looks like. The train shows up as ~50 % teeth, one per gradient reversal, and
they do not stack: the echo spacing is longer than the 360 µs chronaxie. The
verdict, **112 %**, is set instead by the slice-select rewinder at 3.3 ms — one
large excursion on a single axis, nowhere near the readout.

```python
ok, norm, components, t = seq.calculate_pns(
    {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333}, tr="worst_case",
)
ok            # the verdict the predownload gate will reach
norm.max()    # the peak, as a fraction of threshold
```

---

## 6. Past the group cap: the occurrence score

A SPARKLING-style acquisition plays a distinct optimised arm in every readout.
There is no envelope for that — no amplitude makes one arm's shape cover
another's — and grouping by waveform gives one group per repetition. Past
`PULSEG__MAX_SHAPE_GROUPS` the check leaves windows behind and prices the
scan by **linearity**: the response is a sum of per-block responses,

$$R(t) = \sum_i r_i(t - t_i), \qquad r_i = k * \dot G_i ,$$

so at any instant the peak of $\lVert R \rVert_2$ is bounded by a sum over
the blocks that have started. The block playing contributes its **envelope**
— its response peak over each of eight equal windows of its own span — and a
block that ended a gap $\delta$ earlier contributes its **tail peak**
$T_i(\delta) = \sup_{\tau \ge \delta} \lVert r_i(\mathrm{end}_i + \tau)\rVert_2$,
kept at four gap edges where the kernel has fallen to about 1, 1/5, 1/20 and
1/100 of its first tap. A sweep over the scan walks the windows of every
block with earlier blocks migrating outward through the gap zones, so its
cost is the block count, not the scan length or the waveform count.

**Each price is exact for the block on its own.** Every block's response is
sliced to its *interior* slew and priced by its own convolution; the step a
block makes at its start — its first sample against the previous block's
last — is priced per occurrence as one kernel tap of that size, so a
gradient that runs on across blocks is charged its slew and not a fictitious
step at every seam, and one that starts from rest is charged exactly its
start. Prices are computed per **distinct** (shape, amplitude) tuple a block
plays, never per occurrence, and each tuple's own response is computed
exactly by FFT convolution, one tuple at a time, on every core the host
offers (`pulseg_opts.parallel_for_fn`; a scanner-side build leaves it unset
and runs the same loop sequentially). A joint singular-value basis over the
tuples was measured against this and is not kept: an exact element costs
about what one basis column costs and carries no residual, so the basis was
never the cheaper or the tighter of the two once a block played more than a
handful of distinct waveforms.

**A repetition's own curve.** `calculate_pns(hw, tr=k)` past the cap is
repetition `k` played as it stands, and `tr="worst_case"` is the repetition
the score prices highest, evaluated exactly — a witness, not an envelope,
which the diagnostic says.

**The bound decides, or it names what to evaluate.** Below the threshold the
scan passes on the bound alone. Above it, the anchors that exceed are merged
into ranges, each opened a kernel reach of real blocks early so that a cold
evaluation reproduces the scan's own response from its first offending
anchor on, and each is evaluated exactly the way the window path would; a
scan whose bound exceeds in more regions than the check will evaluate is
refused on the bound, with the count in the diagnostic.

| bound over the scan's true peak | |
|---|---:|
| GRE, 2D | 1.14× |
| FSE | 1.42× |
| EPI | 1.24× |
| spiral GRE | 1.44× |
| radial GRE | 1.14× |
| ZTE (gradient continuous across blocks) | 1.74× |
| 64 written-out arms, 4 096 points each | 1.07× |

On the written-out ladder every rung agrees with its `ROTATIONS` twin on
both sides of the threshold, and 8 192 distinct arms are priced in 0.4 s.

## Equivalence tests

Each shortcut is a claim that two calculations agree, and each has a test that
computes both:

| | held against | |
|---|---|---|
| the assembled response | the same window convolved whole, from the published kernel in double precision | $4\times10^{-14}$ % of threshold in double, $2\times10^{-5}$ % as the library computes it |
| the identity two occurrences are matched on | the materialised samples | asserted sample by sample in `PULSEG_DEBUG_MEMO` builds; in a shipped build, a slice with no gradient of its own on that axis sends the window to the exact route |
| the assembled response, sample by sample | the same window convolved whole | every sample of all three axes, wrapped history included, over every fixture the assembly is taken on |
| the canonical window | every repetition it stands for | the window's peak is at least any repetition's, over all four kinds of variation — and exactly equal on the family drawn above |
| one window per shape group | the encoding that needs none | the same spiral as four written-out arms and as one arm turned by a rotation: 122.2063 % either way |
| a one-repetition sequence | the plain convolution | the same waveform under the same model, evaluated two ways |
| the wrapped history | the scan played back to back | the window's peak is the steady-state peak, boundaries included |
| the occurrence score | the scan convolved whole | at or above the true peak on every corpus fixture, within the raster-invariance allowance |
| the score's timeline | the scan's | block spans laid end to end close on the scan's total duration; window and tail prices never exceed the block's own peak |
| the parallel loop | the sequential one | every per-block price identical however the range is dealt out |
| the score's verdict | the ground truth | brackets at 0.5×, 0.99×, 1.01× and 2× the true peak on four families; written-out arms agree with their rotated twins |

This is a predownload cost, not a UI one: `validate_protocol` returns before
any gradient exists to differentiate. What the interpreter pays once the
finished file comes back in — gradient continuity and slew, this check, and the
acoustic analysis together — is what the {doc}`full benchmark <full_benchmark>`
reports as *Safety*.
