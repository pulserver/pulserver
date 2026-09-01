# Mechanical resonance

```{admonition} TL;DR
:class: tip

- The rule is {doc}`the safety page's <../safety/mechanical_resonance>`,
  evaluated in a cheaper order:
  $A_\text{eq}(f) = \tfrac{2}{T_\text{TR}}|S_\text{ax}(f)|$ at every guarded
  frequency.
- **The zero-tolerance threshold had to be calibrated**, because a literal zero
  refuses every sequence. Measured against the shipped corpus it sits in a clean
  gap at **7.5 mT/m** of equivalent sustained amplitude. It is a policy, not a
  physical constant.
- **The window** is the per-position spectral bound, a proven ceiling on the
  scan at every frequency, 1.00–1.08× the drive it stands for.
- **Cost.** One transform per distinct waveform, a rank basis that collapses a
  multishot family, closed forms per shape class, a chirp-z over each band's
  harmonic comb, and an L1 ceiling that settles a whole band without probing it.
  Scan length drops out entirely.
- **A band is covered, not searched**: every harmonic it contains plus the
  finite-repeat lobes between them, each an exact evaluation.
```

{doc}`The safety page <../safety/mechanical_resonance>` fixes the quantity: per
axis, per frequency, the equivalent sustained amplitude

$$A_\text{eq}(f) \;=\; \frac{2}{T_\text{TR}}\bigl|S_\text{ax}(f)\bigr| ,$$

in the mT/m a vendor states its limits in. Two things stand between that and a
verdict: how much is too much, and how to compute it in the time predownload
can spend.

---

## 1. Calibrating the threshold

A band table states a range and, sometimes, an amplitude. It does not say what
makes the range dangerous, what the resonance's Q is, or how the amplitude was
arrived at. Zero cannot be taken literally — a periodic gradient has lines at
every harmonic of its own period, and any real band is far wider than that
spacing, so every band contains lines of every sequence.

So the threshold was measured against the only evidence there is: a vendor's
own product sequences. Some families ship with a frequency lockout and some do
not, and where a lockout exists it is enforced by steering a parameter away from
the band rather than by refusing the scan. Which families those are is the
vendor's business; what the calibration needs from that inspection is one
number.

Every shipped plugin was designed at protocols a console could plausibly
prescribe, on two sets of system limits, and the $A_\text{eq}$ each puts inside
515–1650 Hz — the range where every inspected band falls — was measured the way
the gate measures it:

```{figure} ../assets/mechanical_resonance/threshold_ladder.png
Equivalent sustained amplitude of the shipped plugins across realistic
protocols, against the threshold.
```

| loud — resembles a family a vendor puts a lockout on | in band |
|---|---:|
| bSSFP at the shortest TR, on two gradient systems | 22.5, 22.3 mT/m |
| single-shot EPI, 64 matrix | 11.3 |
| 3D gradient echo, TR 6 ms | 8.9 |

| quiet — resembles a family a vendor leaves alone | in band |
|---|---:|
| radial gradient echo, TR 10 ms | 6.1 |
| spiral 8 arms · stack of stars · 2D gradient echo, TR 9 ms | 5.3, 5.2, 4.9 |
| EPI 96 · MPRAGE · gradient echo TR 100 ms · spin echo | 1.9, 1.2, 0.5, 0.1 |

Everything loud is a sustained comb — a readout train, or a repetition short
enough that the TR harmonic itself lands in the audio range. Everything quiet is
broadband or slow. The loud group starts at 8.9 and the quiet one tops out at
6.1, so the threshold sits in that gap at **7.5 mT/m**. The corpus is bimodal
enough that any value in the gap gives the same verdicts.

Three properties of that number:

It is a **policy, not a physical constant**, and it is ours rather than a
vendor's; the code says so where it is defined.

It is **below** the thresholds implied by bands that do state an amplitude,
which is the right order — a band that forbids a readout outright is a stricter
statement than one permitting a readout up to a level.

A band that states an amplitude keeps it, **converted first**. That amplitude
describes the plateau of a readout train while $A_\text{eq}$ is an equivalent
sinusoid, and over the train shapes a system can play the equivalent sinusoid
runs between $8/\pi^2$ and $4/\pi$ of the plateau. The smallest is taken.

Nothing in the gate reaches into the sequence to apply this. It never asks what
family it is holding, never looks for an echo spacing, never reads a repetition
time as a parameter to be steered. A gradient echo whose readouts are made
bipolar and numerous enough becomes an echo-planar train in everything but
name, and is refused on exactly the arithmetic that refuses an EPI.

---

## 2. What the analysis is given

A scan is $M$ repetitions of one structural TR, an ordered list of **base
block** ids — the tuple of event *definitions* a block plays plus its duration
({doc}`the PulSeg split <../background/pulseg>`). For the gradients that leaves
an ordered list of $(g_x, g_y, g_z)$ definition tuples, each one of two things:

| | identified by | plus, per playout |
|---|---|---|
| trapezoid | delay, rise, flat, fall | amplitude, rotation |
| arbitrary | delay, time shape, sample count | waveform shape, amplitude, rotation |

That table is the whole cost model. Position fixes a start time, so the shift
theorem handles *when* an event plays; a definition fixes a shape, so one
transform serves every playout of it; amplitude and rotation scale and mix an
answer already computed. What is left to transform is the set of distinct
shapes, which for a Cartesian sequence is a handful and never grows with the
matrix.

Note where the waveform shape sits in the arbitrary row. Two spiral arms written
out shot by shot have the *same* definition and different shapes — one entry in
the sequence's library, two distinct waveforms here. Keying on the definition
instead hands every arm the first arm's spectrum.

---

## 3. The canonical TR

The construction is {doc}`the per-position spectral bound <../safety/canonical_tr>`:
positions identical in every repetition enter a coherent complex sum, positions
that vary contribute the largest magnitude over the combinations they really
take. What this page adds is how tight it is, and what the figure shows about
variation that is *not* periodic.

```{figure} ../assets/mechanical_resonance/canonical_tr.png
One repetition played sixteen times, with each kind of variation switched on
alone.
```

Each panel is drawn against the **whole scan**: what it drives at each TR
harmonic, summed from the engine's own per-repetition evaluation, and faintly
behind it what it drives at every frequency, from the rendered record.

That answers the obvious objection — *a phase encode that steps is not itself
periodic, so how can it drive a resonance?* It cannot, and the figure shows it
not doing so. With nothing varying the scan is a clean comb at the harmonics of
$T_\text{TR}$. Switch a variation on and its energy does not move onto that
comb: it **spreads off** it, into fine structure at multiples of
$1/(M\,T_\text{TR})$, the period the scan actually has.

The window covers both, and not loosely:

| | judged / driven, median | at the loudest in-band line |
|---|---:|---:|
| A nothing varies | 1.000× | 1.000× |
| B the encode amplitude varies | 1.025× | 1.038× |
| C the readout waveform varies | 1.061× | 1.070× |
| D both vary | 1.071× | 1.075× |

An amplitude bound is nearly free: one waveform's largest amplitude *is* its
largest contribution, so the bound is something a repetition really plays. A
shape bound costs a little more, because the loudest shape at one frequency is
rarely the loudest at the next and the bound takes the best of each.

Above about 1.2 kHz in C and D the two curves separate widely, and that is the
mechanism working rather than failing. Where the arms decorrelate, what they
have in common at a harmonic cancels, so the scan's *harmonic* content
collapses — while the drive between the harmonics, the faint trace, does not.
The verdict is a statement about every frequency a band contains, so the window
has to stay up there.

Bounding position by position also avoids a choice that cannot be made well.
The arms of a trajectory are near-copies, so their spectra nearly coincide, and
where they separate they separate at the harmonics. There is no reason the
loudest arm at a guarded frequency should be the loudest arm overall — one can
be quieter everywhere except inside the band. Picking a representative shot by
its peak amplitude or its peak slew would pick by the wrong number.

Neither B nor C changes the *period*: amplitude is an instance parameter, and a
gradient definition is its delay and its time shape, not its sample values. The
shipped 3D stack-of-spirals scan with eight arms written out across eight
partitions is detected as 64 repetitions of one 6.5 ms shot, not one 52 ms
shot — the arms are exactly the variation the window has to bound, and there is
no longer period for them to hide in.

---

## 4. One transform per distinct waveform

The TR is read once, from the gradient columns of the block table rather than
from any rendered waveform.

```{figure} ../assets/mechanical_resonance/memoization.png
The block table's gradient columns, gathered by position across the TR
instances and deduplicated on the `(gx, gy, gz)` tuple each position plays. A
position taking one tuple contributes to the coherent sum; a position taking
several is bounded by the loudest of them, and its waveforms are what the rank
basis compresses.
```

**The tuple, not the axis, is the unit of the walk.** What is counted per
position is the number of distinct `(gx, gy, gz, amplitudes, rotation)`
combinations. That is what makes the bound exact: the loudest combination that
*occurs* is not in general the product of the per-axis maxima, and a rotation
mixes the axes so they cannot be bounded separately anyway.

**And then, per axis, a rank basis.** A spectrum is linear in the waveform and a
truncated tail can be bounded and added back, so the distinct waveforms one axis
takes at a varying position are stacked into a matrix and decomposed. Written
out shot by shot, a spiral gives one waveform per arm — but every arm is the
base arm turned, so on each physical axis they span exactly two dimensions
however many arms there are:

```{figure} ../assets/mechanical_resonance/basis_equivalence.png
The arms, their singular values, and the two encodings judged arm by arm.
```

The third singular value is $2\times10^{-8}$ of the first — the precision the
waveform is stored in. With $g_k(t) = \sum_r c_{k,r} v_r(t)$ and the transform
linear, $G_k(f) = \sum_r c_{k,r} V_r(f)$, so one transform per basis vector
replaces one per waveform. The truncated tail is bounded when the basis is built
and added to every magnitude it produces, so a compressed position reads louder,
never quieter.

**Compression is attempted, never assumed, and refused rather than half-taken.**
A rank is accepted only if it is at most half the number of waveforms *and* its
discarded tail stays under $10^{-6}$ of the set's own L1 norm. If no rank meets
both, if the waveforms do not share a sampling, if there are fewer than four of
them, or if the decomposition would cost more than the transforms it saves, no
basis is built and every waveform is transformed individually — exact, and paid
per waveform.

Which puts one thing in the sequence author's hands. A trajectory whose shots
are a base shot turned, or a small set of templates recombined, is low rank and
costs the same however many shots it has. A trajectory whose shots are each
separately optimised — a sparkling-type design — is genuinely high rank, the
basis is refused, and the check pays for every shot.

The right-hand panel above is the same scan encoded both ways — one waveform
turned by a rotation, and eight written out and compressed back to a rank-2
basis — agreeing arm by arm to 4 parts in $10^7$.

The basis does not travel: it is built per position and per axis, so a family of
waveforms appearing at several positions is decomposed once for each. The
transforms are memoized on the waveform identity and shared; only the
decomposition is repeated.

**Then transform each distinct waveform once.** From there the per-axis spectrum
at a frequency is bookkeeping: each occurrence takes its base response, scales
it by its amplitude, mixes the logical axes by its rotation matrix, multiplies
by $e^{-2\pi i f t_k}$, and adds into a complex accumulator. The axes are
combined coherently on the way in, not at the end, because a rotation mixes them
and a spectrum is complex.

```{figure} ../assets/mechanical_resonance/basis_cost.png
Gate cost against a transform of the timeline, against basis size, and against
harmonics in a band.
```

**Scan length: gone.** A transform of the timeline goes from 20 ms to 623 ms
between 32 and 1024 repetitions. The gate goes from 0.02 ms to 0.29 ms, and what
growth there is belongs to the walk that collects the per-position combinations,
not to the spectral sum.

**Basis size: flat.** From 4 to 64 written-out arms the gate holds at 2.2–3.5 ms
with no trend, against 1.6–2.4 ms for the same scan written with a rotation per
shot. Against the same build with the rank basis off, in a dense band:

| written-out arms | one transform per waveform | rank basis | |
|---:|---:|---:|---:|
| 16 | 12.1 ms | 2.5 ms | 4.8× |
| 64 | 35.5 ms | 3.3 ms | 10.6× |
| 256 | 133.5 ms | 8.8 ms | 15.2× |
| 512 | 266.6 ms | 13.0 ms | 20.5× |

Thirty-two times the arms costs twenty-two times as much on the left and five
times as much on the right, and what is left on the right is not transforming at
all — it is the single pass that reads the arms in. The compression is free when
it cannot help: an inversion-prepared scan with no interchangeable waveforms
measures 16.6 ms with the basis off and 16.1 ms with it on.

**Harmonics in the band: what remains.** A band holds `width × T_TR` harmonics,
so a long repetition time puts hundreds of lines inside a hundred-hertz band.
Two harmonics cost 0.13 ms, thirty-two cost 0.34 ms. This is why the expensive
sequence in the corpus is not the biggest one: a long-TR inversion-prepared scan
has a few thousand blocks and a basis of thirteen waveforms, and its two-second
repetition puts two hundred lines in every band.

Two things keep that term from mattering:

*Only the lines a band contains.* There is never a spectrum to compute — only
the $k/T_\text{TR}$ inside a guarded band, widened by half the width of the
narrowest band in the table.

*A ceiling on the drive itself.*
$|S_\text{ax}(f)| \le \int_\text{TR}|g_\text{ax}|\,dt$ at every $f$, since the
phase factor has unit modulus. The right side costs one pass over the events
and sits 1.3–2.0× above the loudest line a full walk finds. Where it is already
under a band's threshold, nothing in that band can violate it and the search is
skipped as a proof rather than sampled into a measurement. On the
inversion-prepared case against a hundred-hertz band: 215.6 ms with every probe
evaluated, 110.3 ms with the probes placed properly, 16.0 ms once the ceiling
settles the band first. Drawing the lines is exempt, so a plotted amplitude is
always the measured one.

---

## 5. Computing one response

Every gradient is a list of (time, amplitude) vertices with the field ramping
between them, so a spiral arm and a trapezoid differ only in vertex count. What
differs is which shortcut earns its place.

Where that model is the played field, and where it approximates it: exact for a
trapezoid, and exact for an arbitrary waveform carrying a **time shape**, whose
samples sit on raster edges. For an arbitrary waveform on a **uniform raster**
it is an approximation — those samples sit at raster *centres*, and the hardware
holds each across its whole cell. Two consequences, both small: a held cell
contributes $\Delta t\,\mathrm{sinc}(f\Delta t)$ where a ramp between centres
contributes $\Delta t\,\mathrm{sinc}^2(f\Delta t)$, 0.05 % apart at the top of
the guarded range; and an interpolant that starts at the first centre and stops
at the last leaves the outer half-cell at each end uncovered. On a real spiral
readout the two models differ by at most 0.006 mT/m.

**A few vertices — trapezoids, ramps, short arbitraries.** The segment integral
of a linear ramp is closed form, so the transform is a four-term sum evaluated
directly. Nothing is tabulated.

**A uniform raster.** When the sample spacing is constant, the segment integrals
are the same for every segment and the start-time phase advances by a constant
rotation. Both come out of the loop — four transcendentals per segment to none —
with the phase recurrence re-anchored on a fixed segment budget so drift stays
bounded whatever the sample count.

**Many vertices, many frequencies — the chirp-z.** Within one band the
frequencies asked for are consecutive TR harmonics, each probed at the same
fixed offsets, so every evaluation sits at $(k+\alpha)/T_\text{TR}$ with $k$
over an integer range. On that comb the uniform-raster transform reduces to one
polynomial

$$W(\omega) \;=\; e^{-i\omega t_0}\Bigl[A(\omega)\bigl(P - v_{n-1}q^{\,n-1}\bigr)
  + B(\omega)\,q^{-1}\bigl(P - v_0\bigr)\Bigr],
\qquad q = e^{-i\omega \Delta t},\;\; P(q)=\sum_k v_k q^k ,$$

with $A$ and $B$ the same segment integrals the direct loop forms. $P$ is the
only term that costs anything, and a chirp-z transform produces it at every
candidate in the band at once — $O((n{+}m)\log(n{+}m))$ against $O(nm)$. This
is a change of summation order, not of model: the same integral of the same
interpolant, in the same double precision. It engages only past a vertex-count
threshold, so a trapezoid never meets it.

**A run of equally spaced copies — the train.** An echo train is one waveform
repeated at a fixed spacing, so its coherent sum is a geometric series — closed
form when the amplitudes are equal, a Horner evaluation when they are not. An
echo train of any length costs one evaluation.

```{figure} ../assets/mechanical_resonance/shape_response.png
Each family's closed-form response against a direct Fourier integral of the
rendered repetition.
```

Held against a direct numerical Fourier integral of the rendered repetition,
sharing no code with the engine: trapezoids and compressed trains agree to 5
parts in $10^7$, the float truncation at the engine's own interface. For a long
arbitrary waveform the median difference is 2 parts in $10^4$, and it does not
shrink when the reference integral is refined, because it is the half-cell at
each end. Absolutely rather than relatively, the largest disagreement anywhere
is 0.006 mT/m against a 7.5 mT/m threshold.

### The lines a finite scan has

A resonance does not know where the harmonics of a repetition time are. It
responds to drive at *its* frequency, so any frequency inside a guarded band
counts — and the frequencies between the harmonics are not empty.

They would be, for an infinite repetition. A real scan is $M$ of them, which
multiplies the single-TR transform by the Dirichlet kernel
$D_M(x) = \sin(M\pi x)/(M\sin(\pi x))$, $x = f\,T_\text{TR}$:

```{figure} ../assets/mechanical_resonance/finite_reps.png
One harmonic of a finite scan, its Dirichlet lobes, and where the probes have to
sit.
```

The kernel peaks at $x = k + (j+\tfrac12)/M$ with heights $2/\pi(2j{+}1)$:
0.64, 0.21, 0.13, 0.09, and down. Those heights do not depend on $M$ — only
their spacing does — so four probes a side cover every lobe above a tenth of the
main one for any scan length. On the harmonic drawn above the sequence puts
14.5 mT/m, and its first sidelobe, 0.6 Hz away, still carries 9.3 — over the
threshold in its own right, and invisible to a test that looks only at
harmonics.

Where the probes sit is not a tuning knob. Sample at multiples of $1/M$ instead
and every probe lands on a **null** of the kernel: the open markers in the
figure read exactly zero, while the record they sample carries up to
0.02 mT/m there.

Each probe needs its own transform, and that part cannot be economised.
$S_\text{TR}(f)$ is not smooth between the harmonics: it is a coherent complex
sum over every event in the repetition, so where the repetition carries nested
periods of its own those factors multiply, and their product can cancel a peak,
split it into a pair, or shift its maximum off the harmonic it belongs to. So a
probe cannot be a harmonic's value scaled down — taking a neighbouring harmonic
and applying the kernel can only ever expose *less*, while the thing it would
hide is a peak that moved.

Asking for a *picture* is a different question with a different answer: a dense
comb over the whole displayed range, from the same chirp-z.

---

## 6. The verdict

A guarded band is not *searched* — it is **covered**. The analysis evaluates a
fixed, known set of frequencies inside it: every harmonic $k/T_\text{TR}$ the
band contains, and between each pair the peaks of the finite-repeat kernel.
Every one is an exact evaluation of $A_\text{eq}$ compared with the band's
threshold on its own. Nothing is measured against a local baseline, nothing is
promoted for standing out from its neighbours, and no peak is picked: the band
passes when every evaluated frequency passes.

The per-harmonic markers are a reporting convention on top of that. Each
harmonic's row carries the worst of the frequencies in the interval up to the
next harmonic, so a row can be flagged by a lobe rather than by its own line.

```{figure} ../assets/mechanical_resonance/epi_comb.png
An echo train's teeth read against two bands that state no amplitude.
```

A single-shot echo train is the clearest case — its teeth are the harmonics of
the echo spacing, and the comb is what the analysis produces without ever being
told there is an echo spacing. Twenty harmonics fall inside the two bands and
two rows are flagged. One is the tooth at 693 Hz, 14.5 mT/m on its own harmonic.
The other is the harmonic below it at 674 Hz, whose own line is 0.5 mT/m: the
frequencies its row covers run up to the tooth's first sidelobe at 9.2 mT/m.
That 9.2 is not inferred from the shape of the comb — it is what the scan drives
at 693.9 Hz, computed there.

```python
freqs, spectrum, *_ , lines = seq.calculate_gradient_spectrum(
    plot=True, tr="worst_case", resonance_lines=True, bands=bands,
)
lines.line_freqs      # the harmonics inside a guarded band
lines.line_a_eq       # their equivalent sustained amplitudes, per axis
lines.ok              # the verdict the predownload gate will reach
```

---

## Equivalence tests

| | held against | |
|---|---|---|
| the closed forms | a direct numerical Fourier integral of the rendered TR | float precision for trapezoids and trains; 0.006 mT/m at most for arbitrary waveforms, from the outer half-cell |
| the chirp-z tabulation | the direct closed-form loop, waveform by waveform | the same sum reassociated, equal to the last bits |
| the canonical window | every repetition it stands for | at every guarded line and axis, the judged number is at least what any repetition drives |
| a one-repetition sequence | the plain coherent sum | bit for bit identical, which keeps the bound from being a blanket margin |
| the rank basis | the encoding that needs none | the same spiral turned by rotations and written out, at arm counts below and above the compression threshold, arm by arm |
| the ceiling | the search it skips | the threshold placed either side of a sequence's own loudest line, over bands wide and narrow: the verdict flips exactly on the peak |
| the drawn lines | the predownload verdict | on recorded sequences, through two independent paths into the engine |

This is a predownload cost, not a UI one: `validate_protocol` returns before any
gradient exists to transform. What the interpreter pays when the finished file
comes back in — `pulseg_check_safety`, gradient continuity and slew, PNS and
this analysis together — is what the {doc}`full benchmark <full_benchmark>`
reports as *Safety*, over the largest protocol of every shipped family with two
guarded bands: 2 ms to 11.3 s. The top of that range is not the biggest scan; it
is a ZTE shell, where the number of distinct waveforms grows with the scan as
well as the window does — every shot is its own shell — which is exactly the
case the rank basis cannot help with. What the analysis holds in memory is the
basis and the tabulated transforms for one band, both bounded by the window and
released with it.
