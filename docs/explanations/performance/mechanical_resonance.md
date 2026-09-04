# Mechanical resonance

```{admonition} TL;DR
:class: tip

- **What is judged.** Per physical axis and per frequency inside a forbidden
  band, the amplitude of the sinusoid the scan sustains over the coil's
  resonance memory $W$ (`pulseg_opts.mech_memory_us`, 20 ms), against the
  band's tolerance. The rule is {doc}`the safety page's
  <../safety/mechanical_resonance>`; this page shows where its constants come
  from and how it is computed in well under a second.
- **Calibration.** The vendor's three product checks — the EPI echo-spacing
  tables, the FIESTA repetition-time lockouts, the multi-echo spacing
  lockout — are one amplitude rule read at three harmonic orders. Read in
  the vendor's own bands, what the vendor refuses sits in a narrow range of
  sustained amplitude and what it runs unchecked sits below that range; the
  zero-column floor `SA_ZERO_BAND_SINUSOID_MT_PER_M` sits between the two.
  `docs/_bench/mechres_calibration.py` reads the tables and prints the
  bracket.
- **Two paths, one reading.** While the repetitions play the same waveform
  sets, the reading comes from the structural TR laid out over the memory,
  varying positions bounded by their largest magnitude; a bound that refuses,
  or a varying position longer than the memory, is settled by reading the
  scan exactly. Past `PULSEG__MAX_SHAPE_GROUPS` every event of the scan is a
  term from the start.
- **Cost.** One FFT record per distinct waveform, shared across the physical
  axes it is rotated onto and kept across refinement; the verdict starts on a
  coarse grid and refines only bands whose bound crosses their tolerance.
  The corpus is judged in a few milliseconds per scan; a scan of 8 192
  distinct 4 096-sample arms in about 0.2 s on the development machine, and
  the check scales linearly with the arms from there.
```

{doc}`The safety page <../safety/mechanical_resonance>` fixes the quantity:

$$A_W(f) = \max_{t_0}\;\frac{2}{W}\Bigl|\int_{t_0}^{t_0+W} g_\text{ax}(t)\,e^{-2\pi i f t}\,dt\Bigr|.$$

Two things stand between that and a verdict: how much is too much, and how
to compute it without touching a minutes-long record.

---

## 1. What the vendor's tables say

The vendor ships one lockout per product family, each on the parameter that
sets that family's drive frequency:

| table | what it locks | the band it guards |
|---|---|---|
| `lockout/epiesp*.dat`, one file per gradient coil, one section per physical axis | echo-spacing ranges, with a tolerance in G/cm that is zero in almost every row | the echo-planar fundamental $1/(2\,\mathrm{ESP})$ |
| `greAcousticLimit.<coil>.dat` | repetition-time ranges for FIESTA | the harmonic of the readout comb, $k/T_R$, that falls in the coil's band |
| `greAcousticLimitEsp.dat` | echo-spacing ranges for multi-echo gradient echo | the echo period $1/\mathrm{ESP}$ |

The three tables of a coil carry the same resonances: a locked TR range is
the range that puts a harmonic of the readout comb into the band the EPI
table states, and a locked multi-echo spacing puts the echo period there.
One resonance, three families, each read at the harmonic order it drives
the mode at, and which order counts is decided by amplitude. Read the way
the gate reads them, a bSSFP at a locked TR sustains the harmonic in the band
at tens of mT/m where it is the second; the same sequence at a TR whose third
harmonic falls in the band sustains a third of that and is not locked; at a
TR clear of the band it sustains next to nothing there.

Structural rules were tried and measured before this one was kept: a
lobe-pair spacing test is ambiguous by half an octave on one-cycle patterns;
same-sign trains drive the coil as much as alternating ones; a line-to-peak
fraction refuses and allows families the vendor treats the other way round.
The amplitude rule is the only one that reproduces every product decision,
and it needs to be told nothing about the sequence.

## 2. Calibrating the floor

The tolerance column of an EPI table is almost always zero, and zero cannot
be read literally: every gradient puts *some* drive into every band. The
floor a zero column is held to, `SA_ZERO_BAND_SINUSOID_MT_PER_M`, is an
estimate of the harm threshold bracketed by the vendor's own decisions.
`mechres_calibration.py` designs, on two sets of system limits, the sequences
the vendor refuses — an echo train whose fundamental is placed inside a coil's
band by sweeping matrix and receiver bandwidth, a FIESTA at the middle of a
locked TR range, a multi-echo train at a locked spacing — and every family it
runs without a check. A refusal is read in the bands of the coil whose table
refuses it; a family the vendor runs everywhere is read in every coil's
bands:

```{figure} ../assets/mechanical_resonance/scenario_table.png
Every family and every edge case, read as the gate reads it inside the
vendor's own bands. Orange bars are sequences the vendor refuses, blue ones
sequences it runs unchecked, grey the edge cases below; the dashed line is
the floor.
```

The script prints the bracket. On the tables available here the vendor's
refusals cluster: an echo train whose fundamental sits in a coil's band and a
FIESTA at a locked TR, on either coil that locks one, read within a few
tenths of a mT/m of each other at about ten, on the physical axis they drive;
what the vendor runs unchecked reads lower, the loudest being an echo-planar
train whose fundamental sits just outside a band and leaks into it through
the window's own resolution, then the stack of spirals, the MPRAGE and the 3D
gradient echo. The script also prints the tolerances the tables do state,
converted to sinusoid amplitude through the triangle-train factor $8/\pi^2$:
the one stated for the HRMw coil converts to a value above the vendor's own
cluster. The floor sits at that cluster, refusing every refusal on these
tables with half a mT/m of margin to the loudest allowed reading. It is one
constant, set in one place and named where it is used.

```{figure} ../assets/mechanical_resonance/epi_fiesta.png
The vendor's refusals and a free TR, per physical axis, with every band the
echo-spacing tables guard on x shaded.
```

Below the loud lines sit the companions every sequence carries: phase-encode
blips, the overtones of a ramp-sampled train, spoilers. They read at a
fraction of a mT/m to a couple of mT/m — an order of magnitude under the
floor — which is why a zero column can be held to a floor at all.

### Where the floor leaves the vendor's decisions

The product locks a family out by its parameter alone; the criterion reads
amplitude. At the floor, the echo train in band and the FIESTA at a locked
TR are refused as the vendor refuses them. Two kinds of train the vendor
refuses read below the floor and pass: a train whose plateau is under the
floor, which cannot sustain a sinusoid above it; and a multi-echo packet of
a few echoes at the locked spacing, monopolar or bipolar, which fills a
fraction of the memory. The calibration script lists them as accepted
divergences and prints the rest with their readings. An acquisition-based
clause that refused these trains would refuse a spiral of the same plateau,
so the reading stands and the floor is the one policy in the check.

### The memory

$W$ is the coil's, not the band's. No vendor datum constrains it: trains and
combs do not move with it, and a comb's reading changes with $W$ only by the
whole events a window takes at its end. What the memory prices is a sweep or
a burst:

```{figure} ../assets/mechanical_resonance/long_events.png
A short spiral, a long one and a scan of distinct arms under the 20 ms
memory: what a sweep sustains inside a band is what it did while crossing
it, over the part of the memory that took.
```

Band-derived memories were measured and rejected: a memory of a coil's
narrowest band passes the echo train the vendor refuses, and a memory of its
widest band reads a radial comb up to the floor and collapses the bracket.

---

## 3. The structural TR and its bound

The interpreter detects the **structural TR** from block content
({doc}`TR and segmentation <../sequence_model/tr_and_segmentation>`). What a
position plays across repetitions is one of three things:

| | at that position, across repetitions | example |
|---|---|---|
| **A** | nothing varies | excitation, slice select, spoiler, an unrotated readout |
| **B** | the amplitude varies, or the rotation | a phase encode; a radial spoke turned by a `ROTATIONS` extension |
| **C** | the waveform itself varies | a multishot readout written out arm by arm |

```{figure} ../assets/mechanical_resonance/memoization.png
The block table's gradient columns, gathered by position across the TR
instances and deduplicated on the tuple each position plays.
```

A position of kind A contributes its exact complex term to a **coherent
sum**; a position of kind B or C contributes the largest magnitude over the
(waveform, amplitude, rotation) tuples it really takes, per physical axis.
That is {doc}`construction 2 of the canonical-TR page <../safety/canonical_tr>`
and a proved ceiling: every repetition's transform is bounded by the coherent
sum plus the magnitude terms by the triangle inequality, so the scan is under
the window at every frequency. Its cost is tightness where positions vary:

```{figure} ../assets/mechanical_resonance/canonical_tr.png
One repetition played sixteen times, each kind of variation switched on
alone, judged against the whole scan.
```

Written out shot by shot, a spiral gives one waveform per arm, but every arm
is the base arm turned, so on each physical axis the set spans two dimensions
however many arms there are; the varying position's waveforms are stacked,
decomposed, and transformed once per basis vector with the truncated tail
bounded and added back:

```{figure} ../assets/mechanical_resonance/basis_equivalence.png
The arms, their singular values, and the two encodings judged arm by arm.
```

**The bound never refuses on its own.** A refusal reached through magnitude
terms is settled by reading the scan exactly — section 4's second path — so
a phase-encoded gradient echo whose bound crosses a tolerance is refused only
if the scan itself sustains the reading. A varying position longer than the
memory cannot be windowed as a magnitude term at all, so a repetition holding
one is read exactly from the start. Between one and 64 shape groups nothing
else changes: a position that plays several waveforms is kind C with all of
them among its tuples, whatever order the repetitions play in.

---

(mechres-scan-window)=
## 4. The criterion, computed two ways

Every gradient event of a scan is one term — its transform at $f$ times its
amplitude and its placement phasor — and a window of the memory slid over
the scan sums the terms of the events that start inside it:

$$A_W(f) = \max_{t_0}\;\frac{2}{W}\Bigl|\sum_{t_m \in [t_0,\,t_0+W)} a_m\,T_m(f)\,e^{-2\pi i f t_m}\Bigr|.$$

An event longer than the memory is cut into pieces of an eighth of it —
raster-aligned slices of its samples, occurrences of a fused train — so the
window reads the loudest stretch rather than the average over the whole
event; events shorter than the memory are counted whole. Below $f < 1/W$ the
reading is zero.

**From the structural TR.** Section 3's event model is the whole input: the
coherent events laid out over as many repetitions as a window starting inside
the first can reach, at most as many as the scan plays, every varying
position a magnitude term at its time, the windows placed exactly as on any
scan. With nothing varying and $W$ a multiple of the TR this is the periodic
line amplitude; with positions varying it is the ceiling applied window by
window.

**From every event.** A scan with more distinct waveform sets than the
grouping holds — a distinct optimised readout in every repetition — has no
useful bound, so the events of the whole scan are the terms. One real FFT per
distinct waveform gives its transform on the bins of its own grid; a
Kaiser–sinc kernel interpolates it onto the band's grid with an additive
guard; a chunked pass with prefix sums places the windows. Nothing on this
path is bounded.

**Why they agree.** Both are the same sum over the same terms; the periodic
form only knows in advance that the terms repeat. The same events written as
$N$ repetitions of $K$ blocks or as one repetition of $NK$ blocks read the
same amplitude, and the tests hold the two paths to each other on repeated
trapezoids, on a scan whose repetitions differ, and under a prescription
rotation against every transform evaluated directly.

**The grid and its guard.** The reading between two grid points is bounded
by Bernstein's inequality from the run a window covers,
$1/(1-\pi\,\mathrm{span}\,\delta f/2)$. The display reads every band on a
`SA_SCAN_FINE_DF_HZ` grid (0.5 Hz, a factor of a few percent at a 20 ms
memory); the verdict starts on `SA_SCAN_COARSE_DF_HZ` (4 Hz) and settles a
band there when its guarded reading is under the tolerance or its raw reading
is over it, halving the spacing only where the bound crosses and the reading
does not. The two grids reach the same verdict, which is asserted on the
corpus.

---

## 5. Evaluating it cheaply

The cost has three parts, and only one of them scales with how many distinct
waveforms a scan plays.

**Records: one FFT per distinct waveform, whichever axes play it.** A
waveform's transform does not depend on its amplitude, its rotation or its
position, so it is computed once from a real FFT of its samples zero-padded
to twice their count, the piecewise-linear factors applied analytically at
each bin, and kept as a record of the bins each band's kernel taps reach. The
records are keyed on the waveform and the piece it was cut into, so a shape
rotated onto three physical axes by an oblique prescription is transformed
once, and they are kept for the whole verdict while the grid refines. The
FFT is the floor: a 4 096-sample arm costs one 8 192-point real transform,
and a pruned or banded transform of the same bins saves little against it.

**Placement: kernel taps per event per grid point.** Each event's line at
every point of the band is its record interpolated by a 16-tap kernel,
multiplied by its train sum and placement phasor. The work is proportional to
events times grid points, which is why the verdict starts coarse.

**Windows: prefix sums per chunk.** Events are sorted by start, chunked, and
each chunk's windows are read from prefix sums of the lines, so placing the
windows costs one pass over the events per grid point. The chunks and the
records are spread over the machine's cores.

```{figure} ../assets/mechanical_resonance/scale.png
The check alone on scans of distinct 4 096-sample arms, two and three
gradient axes, from the mechanical-resonance micro-benchmark
(`pipeline_budget.py --mech-only`). Dotted: the linear extrapolation to the
128 K arms of the pipeline budget.
```

On the shipped corpus a two-band verdict takes a few milliseconds; the arms
scale linearly, and the {doc}`pipeline budget <pipeline_budget>` holds the
check to its line at 128 K arms. The 128 K point itself is extrapolated: a
scan that large is refused by the file parser before any check runs, which is
a limit of the sequence representation and not of this check.

### How one waveform's transform is computed

Every gradient is a list of (time, amplitude) vertices with the field ramping
linearly between them, and its transform at $f$ is the sum of the segment
integrals, each closed form:

| the gradient | its vertices | its record |
|---|---|---|
| trapezoid | four, from delay, rise, flat, fall | the closed form at the bins of its own grid |
| extended trapezoid | its own vertices at their own times | the closed form summed over the vertices, at the bins |
| arbitrary, on the raster | samples at cell centres, uniform spacing | one real FFT of the samples, the piecewise-linear factors applied at each bin, centred on the waveform's support |
| a train of equally spaced copies | one waveform and a spacing | one record, the copies a geometric or Horner sum at placement |

```{figure} ../assets/mechanical_resonance/shape_response.png
Each family's closed-form response against a direct Fourier integral of the
rendered repetition.
```

For an arbitrary waveform the vertex model is an approximation of the held
samples the hardware plays — half a cell uncovered at each end, a $\mathrm{sinc}$
against a $\mathrm{sinc}^2$ per cell — and on a real spiral readout the two
differ by hundredths of a mT/m.

### The drawn lines

The verdict reads every grid point, so the harmonic lines a sequence plot
draws decide nothing; they are display products of the periodic model, and
they are made honest the same way: a finite scan of $M$ repetitions has
Dirichlet sidelobes between its harmonics, and the drawn lines probe them.

```{figure} ../assets/mechanical_resonance/finite_reps.png
One harmonic of a finite scan, its Dirichlet lobes, and where the probes sit.
```

```{figure} ../assets/mechanical_resonance/epi_seginer_fig1_reproduction.png
The echo-planar comb of Seginer et al. (arXiv 2508.03220) reproduced from
event terms alone: on-raster echo and slice spacing keeps one dominant peak,
off-raster spacing spreads it, with no echo-spacing or slice factor anywhere
in the engine.
```

---

## 6. The verdict and what it names

A band is **covered**, not searched: every grid point on every axis the band
names is compared with the band's tolerance on its own. The refusal names the
frequency, the amplitude against its tolerance, the axis, and the gradient
definitions behind the loudest refused reading with their share of it — the
reading is linear in the events, so the shares are exact, and a share above
one marks a lobe partly cancelled by the others.

```{figure} ../assets/mechanical_resonance/epi_comb.png
An echo train's teeth read against two bands that state no amplitude.
```

```{figure} ../assets/mechanical_resonance/derate_example.png
The 3D gradient echo at TR 6 ms: the loudest in-band line belongs to the
spoiler and the prewinder, not to the readout. Built against a system derated
to half its gradient amplitude — the same areas, the trapezoids longer, the
readout untouched — the line moves.
```

```python
freqs, spectrum, *_ , lines = seq.calculate_gradient_spectrum(
    plot=True, tr="worst_case", resonance_lines=True, bands=bands, memory=0.02,
)
lines.candidate_freqs   # the grid the verdict read
lines.candidate_a_eq    # the window reading there, per axis
lines.tolerance         # what each candidate was judged against
lines.ok                # the verdict the predownload gate will reach
lines.contributors      # (definition, share) behind the loudest refusal
```

`tr="worst_case"` is section 4 computed from section 3's bound and settled
exactly where the bound refuses; `tr=k` is repetition `k` as it plays, its
events alone. To move a lobe the report names, build the sequence on a
derated system (`pp.apply_system_derates`), which is what the last figure
shows; a readout's line is moved through its bandwidth or spacing.

---

## 7. Cases and questions

**One repetition only.** A scan whose structural TR is the whole scan is the
periodic form with one copy plus the window's reach, which is the whole-scan
form: a single-shot single-slice EPI reads its train at the train's amplitude
and is refused when its fundamental sits in a band, exactly as a multi-slice
one is.

**A very short repetition.** A bSSFP with a 3 ms TR has harmonics 330 Hz
apart, and the window reads the amplitude at every grid point between them:
between the harmonics a periodic drive has only the finite-scan sidelobes,
and a window of the memory reads them at their size.

**The same events as $N$ repetitions of $K$ blocks or as one repetition of
$NK$ blocks.** The same reading: the window does not know how the scan was
cut. With amplitudes varying the periodic form takes the magnitude bound
where the long form sums the actual amplitudes, and a bound that refuses is
settled by the long form.

**A spiral, a SPARKLING arm, a radial spoke.** No equivalent echo spacing is
looked for. A spiral crosses a band once and reads what it sustained while
inside it over the memory that took; an arm optimised per shot is a distinct
waveform, transformed once and placed where it plays; a spoke turned by a
rotation is the same waveform at another amplitude per axis. All three are
read by the same sum as an echo train.

**An oblique prescription.** The composed rotation is installed once per
check, so every physical axis carries what the scanner plays on it, and the
records a shape needs on three axes are one record. Only this check is judged
in the prescribed frame.

**Why two ways of computing it.** Cost only. The periodic form touches each
distinct waveform once and a window's worth of terms; the whole-scan form
touches every event. They are the same criterion, and the tests hold them to
each other.

---

## Equivalence tests

| | held against | |
|---|---|---|
| the canonical window | every repetition it stands for | at every guarded line and axis the judged number is at least what any repetition drives |
| a bound that refuses | the scan read exactly | a tolerance the bound alone refuses passes when the scan sustains less |
| the scan window on a repeated TR | the periodic line | at three harmonics, within 2 % |
| the scan window on distinct repetitions | the rendered scan, summed by the definition | within 0.3 % |
| a long event | a window slid over its samples on the same piece grid | within 3 %, and far from the whole-event average |
| shared records under a prescription | every transform evaluated directly | within 2 % |
| the coarse verdict grid | the fine display grid | the same verdict on the corpus, zero and stated bands, tagged and untagged |
| the stated-tolerance factor | the Fourier coefficient of the rendered lobe pattern | within 2 %, square and same-sign trains |
| the memory | a sweep and a comb | the sweep's reading doubles when the memory halves; the comb's does not move |
| the FFT record | every transform summed directly | within $7\times10^{-5}$, never below |
| the drawn verdict | the predownload gate | the same verdict through two independent paths into the engine |

This is a predownload cost, not a UI one: `validate_protocol` returns before
any gradient exists to transform. What the interpreter pays when the finished
file comes back in — gradient continuity and slew, PNS and this analysis
together — is what the {doc}`full benchmark <full_benchmark>` reports as
*Safety*, and what the {doc}`pipeline budget <pipeline_budget>` holds to its
line.
