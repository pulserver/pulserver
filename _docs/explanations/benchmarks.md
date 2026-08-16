# Benchmarks

This page is about the on-scanner path: what the operator waits for, and
what has to fit in target memory. That includes the sequence plugin itself
— in server mode the scanner runs `validate_protocol` and `make_sequence` on
its own host, so the Python design code is inside the budget, not beside it.
What is genuinely offline, and therefore not measured here for its own sake,
is the authoring-time inspection API: `Sequence.pns`, `Sequence.grad_spectrum`
and the plotting entry points exist to be looked at while writing a
sequence, and nothing waits on them.

Everything below was measured with
[`docs/_bench/`](https://github.com/INFN-MRI/pulserver/tree/main/docs/_bench)
on the shipped example plugins. Timings are the minimum of three runs. Heap
figures are live bytes attributable to the library, measured by interposing
the allocator (`-Wl,--wrap=malloc`), so they cover every allocation path and
include allocator rounding — what the process holds, not what it asked for.

Absolute numbers are from one desktop machine and a scanner's host will
differ. The *ratios* and the *scaling* are the point, and those are
properties of the design. This page deliberately keeps *how* each result is
produced out of the way — {doc}`sequence_representation/pulseg`,
{doc}`sequence_representation/pulserver`, {doc}`safety/gradient_slew`,
{doc}`safety/mechanical_resonance` and {doc}`safety/pns` are the method
pages; this is only the measurement.

## The three moments that matter

A sequence is touched at three very different times, and conflating them
hides the only interesting results on this page.

**Protocol edit, on the host, on every keystroke.** The UI calls
`validate_protocol` each time a parameter changes. This is an interactive
latency budget: tens of milliseconds, or the UI feels broken.

**Download, on the host, once per prescription.** The plugin designs the
sequence and writes a `.seq`; the C library parses it, converts it to the
intermediate representation, runs the safety checks, and writes the cache.
The operator is waiting on all of it. The host is a general-purpose
computer, so seconds matter here and memory mostly does not.

**Scan, on the target, once per scan.** Load the cache and play it. Here
memory is the scarce resource, not seconds.

The cache exists to separate the last from the first two, and its
[section structure](sequence_representation/pulserver.md#the-cache-and-why-it-has-sections)
exists so the target never pays for what only the host needs.

![Benchmark scaling across the zoo](assets/benchmarks/pipeline_scaling.png)

## Protocol edit: `validate_protocol`

| Sequence | `validate_protocol` |
| --- | ---: |
| GRE 64² × 1 sl | 57 ms |
| GRE 128² × 16 sl | 49 ms |
| GRE 256² × 32 sl | 33 ms |
| GRE 3D 128² × 64 par | 51 ms |
| FSE 256² × 20 sl, ETL 16 | 104 ms |
| EPI 256² × 20 sl, ETL 16 | 157 ms |
| bSSFP 128², 1 frame | 23 ms |
| bSSFP 128², 64 frames | 22 ms |
| MPRAGE 192² × 128 par | 69 ms |
| Spiral, 48 short shots | 25 ms |
| Spiral, 4 long shots | 59 ms |

**22–157 ms, and flat in sequence size.** GRE at 256² × 32 slices validates
no slower than at 64² × 1 — in fact marginally faster, which is noise. That
is the whole design intent of the
[timing-from-module-durations pattern](../tutorials/build_a_sequence_plugin.md#3-budget-te-and-tr-from-module-durations):
`validate_protocol` builds the modules once and does arithmetic on their
`duration`, `t_first_echo_s` and `esp`. It never appends a block, so it
never sees the loop that makes the sequence big.

The two slowest rows say what the cost actually is. EPI (157 ms) and FSE
(104 ms) both design more module *types* than a GRE does — an echo train,
diffusion and reference-scan prep for EPI, an echo train for FSE — and
module *construction* is what is being timed. A plugin that did its TE
budget by building the sequence and measuring it would put the Design
column below into this table instead, and the UI would stall for seconds on
every keystroke.

## Download

The operator-visible wait is the whole chain: design, serialise, write, then
parse, convert, cache.

| Sequence | Design | Serialise | Disk | Parse | Convert | Cache write | **Total** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRE 64² × 1 sl | 0.10 s | 0.01 s | 0.4 ms | 4.0 ms | 9.7 ms | 0.7 ms | **0.12 s** |
| GRE 128² × 16 sl | 1.26 s | 0.24 s | 1.3 ms | 94.1 ms | 17.5 ms | 11.2 ms | **1.62 s** |
| GRE 256² × 32 sl | 4.68 s | 0.97 s | 4.6 ms | 383 ms | 39.0 ms | 31.5 ms | **6.11 s** |
| GRE 3D 128² × 64 par | 4.75 s | 1.04 s | 5.1 ms | 382 ms | 40.1 ms | 28.3 ms | **6.25 s** |
| FSE 256² × 20 sl, ETL 16 | 4.21 s | 0.42 s | 3.8 ms | 195 ms | 51.1 ms | 14.6 ms | **4.89 s** |
| EPI 256² × 20 sl, ETL 16 | 0.77 s | 0.21 s | 1.7 ms | 79.8 ms | 10 238 ms | 4.8 ms | **11.30 s** |
| bSSFP 128², 1 frame | 0.11 s | 0.01 s | 0.3 ms | 3.5 ms | 12.2 ms | 1.0 ms | **0.14 s** |
| bSSFP 128², 64 frames | 5.23 s | 0.47 s | 3.4 ms | 239 ms | 25.6 ms | 15.8 ms | **5.98 s** |
| MPRAGE 192² × 128 par | 14.53 s | 3.05 s | 15 ms | 1 191 ms | 124 ms | 89.8 ms | **19.00 s** |
| Spiral, 48 short shots | 1.07 s | 0.07 s | 1.0 ms | 34.1 ms | 116 ms | 2.4 ms | **1.29 s** |
| Spiral, 4 long shots | 0.17 s | 0.01 s | 0.4 ms | 9.5 ms | 52.2 ms | 0.9 ms | **0.24 s** |

Ordinary clinical protocols land between 0.1 s and 6 s. Four observations.

**Python design dominates everything except two cases.** For GRE 256²,
design plus serialisation is 5.6 s of a 6.2 s total — the C library is 9 %
of the wait. Optimising the C side would be optimising the wrong thing for
most of this table.

**MPRAGE's cost is now almost entirely its design.** 14.5 s of design and
3.0 s of serialisation against 124 ms of conversion, for 19.0 s end to end.
Conversion used to be 6.5 s of a 25.9 s total here; it is now under 1 %.
This row still sets the practical ceiling on prescription size, but the
ceiling is Python, not C.

**EPI's convert stage is the one remaining exception, and it is a single
number.** 10.2 s of an 11.3 s total is the C conversion stage alone — 91 %
of the wait, against 80 ms of parsing the same file. See
[Where the C time goes](#where-the-c-time-goes) for why: unlike FSE at the
same nominal size, EPI's design does not split the scan into one TR per
shot, so conversion sees one 6 721-block TR instead of 320 small ones. With
MPRAGE's conversion cost gone, EPI is now the only sequence in the table
where the C library dominates the operator's wait.

**Serialisation is a consistent 5–20 % of design**, and the disk write is
never measurable — under 15 ms even for a 19 MB `.seq`. Writing the file is
not the cost; building it is.

### Where the C time goes

The C half of the download is 14 ms to 10.3 s, against these block counts:

| Sequence | Raw blocks | Unique | Parse | Convert | Cache write | **C total** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRE 64² × 1 sl | 512 | 8 | 4.0 ms | 9.7 ms | 0.7 ms | **14.4 ms** |
| GRE 128² × 16 sl | 14 464 | 8 | 94.1 ms | 17.5 ms | 11.2 ms | **123 ms** |
| GRE 256² × 32 sl | 57 600 | 8 | 383 ms | 39.0 ms | 31.5 ms | **454 ms** |
| GRE 3D 128² × 64 par | 49 280 | 7 | 382 ms | 40.1 ms | 28.3 ms | **451 ms** |
| FSE 256² × 20 sl | 22 080 | 9 | 195 ms | 51.1 ms | 14.6 ms | **261 ms** |
| EPI 256² × 20 sl | 6 721 | 9 | 79.8 ms | 10 238 ms | 4.8 ms | **10 322 ms** |
| bSSFP 128², 1 frame | 259 | 7 | 3.5 ms | 12.2 ms | 1.0 ms | **16.8 ms** |
| bSSFP 128², 64 frames | 16 576 | 7 | 239 ms | 25.6 ms | 15.8 ms | **281 ms** |
| MPRAGE 192² × 128 par | 181 248 | 10 | 1 191 ms | 124 ms | 89.8 ms | **1 405 ms** |
| Spiral, 48 short shots | 3 889 | 7 | 34.1 ms | 116 ms | 2.4 ms | **152 ms** |
| Spiral, 4 long shots | 325 | 7 | 9.5 ms | 52.2 ms | 0.9 ms | **62.6 ms** |

The Unique column — distinct base-block definitions, before per-instance
rotation and amplitude scaling — stays in the single digits across every
sequence and every size. That is
[base-block deduplication](sequence_representation/pulseg.md#deduplication-the-first-thing-conversion-does)
doing its job: a 256² acquisition and a 64² one differ in how many
*instances* of a block get played, not in how many block *definitions*
exist.

**Parse scales with the file; convert scales with the TR.** Parsing is
essentially linear in raw block count — 512 blocks in 3.4 ms, 57 600 in
353 ms, a steady ~6 µs per block. Conversion is not: bSSFP at 16 576 blocks
converts in 6.4 ms while FSE at 22 080 takes 110 ms, ~13× more per block. The
difference is `tr_size` — FSE's TR is 69 blocks, one 16-echo shot; bSSFP's
is 257. TR-period detection compares candidate periods across the imaging
region, so its cost grows with the *period length*, not just the block
count.

**EPI's 6 721-block TR is a single canonical period spanning the whole 640 s
protocol, and that is why it dominates this table.** Every other multi-shot
sequence here designs one TR per shot — FSE's is 69 blocks, repeated for
320 shots — so the TR-period gradient-waveform analysis only ever
interpolates a few tens of milliseconds at a time. EPI's design does not
split its 320 shots into separate TRs; `tr_size` is the entire acquisition.
Interpolating gradient waveforms across a 640 s span at a 5 µs raster is on
the order of a hundred million samples per axis, and that single call is
essentially all of the 4.0 s convert time — and, as the next section shows,
all of its memory too.

**MPRAGE's 6.5 s conversion is the largest in wall-clock terms.** 181 248
raw blocks across 3 072 TRs of a 59-block inversion-prepared segment. It is
the case where the cache earns its keep most clearly (paid once, then
0.14 ms per scan), and, together with EPI, one of the two cases that would
need attention first if prescriptions get larger.

### Memory during conversion

| Sequence | Raw model | Collection (live) | Conversion (peak) |
| --- | ---: | ---: | ---: |
| GRE 64² × 1 sl | 44 KB | 642 KB | 3 MB |
| GRE 128² × 16 sl | 1 MB | 2 MB | 13 MB |
| GRE 256² × 32 sl | 4 MB | 6 MB | 47 MB |
| GRE 3D 128² × 64 par | 4 MB | 6 MB | 53 MB |
| FSE 256² × 20 sl | 2 MB | 9 MB | 30 MB |
| EPI 256² × 20 sl | 887 KB | 1.4 GB | 3.3 GB |
| bSSFP 128², 1 frame | 44 KB | 929 KB | 3 MB |
| bSSFP 128², 64 frames | 3 MB | 3 MB | 33 MB |
| MPRAGE 192² × 128 par | 14 MB | 20 MB | 158 MB |
| Spiral, 48 short shots | 393 KB | 18 MB | 42 MB |
| Spiral, 4 long shots | 73 KB | 7 MB | 17 MB |

Peak is consistently at or above live — 10× for bSSFP, 2.4× for GRE 3D —
because conversion allocates working structures it then releases. Sizing a
host allocator against the live figure would be wrong; the peak is what has
to fit.

The collection is now the same order of magnitude as the raw model it comes
from — MPRAGE 14 MB in, 20 MB live — because only genuinely per-occurrence
data is stored per occurrence. MPRAGE's live collection was 1 176 MB before
the label table and execution stream were fixed; see
[the cache section](#the-cache-is-about-the-size-of-the-seq).

**EPI's 3.3 GB peak is the largest figure on this page, by more than 20×.**
Both raw model and unique-block count are unremarkable — 887 KB and 9
definitions, smaller than FSE's. The 3.3 GB is entirely the
gradient-waveform interpolation from the previous section: one TR spanning
the whole 640 s scan, sampled at 5 µs, on all three axes. A sequence that
split its shots into per-TR chunks the way FSE does would not pay this;
that EPI's design does not is the single fact that explains both its
conversion time and its memory. It is now the only outlier left on this
page, and the clearest candidate for the next piece of work.

## Scan time: the section split

This is the result the cache's section structure exists to produce.

| Sequence | Cache file | Pulsegen load | Scan-loop load | Memory ratio |
| --- | ---: | ---: | ---: | ---: |
| GRE 64² × 1 sl | 53 KB | 10 KB / 0.01 ms | 53 KB / 0.04 ms | 5× |
| GRE 128² × 16 sl | 1 MB | 10 KB / 0.01 ms | 1 MB / 0.86 ms | 125× |
| GRE 256² × 32 sl | 5 MB | 11 KB / 0.02 ms | 5 MB / 3.12 ms | 468× |
| GRE 3D 128² × 64 par | 4 MB | 10 KB / 0.01 ms | 4 MB / 3.07 ms | 474× |
| FSE 256² × 20 sl | 2 MB | 23 KB / 0.02 ms | 2 MB / 1.52 ms | 96× |
| EPI 256² × 20 sl | 805 KB | 14 KB / 0.02 ms | 805 KB / 0.75 ms | 58× |
| bSSFP 128², 1 frame | 66 KB | 32 KB / 0.02 ms | 67 KB / 0.04 ms | 2× |
| bSSFP 128², 64 frames | 2 MB | 32 KB / 0.02 ms | 2 MB / 1.74 ms | 65× |
| MPRAGE 192² × 128 par | 15 MB | 18 KB / 0.02 ms | 15 MB / 12.64 ms | 871× |
| Spiral, 48 short shots | 487 KB | 15 KB / 0.01 ms | 488 KB / 0.27 ms | 33× |
| Spiral, 4 long shots | 82 KB | 43 KB / 0.02 ms | 82 KB / 0.05 ms | 2× |

The pulse-generation load — `COMMON + SHAPES` — never leaves the tens of
kilobytes, across a 330× range of `.seq` size, and never takes more than
0.02 ms. The scan-loop load spans 53 KB to 15 MB over the same range —
including EPI, whose 640 s of scan-loop instructions is still under a
megabyte to load, unlike the gradient-waveform pass it needs at conversion
time.

That flatness is not luck. It is
[base-block deduplication](sequence_representation/pulseg.md#deduplication-the-first-thing-conversion-does)
showing up as a memory figure: what pulse generation needs is bounded by the
number of *distinct waveform shapes*, and a 256² acquisition has barely more
distinct shapes than a 64² one — phase encoding scales a definition rather
than creating one. Everything that does grow with scan length lives in
`INSTANCES`, `ROTATIONS` and `SCANLOOP`, and the pulse-generation pass never
opens them.

MPRAGE is the case that makes it concrete: the phase that runs where memory
is scarcest reads 18 KB, while the whole collection is 15 MB. Without the
split it would read all of it.

The ratio column is worth reading as a *floor* rather than a headline. It
was 50 169× for MPRAGE when the scan-loop side carried a label table
duplicated per TR and four integers per block; shrinking the scan-loop side
by 57× cut the ratio to 871× without pulse generation changing at all. The
split is what makes the pulsegen figure flat; it is not what makes the
scan-loop figure small.

### The cache is about the size of the `.seq`

| Sequence | `.seq` | Cache | Ratio |
| --- | ---: | ---: | ---: |
| GRE 64² × 1 sl | 56 KB | 53 KB | 0.9× |
| GRE 128² × 16 sl | 1 MB | 1 MB | 0.9× |
| GRE 256² × 32 sl | 6 MB | 5 MB | 0.8× |
| GRE 3D 128² × 64 par | 6 MB | 4 MB | 0.8× |
| FSE 256² × 20 sl | 3 MB | 2 MB | 0.7× |
| EPI 256² × 20 sl | 1 MB | 805 KB | 0.7× |
| bSSFP 128², 1 frame | 59 KB | 66 KB | 1.1× |
| bSSFP 128², 64 frames | 4 MB | 2 MB | 0.6× |
| MPRAGE 192² × 128 par | 18 MB | 15 MB | 0.8× |
| Spiral, 48 short shots | 608 KB | 487 KB | 0.8× |
| Spiral, 4 long shots | 165 KB | 82 KB | 0.5× |

The cache is not a compressed `.seq`; it is a *materialised* one. Text that
says "block 7 again, scaled by 0.31" becomes explicit per-instance rows in
`INSTANCES` and `ROTATIONS`, plus a derived trajectory and
frequency-modulation plan. What keeps that from exploding is that only the
things which genuinely vary per occurrence are stored per occurrence.

Every sequence in the table now lands between 0.5× and 1.1× of its `.seq`.
That is a recent result, and MPRAGE is the row that shows why it was not
always so. It used to produce a 882 MB cache from an 18 MB `.seq` — 48× —
from two causes, both since removed. The label table was allocated once per
ADC *per TR* rather than once per ADC, which for 3 072 TRs meant 3 072
identical copies of the same table. And the execution stream stored four
integers for every block of the scan, when what it was encoding was
"positions 0..181 247 play blocks 0..181 247 in order" — one run — plus a
segment order that repeats every TR.

Both are now stored as what they are: the label table once, and the
execution stream as runs plus a single period of the segment order (see
{doc}`sequence_representation/pulseg`). MPRAGE's cache is 15 MB, 0.8× its
`.seq`.

EPI is the reminder that this section is not the whole story: its cache
ratio is an unremarkable 0.7×, because its cost is entirely in
conversion-time working memory (the previous section), not in what
ultimately gets written to disk.

## Safety checks

Gradient amplitude/slew, mechanical resonance and PNS all run at
predownload, all on the host, alongside the download chain above — this
section is their measurement, kept separate from
{doc}`safety/gradient_slew`, {doc}`safety/mechanical_resonance` and
{doc}`safety/pns` so a method and its cost can be read independently.

![Mechanical resonance vs. PNS cost across the zoo](assets/benchmarks/safety_scaling.png)

**Mechanical resonance cost depends on the complexity of one canonical
window, not on sequence duration, TR count or pass count.** Three
quantities matter: $D$, the unique gradient definitions in the window
(amortised to $O(1)$ per occurrence by the `(def_id, frequency)` transform
cache in `pulseg_safety.c`); $M$, the outer repeat count, which costs
nothing extra because the finite-$M$ Dirichlet fold adds a fixed number of
samples per harmonic, not a growing one; and $N$, the materialised instances
of one definition inside the window, which is the coherent-sum length. The
table shows the scaling directly: FSE has 69 blocks in its window and costs
785 ms; GRE has 8 and costs 25 ms. What it does *not* show is any dependence
on scan length — GRE at 64 TRs and GRE at 8192 TRs have the same window and
the same cost. Display-only extras (a dense FFT of the NEX-expanded
waveform, a dense analytic envelope, both used only for the plots in
{doc}`safety/mechanical_resonance`) are never paid for by the gate, which
always runs with `compute_dense_envelope=0`.

**PNS cost tracks TR duration, not sequence complexity**, for the structural
reason given in {doc}`safety/pns`: it is the one analysis that has to
materialise a waveform, sampled at the gradient raster, over the whole
canonical TR.

| Sequence | Canonical TR | Mech-res gate | PNS: waveform build | PNS: nerve model | PNS total |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRE | 250 ms | 25 ms | 707 ms | 16 ms | 723 ms |
| FSE, ETL 16 | 3000 ms | 785 ms | 8 442 ms | 208 ms | 8 650 ms |
| bSSFP | 129 ms | 41 ms | 369 ms | 9 ms | 378 ms |
| MPRAGE | 2163 ms | 360 ms | 6 099 ms | 150 ms | 6 249 ms |
| Spiral, 48 short shots | 960 ms | 377 ms | 2 694 ms | 61 ms | 2 755 ms |
| Spiral, 4 long shots | 80 ms | 83 ms | 236 ms | 5 ms | 241 ms |

Reproduce with `python docs/_bench/bench_safety.py --esp <table>` then
`python docs/_bench/plot_safety_benchmarks.py`.

Three things follow from the table and the plot.

**Sample count is $T_\text{TR}/\Delta t$ and nothing else.** FSE's 3-second
TR costs 12× GRE's 250 ms one for the waveform build, and the ratio of their
slew-sample counts (601 409 vs. 51 427) is 11.7 — a near-exact match.
Compare with the mechanical-resonance column, which tracks the *number of
blocks* in the window instead: the signature of a structural analysis versus
a materialised one.

**The nerve model is 2–3 % of PNS's cost; building the waveform is the
rest.** This is the payoff of the function-pointer split described in
{doc}`safety/pns`: swapping in a different nerve model, or evaluating
several, is nearly free, because the expensive half — expanding block
definitions into a padded, rasterised slew waveform — is done once and
shared.

**PNS is the more expensive gate on every sequence in the corpus**, by a
factor that itself depends on the sequence: closer to 2× for FSE (whose
mechanical-resonance window, 69 blocks, is unusually large) and closer to
30× for GRE and bSSFP (whose windows are small but whose TR still has to be
fully rasterised for PNS). Both gates are paid at predownload while the
operator waits, so which one dominates is worth knowing per sequence family,
not as a single constant.

**A note on what these numbers are not.** The peak stimulation percentages
this sweep produces are not meaningful safety figures — the Irnich constants
used are representative body-gradient values chosen to exercise the code
path, not any particular system's, and the real ones come from scanner
configuration. What the table and plot measure is cost and its scaling.

## Host memory during download

The design pass and the conversion pass both run on the host, one after the
other, so the host's requirement is the larger of the two — and which one
that is depends on the sequence.

| Sequence | Python peak (design) | C peak (convert) | `.seq` |
| --- | ---: | ---: | ---: |
| GRE 64² × 1 sl | 5.8 MB | 2.7 MB | 56 KB |
| GRE 128² × 16 sl | 20 MB | 13 MB | 1.4 MB |
| GRE 256² × 32 sl | 83 MB | 47 MB | 5.8 MB |
| GRE 3D 128² × 64 par | 84 MB | 53 MB | 5.8 MB |
| FSE 256² × 20 sl | 43 MB | 37 MB | 3.3 MB |
| EPI 256² × 20 sl | 16 MB | 3.3 GB | 1.1 MB |
| bSSFP 128², 64 frames | 48 MB | 33 MB | 3.6 MB |
| MPRAGE 192² × 128 par | 262 MB | 158 MB | 18 MB |
| Spiral, 48 short shots | 7.6 MB | 42 MB | 608 KB |

The Python peak counts Python objects only (`tracemalloc`), so it
understates the interpreter's real footprint; the C peak is exact. Python is
the larger side for the Cartesian GRE, bSSFP and FSE rows, by 1.2–2.2×: the
host's requirement there is set by design, not conversion. Spiral flips
that — 42 MB of C against 7.6 MB of Python, because a non-Cartesian
trajectory's per-instance state is built from the gradient waveforms, not
the module tree. MPRAGE and EPI go further still, in different ways:
MPRAGE's design is itself expensive (262 MB of Python objects, against a
158 MB C peak) and its 19.0 s download is the longest wait on this page —
and it is now Python-bound on both time and memory, where the C peak used to
be 1.2 GB. EPI's design is cheap (16 MB) but its C peak is 3.3 GB — the
largest number here by more than 20×, and, per the previous section,
entirely attributable to interpolating gradient waveforms over its 640 s
monolithic TR rather than to anything about the sequence's raw size.

## Reproducing

```bash
bash docs/_bench/build_bench.sh
python docs/_bench/bench_zoo.py --repeats=3
python docs/_bench/plot_benchmarks.py
python docs/_bench/bench_safety.py --esp <vendor table>
python docs/_bench/plot_safety_benchmarks.py
```

`bench_zoo.py --c-only` re-times the C stages against `.seq` files a
previous run produced — the right thing after rebuilding the library, since
the host-side pass is the slow part of the sweep and does not change.
`bench_safety.py` uses a synthetic band table when no vendor table is
provided, so those results are performance data, not a safety claim. See
[`docs/_bench/README.md`](https://github.com/INFN-MRI/pulserver/tree/main/docs/_bench/README.md).
