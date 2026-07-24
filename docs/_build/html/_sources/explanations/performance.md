# Runtime and memory

This page is about the on-scanner path: what the operator waits for, and what
has to fit in target memory. That includes the sequence plugin itself — in
server mode the scanner runs `validate_protocol` and `make_sequence` on its
own host, so the Python design code is inside the budget, not beside it. What
is genuinely offline, and therefore not measured here for its own sake, is the
authoring-time inspection API: `Sequence.pns`, `Sequence.grad_spectrum` and
the plotting entry points exist to be looked at while writing a sequence, and
nothing waits on them.

Everything below was measured with
[`docs/_bench/`](https://github.com/INFN-MRI/pulserver/tree/main/docs/_bench)
on the shipped example plugins. Timings are the minimum of three runs. Heap
figures are live bytes attributable to the library, measured by interposing
the allocator (`-Wl,--wrap=malloc`), so they cover every allocation path and
include allocator rounding — what the process holds, not what it asked for.

Absolute numbers are from one desktop machine and a scanner's host will differ.
The *ratios* and the *scaling* are the point, and those are properties of the
design.

## The three moments that matter

A sequence is touched at three very different times, and conflating them hides
the only interesting results on this page.

**Protocol edit, on the host, on every keystroke.** The UI calls
`validate_protocol` each time a parameter changes. This is an interactive
latency budget: tens of milliseconds, or the UI feels broken.

**Download, on the host, once per prescription.** The plugin designs the
sequence and writes a `.seq`; the C library parses it, converts it to the
intermediate representation, runs the safety checks, and writes the cache. The
operator is waiting on all of it. The host is a general-purpose computer, so
seconds matter here and memory mostly does not.

**Scan, on the target, once per scan.** Load the cache and play it. Here
memory is the scarce resource, not seconds.

The cache exists to separate the last from the first two, and its
[section structure](sequence_representation.md#the-cache-and-why-it-has-sections)
exists so the target never pays for what only the host needs.

## Protocol edit: `validate_protocol`

| Sequence | `validate_protocol` |
| --- | ---: |
| GRE 64² × 1 sl | 41 ms |
| GRE 128² × 16 sl | 43 ms |
| GRE 256² × 32 sl | 35 ms |
| GRE 3D 128² × 64 par | 34 ms |
| FSE 256² × 20 sl, ETL 16 | 112 ms |
| bSSFP 128², 1 frame | 23 ms |
| bSSFP 128², 64 frames | 22 ms |
| MPRAGE 192² × 128 par | 85 ms |
| Spiral, 48 short shots | 33 ms |
| Spiral, 4 long shots | 45 ms |

**22–112 ms, and flat in sequence size.** GRE at 256² × 32 slices validates no
slower than at 64² × 1 — in fact marginally faster, which is noise. That is
the whole design intent of the
[timing-from-module-durations pattern](../tutorials/build_a_sequence_plugin.md#3-budget-te-and-tr-from-module-durations):
`validate_protocol` builds the modules once and does arithmetic on their
`duration`, `t_first_echo_s` and `esp`. It never appends a block, so it never
sees the loop that makes the sequence big.

The two slowest rows say what the cost actually is. FSE (112 ms) and MPRAGE
(85 ms) both design more modules than a GRE does — an echo train, an inversion
preparation — and module *construction* is what is being timed. A plugin that
did its TE budget by building the sequence and measuring it would put the
Design column below into this table instead, and the UI would stall for
seconds on every keystroke.

## Download

The operator-visible wait is the whole chain: design, serialise, write, then
parse, convert, cache.

| Sequence | Design | Serialise | Disk | Parse | Convert | Cache write | **Total** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRE 64² × 1 sl | 0.08 s | 0.01 s | 0.5 ms | 3 ms | 10 ms | 0.4 ms | **0.11 s** |
| GRE 128² × 16 sl | 1.17 s | 0.23 s | 1.6 ms | 87 ms | 49 ms | 8 ms | **1.55 s** |
| GRE 256² × 32 sl | 4.42 s | 0.93 s | 4.4 ms | 343 ms | 344 ms | 41 ms | **6.08 s** |
| GRE 3D 128² × 64 par | 4.58 s | 1.03 s | 5.1 ms | 359 ms | 177 ms | 32 ms | **6.19 s** |
| FSE 256² × 20 sl, ETL 16 | 3.87 s | 0.42 s | 2.1 ms | 172 ms | 628 ms | 13 ms | **5.11 s** |
| bSSFP 128², 1 frame | 0.10 s | 0.01 s | 0.3 ms | 4 ms | 11 ms | 0.3 ms | **0.13 s** |
| bSSFP 128², 64 frames | 4.99 s | 0.47 s | 3.6 ms | 216 ms | 26 ms | 9 ms | **5.71 s** |
| MPRAGE 192² × 128 par | 13.71 s | 2.91 s | 15 ms | 1 055 ms | 11 405 ms | 716 ms | **29.80 s** |
| Spiral, 48 short shots | 1.05 s | 0.07 s | 0.8 ms | 33 ms | 99 ms | 2 ms | **1.25 s** |
| Spiral, 4 long shots | 0.18 s | 0.01 s | 0.3 ms | 9 ms | 46 ms | 0.3 ms | **0.25 s** |

Ordinary clinical protocols land between 0.1 s and 6 s. Three observations.

**Python design dominates everything except the largest case.** For GRE 256²,
design plus serialisation is 5.4 s of a 6.1 s total — the C library is 12 % of
the wait. Optimising the C side would be optimising the wrong thing for most
of this table.

**MPRAGE is the exception, and both halves are expensive.** 13.7 s of design
and 11.4 s of conversion, for 29.8 s end to end. That is the row that sets the
practical ceiling on prescription size today, and it is worth knowing that
neither half alone explains it.

**Serialisation is a consistent 10–20 % of design**, and the disk write is
never measurable — under 15 ms even for a 19 MB `.seq`. Writing the file is
not the cost; building it is.

### Where the C time goes

The C half of the download is 14 ms to 13 s, against these block counts:

| Sequence | Raw blocks | Unique | Parse | Convert | Cache write | **C total** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRE 64² × 1 sl | 512 | 34 | 3 ms | 10 ms | 0.4 ms | **14 ms** |
| GRE 128² × 16 sl | 14 464 | 48 | 87 ms | 49 ms | 8 ms | **144 ms** |
| GRE 256² × 32 sl | 57 600 | 54 | 343 ms | 344 ms | 41 ms | **728 ms** |
| GRE 3D 128² × 64 par | 49 280 | 803 | 359 ms | 177 ms | 32 ms | **568 ms** |
| FSE 256² × 20 sl, ETL 16 | 22 080 | 105 | 172 ms | 628 ms | 13 ms | **813 ms** |
| bSSFP 128², 1 frame | 259 | 43 | 4 ms | 11 ms | 0.3 ms | **15 ms** |
| bSSFP 128², 64 frames | 16 576 | 43 | 216 ms | 26 ms | 9 ms | **250 ms** |
| MPRAGE 192² × 128 par | 181 248 | 1 112 | 1 055 ms | 11 405 ms | 716 ms | **13 176 ms** |
| Spiral, 48 short shots | 3 889 | 7 | 33 ms | 99 ms | 2 ms | **133 ms** |
| Spiral, 4 long shots | 325 | 7 | 9 ms | 46 ms | 0.3 ms | **55 ms** |

**Parse scales with the file; convert scales with the TR.** Parsing is
essentially linear in raw block count — 512 blocks in 3 ms, 57 600 in 343 ms,
a steady ~6 µs per block. Conversion is not: bSSFP at 16 576 blocks converts
in 26 ms while FSE at 22 080 takes 628 ms, 24× more per block. The difference
is `tr_size`. FSE's TR is 1 104 blocks (a 16-echo train across 20 slices);
bSSFP's is 257. TR-period detection compares candidate periods across the
imaging region, so its cost grows with the *period length*, not just the block
count. A sequence with a very long canonical TR pays for it here.

**MPRAGE's 11.4 s conversion is the one number to watch.** 181 248 raw blocks
and 1 112 unique definitions — 3 072 TRs of a 59-block inversion-prepared
segment. It is the case where the cache earns its keep most clearly (paid
once, then 0.14 ms per scan), and the case that would need attention first if
prescriptions get larger.

### Memory during conversion

| Sequence | Raw model | Collection (live) | Conversion (peak) |
| --- | ---: | ---: | ---: |
| GRE 64² × 1 sl | 44 KB | 727 KB | 2.7 MB |
| GRE 128² × 16 sl | 1.1 MB | 6.2 MB | 13 MB |
| GRE 256² × 32 sl | 4.5 MB | 40 MB | 48 MB |
| GRE 3D 128² × 64 par | 4.5 MB | 23 MB | 55 MB |
| FSE 256² × 20 sl | 2.3 MB | 117 MB | 267 MB |
| bSSFP 128², 64 frames | 2.6 MB | 3.2 MB | 34 MB |
| MPRAGE 192² × 128 par | 14 MB | 1 204 MB | 1 218 MB |
| Spiral, 48 short shots | 393 KB | 19 MB | 43 MB |

Peak is consistently above live — 2× for FSE, 10× for bSSFP — because
conversion allocates working structures it then releases. Sizing a host
allocator against the live figure would be wrong; the peak is what has to fit.

The raw parsed model is small next to the collection it produces, which is the
right way round: the expensive object is the one that carries per-instance
state for every block of the scan.

## Scan time: the section split

This is the result the cache's section structure exists to produce.

| Sequence | Cache file | Pulsegen load | Scan-loop load | Memory ratio |
| --- | ---: | ---: | ---: | ---: |
| GRE 64² × 1 sl | 121 KB | 23 KB / 0.03 ms | 122 KB / 0.05 ms | 5× |
| GRE 128² × 16 sl | 4.4 MB | 30 KB / 0.03 ms | 4.4 MB / 1.0 ms | 152× |
| GRE 256² × 32 sl | 30 MB | 33 KB / 0.04 ms | 30 MB / 6.9 ms | 926× |
| GRE 3D 128² × 64 par | 17 MB | 41 KB / 0.10 ms | 17 MB / 5.0 ms | 426× |
| FSE 256² × 20 sl | 3.8 MB | 162 KB / 0.06 ms | 3.8 MB / 1.6 ms | 24× |
| bSSFP 128², 64 frames | 2.2 MB | 41 KB / 0.03 ms | 2.2 MB / 1.6 ms | 55× |
| MPRAGE 192² × 128 par | 882 MB | **70 KB / 0.14 ms** | **882 MB / 476 ms** | **12 957×** |
| Spiral, 48 short shots | 547 KB | 15 KB / 0.02 ms | 535 KB / 0.28 ms | 37× |
| Spiral, 4 long shots | 87 KB | 43 KB / 0.02 ms | 87 KB / 0.04 ms | 2× |

The pulse-generation load — `COMMON + SHAPES` — never leaves the tens-to-low-
hundreds of kilobytes, across a 350× range of sequence size, and never takes
more than 0.14 ms. The scan-loop load spans 87 KB to 882 MB over the same
range.

That flatness is not luck. It is the
[base-block deduplication](sequence_representation.md#deduplication-the-first-thing-conversion-does)
showing up as a memory figure: what pulse generation needs is bounded by the
number of *distinct waveform shapes*, and a 256² acquisition has barely more
distinct shapes than a 64² one — phase encoding scales a definition rather
than creating one. Everything that does grow with scan length lives in
`INSTANCES`, `ROTATIONS` and `SCANLOOP`, and the pulse-generation pass never
opens them.

MPRAGE is the case that makes it concrete: the phase that runs where memory is
scarcest reads 70 KB, while the whole collection is 882 MB. Without the split
it would read all of it.

### The cache is larger than the `.seq`

| Sequence | `.seq` | Cache | Ratio |
| --- | ---: | ---: | ---: |
| GRE 128² × 16 sl | 1.5 MB | 4.7 MB | 3.1× |
| GRE 256² × 32 sl | 6.1 MB | 31 MB | 5.1× |
| FSE 256² × 20 sl | 3.5 MB | 3.9 MB | 1.1× |
| bSSFP 128², 64 frames | 3.8 MB | 2.3 MB | 0.6× |
| MPRAGE 192² × 128 par | 19 MB | 925 MB | 48× |
| Spiral, 4 long shots | 169 KB | 89 KB | 0.5× |

The cache is not a compressed `.seq`; it is a *materialised* one. Text that
says "block 7 again, scaled by 0.31" becomes explicit per-instance rows in
`INSTANCES` and `ROTATIONS`, plus a derived trajectory and frequency-modulation
plan. Sequences whose text is already close to fully enumerated (bSSFP frames,
a short spiral) come out smaller; ones whose text is compact but whose
expansion is not — MPRAGE's 3 072 TRs — come out much larger.

925 MB of cache for a 19 MB `.seq` is worth flagging as a real constraint, not
a curiosity. It is the same MPRAGE row that costs 11 s to convert, and the two
have one cause: per-instance state for 181 248 blocks.

## Safety checks

The safety analyses run at predownload too, and are covered where they are
explained:
[mechanical resonance](mechanical_resonance_safety.md#computational-efficiency)
costs 25–800 ms and scales with the number of blocks in one canonical TR;
[PNS](pns_safety.md#what-this-costs-and-on-what) costs roughly 10× that and
scales with the TR's *duration*, because it is the one analysis that has to
materialise a waveform.

## Host memory during download

The design pass and the conversion pass both run on the host, one after the
other, so the host's requirement is the larger of the two — and which one that
is depends on the sequence.

| Sequence | Python peak (design) | C peak (convert) | `.seq` |
| --- | ---: | ---: | ---: |
| GRE 64² × 1 sl | 6 MB | 2.7 MB | 58 KB |
| GRE 128² × 16 sl | 21 MB | 13 MB | 1.5 MB |
| GRE 256² × 32 sl | 86 MB | 48 MB | 6.1 MB |
| GRE 3D 128² × 64 par | 87 MB | 55 MB | 6.1 MB |
| FSE 256² × 20 sl | 44 MB | 267 MB | 3.5 MB |
| bSSFP 128², 64 frames | 50 MB | 34 MB | 3.8 MB |
| MPRAGE 192² × 128 par | 273 MB | 1 218 MB | 19 MB |
| Spiral, 48 short shots | 8 MB | 43 MB | 623 KB |

The Python peak counts Python objects only (`tracemalloc`), so it understates
the interpreter's real footprint; the C peak is exact. For the Cartesian GRE
rows the Python side is larger, and for the two rows with the most per-instance
state — FSE and MPRAGE — the C side is several times larger. MPRAGE needs
about 1.2 GB at its worst moment, which together with its 30 s download is what
makes it the case worth watching.

## Reproducing

```bash
bash docs/_bench/build_bench.sh
python docs/_bench/bench_zoo.py --repeats=3
python docs/_bench/bench_safety.py --esp <vendor table>
```

`bench_zoo.py --c-only` re-times the C stages against `.seq` files a previous
run produced — the right thing after rebuilding the library, since the
host-side pass is the slow part of the sweep and does not change. See
[`docs/_bench/README.md`](https://github.com/INFN-MRI/pulserver/tree/main/docs/_bench/README.md).
