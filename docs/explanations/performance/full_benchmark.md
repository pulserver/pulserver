# Full benchmark

```{admonition} TL;DR
:class: tip

- Every shipped plugin, at four prescribable sizes each, through the two entry
  points a console calls, with nothing separated out.
- Two clocks: **`validate_protocol()`** on every parameter the operator touches,
  and **one press of *Save Rx*** end to end in one process.
- The column that explains most of the rest is the **TR window** — the duration
  of the structural repeating unit every gradient-side check runs on.
```

The pages before this one isolate one stage at a time. This one does the
opposite.

`validate_protocol()`
: Runs on every parameter the operator touches. It re-derives the sequence's
  timing far enough to answer *is this feasible, and how long will it take*,
  and the console waits for it before it will redraw. This is the number that
  decides whether the UI feels immediate.

*Save Rx*
: Everything one press costs, end to end and in one process:
  `make_sequence` — the design loop, deduplication, and the binary write — then
  the interpreter's `pulseg_read`, which parses, converts and writes the binary
  cache beside the file, then `pulseg_check_safety` over the canonical TR. Its
  peak resident set size is the memory the scanner host has to find, and the
  two files it leaves behind are what the host has to store.

## Every family, at its largest

Each family at the largest protocol it can be prescribed at. "TR window" is the
duration of the structural repeating unit — the window every gradient-side
check runs on — and it is the column that explains most of the rest.

| Family | Largest protocol | Blocks | TR window | `validate_protocol` | Save Rx | Peak RSS | `.seq` | Cache |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gre2D` | 512 Nx, 512 Ny, 24 slices | 76 032 | 21 ms | 13 ms | 0.2 s | 0.17 GB | 2.7 MB | 4.7 MB |
| `gre3D` | 512 Nx, 512 Ny, 512 slices | 823 692 | 14 ms | 37 ms | 2.4 s | 0.50 GB | 30 MB | 52 MB |
| `se2D` | 512 Nx, 512 Ny, 20 slices, 20 TE ms | 92 160 | 50 ms | 12 ms | 0.2 s | 0.17 GB | 3.2 MB | 5.7 MB |
| `se3D` | 256 Nx, 256 Ny, 256 slices | 411 448 | 100 ms | 16 ms | 0.7 s | 0.29 GB | 14 MB | 25 MB |
| `fse2D` | 512 Nx, 512 Ny, 16 slices | 35 840 | 250 ms | 10 ms | 0.1 s | 0.16 GB | 1.3 MB | 2.3 MB |
| `fse3D` | 192 Nx, 192 Ny, 96 slices | 59 664 | 1 000 ms | 25 ms | 0.4 s | 0.18 GB | 2.2 MB | 3.8 MB |
| `epi2D` | 128 Nx, 128 Ny, 60 slices | 31 800 | 189 ms | 8 ms | 0.1 s | 0.16 GB | 1.1 MB | 2.1 MB |
| `epi3D` | 128 Nx, 128 Ny, 64 slices | 8 783 | 189 ms | 9 ms | 0.1 s | 0.16 GB | 0.3 MB | 0.7 MB |
| `bssfp2D` | 256 Nx, 256 Ny, 60 slices | 48 060 | 1 301 ms | 18 ms | 0.3 s | 0.16 GB | 1.8 MB | 3.1 MB |
| `bssfp3D` | 128 Nx, 128 Ny, 48 slices | 14 454 | 14 067 ms | 18 ms | 2.6 s | 0.20 GB | 0.6 MB | 1.1 MB |
| `gre_multiecho2D` | 512 Nx, 512 Ny, 40 slices | 232 320 | 50 ms | 7 ms | 0.7 s | 0.24 GB | 8.9 MB | 15 MB |
| `gre_multiecho3D` | 512 Nx, 512 Ny, 512 slices | 2 059 230 | 35 ms | 37 ms | 6.5 s | 1.02 GB | 79 MB | 133 MB |
| `mprage3D` | 512 Nx, 1024 Ny, 512 slices | 1 673 100 | 2 000 ms | 334 ms | 6.3 s | 1.12 GB | 60 MB | 105 MB |
| `mprage_stack_of_spirals3D` | 128 Nx, 192 slices, 384 arms | 297 220 | 8 000 ms | 11 ms | 4.6 s | 1.10 GB | 13 MB | 19 MB |
| `gre_radial2D` | 512 Nx, 40 slices, 805 spokes | 131 360 | 20 ms | 9 ms | 0.5 s | 0.21 GB | 5.4 MB | 8.3 MB |
| `gre_stack_of_stars3D` | 256 Nx, 256 slices, 805 spokes | 824 448 | 10 ms | 7 ms | 3.7 s | 0.60 GB | 33 MB | 52 MB |
| `gre_spiral2D` | 256 Nx, 60 slices, 64 arms | 24 000 | 20 ms | 15 ms | 0.1 s | 0.91 GB | 1.0 MB | 1.5 MB |
| `gre_stack_of_spirals3D` | 128 Nx, 256 slices, 64 arms | 65 664 | 6 ms | 9 ms | 0.3 s | 0.94 GB | 2.7 MB | 4.2 MB |
| `se_propeller2D` | 256 Nx, 60 slices, 32 blade width, 32 blades, 120 TE ms | 76 800 | 167 ms | 10 ms | 0.3 s | 0.19 GB | 2.5 MB | 5.4 MB |
| `zte3D` | 64 Nx | 26 002 | 8 527 ms | 18 ms | 11.1 s | 0.20 GB | 1.3 MB | 4.9 MB |

Two families have no repeating unit shorter than the scan, so their window is
the whole acquisition: a 3D balanced train is one continuous steady state, and a
ZTE shell is traversed once. Both are capped at the size above which the
interpreter refuses the file with *"no periodic TR pattern found"*.

## Method

A size is a *protocol*: the plugin's own default with the prescribed quantities
overridden, exactly as the console would send it. Every case runs in its own
subprocess, so a peak RSS is that case's rather than the sweep's high-water
mark, and the reported `validate_protocol` time is the fastest of seven calls in
a warm process — the state a scanner-side plugin server is in.

System limits are the defaults, 40 mT/m and 170 T/m/s. The safety gate gets two
forbidden bands at 550–650 and 1100–1250 Hz, with the amplitude limit and the
PNS ceiling left wide open so every case runs the whole check instead of
returning early on a verdict. The gate's *cost* is what this page reports; its
*verdicts* are {doc}`../safety/index`.

A 3D Cartesian scan reaches one to two million blocks; a multi-slice 2D scan is
tens of thousands however high its resolution, because what it has is slices
rather than partitions.

In every chart the x axis is scan size in blocks, each family has its own colour
and marker, and a guide is fitted through that family's four points — in log
space where both axes are logarithmic, so slope one is a cost proportional to
the scan.

## Protocol validation

```{figure} ../assets/full_benchmark/validate.png
`validate_protocol` runtime against scan size, every shipped family. This is
the call the console makes on each protocol edit.
```

Flat, and in the tens of milliseconds: the call designs *one repetition*, so
what it costs is a property of the family rather than of the prescription.
Eighteen of the twenty stay between 6 and 25 ms across four orders of magnitude
of scan size.

The exception is the largest MPRAGE, at 334 ms. What scales there is the
*encoding plan*: an inversion-prepared 512 × 1024 × 512 acquisition ranks and
deals half a million views into echo trains, and the ordering is the answer to
"how many shots, therefore how long". At 256-cubed the same call is about 40 ms.

## Save Rx, end to end

```{figure} ../assets/full_benchmark/save_rx.png
What one press of *Save Rx* costs, split into design, conversion, safety and
cache write, against scan size.
```

The band running up the middle is what a design loop costs: a few microseconds
per block, so a two-million-block scan is a few seconds and everything smaller
is proportionally less. Conversion and the cache ride along at about a tenth of
it.

What lifts a family out of that band is never the block count. It is the
**TR window**, because the acoustic check evaluates the drive spectrum at the
TR harmonics inside the forbidden bands, and there are as many of those as the
window is long. Reading the extremes off the table:

| | Blocks | TR window | Design | Convert + cache | Safety |
|---|---:|---:|---:|---:|---:|
| `gre_multiecho3D` | 2 059 230 | 40 ms | 5.81 s | 0.55 s | 0.18 s |
| `mprage3D` | 1 673 100 | 2.0 s | 4.93 s | 0.40 s | 0.92 s |
| `gre_stack_of_stars3D` | 824 448 | 10 ms | 3.13 s | 0.34 s | 0.21 s |
| `mprage_stack_of_spirals3D` | 297 220 | 8.0 s | 1.27 s | 0.09 s | 3.27 s |
| `bssfp3D` | 14 454 | 14.1 s | 0.25 s | 0.01 s | 2.34 s |
| `zte3D` | 26 002 | 8.5 s | 0.12 s | 0.01 s | 10.95 s |

Two million blocks over a 40 ms window is checked in a fifth of a second;
twenty-six thousand blocks over an eight-second window take fifty times as
long. ZTE is the worst of both, because there the number of distinct waveforms
grows with the scan as well — every shot is its own shell — and the gate costs
the product of the two.

## Where the time goes inside each half

*Save Rx* is two halves — building the sequence, then reading it back — and the
table above gives each as one number.

```{figure} ../assets/full_benchmark/time_split.png
The two halves of one download, by stage, on two protocols of the same size:
a Cartesian MPRAGE and a stack-of-spirals MPRAGE, both 2 103 300 blocks at 512
partitions and 1024 views per inversion train. The spiral is drawn in the
encoding the manual recommends — one arm stored and turned per shot by a
`ROTATIONS` extension.
```

**On the design side the loop dominates, on both protocols.** Two thirds to
three quarters is the plugin's own `add_block` calls; `TransformFOV` is the next
sixth; deduplication is a tenth and pays for itself several times over
downstream; and the binary write, on a hundred-megabyte file, is two or three
percent. There is no serialisation problem to solve — the cost is the number of
blocks, spent before anything is written.

**On the interpreter side the safety gate is the whole story.** It is 79 % of
that half on the Cartesian protocol and 64 % on the spiral one, against 8–13 %
for the parse and 12–21 % for the cache write. Everything the
{doc}`conversion` page measures — the parse, structure detection, the cache —
happens in the shadow of the check that decides whether the scan is allowed to
run at all.

That is also where the two protocols separate, and not in the direction the
block count suggests. Same size, same window, and the gate costs **8.7 s** on
the Cartesian scan against **3.6 s** on the spiral: what it is paid per is the
number of distinct waveforms and the harmonics inside the guarded bands, not the
number of blocks. Written out arm by arm instead of turned, the same spiral scan
costs **84.9 s** — a twenty-four-fold penalty for an encoding choice that leaves
the images identical, and the sharpest argument this manual has for the
`ROTATIONS` extension.

**Structure detection is one to two percent of that half.** Finding the period,
partitioning it and checking the result together cost tens of milliseconds on
two million blocks, because detection runs on normalised block identities the
conversion had to compute anyway.

Both halves read and write the **binary** form. The same file in text parses in
about twice the time, and nothing on the scanner path needs it.

## Peak memory

```{figure} ../assets/full_benchmark/peak_rss.png
Peak resident set size against scan size — the figure that decides whether a
protocol fits on the scanner at all.
```

A floor of about 0.16 GB — the Python interpreter and a plugin's imports, not
the sequence — and two things above it. Building the scan costs roughly half a
gigabyte per two million blocks. Designing a spiral costs about 0.75 GB flat,
whatever the scan: the three spiral families sit on their own shelf from their
smallest protocol onwards, because what allocates is the arm solver, once.

Nothing here approaches what a scanner host has. It is the figure that would
stop being uninteresting first if a design loop started holding per-block
objects.

## The two files a download produces

```{figure} ../assets/full_benchmark/seq_size.png
Binary sequence file size against scan size: a row per block, plus a library
that does not grow with the scan.
```

```{figure} ../assets/full_benchmark/cache_size.png
Interpreter cache size against scan size, the sidecar the scanner reads in
place of re-deriving the structure.
```

One line each, for every family: **37 bytes per block** written and **63 bytes
per block** cached, whatever the sequence is. That is the representation doing
its job — a block is a row of indices into libraries that hold each waveform
once, so the file is the block table and the block table is a row per block.
The offset at the left is the fixed header, which stops mattering above a few
thousand blocks.

ZTE is the one family off the line, and only in the cache: its shots are
distinct waveforms rather than references to one, so the shape library it stores
is comparable to the scan itself.

The cache is larger than the file it derives from, which is the trade
{doc}`conversion` describes: resolved tables cost bytes and save a parse.

