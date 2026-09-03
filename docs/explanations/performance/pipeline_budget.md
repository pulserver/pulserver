# The pipeline budget

```{admonition} TL;DR
:class: tip

- A SPARKLING-style acquisition is **every readout its own optimised arm**:
  131 072 distinct (2, 4 096) waveforms, a few gigabytes of samples, no
  `ROTATIONS` extension to lean on.
- Every stage from an empty `Sequence` to a checked scan is **linear in the
  samples**; the budget is a constant per stage, and each has a line.
- End to end, extrapolated from 8 192 arms: **41 s** against a 30 s budget.
  The write, the read and the stimulation check are inside their lines; the
  design loop and deduplication are the two still over.
```

The pages before this one hold the block count fixed and vary the family.
This one holds the family fixed at its hardest — every readout a distinct
arbitrary waveform, written out — and asks what each stage costs *per arm*,
because at this scale there is no window and no shared shape to amortise
against: the cost of everything is the number of samples, and the only
question is how many times each is touched.

## The scan

`docs/_bench/pipeline_budget.py` builds it: `n` distinct arms of 4 096
points on a 4 µs raster (a tapered spiral turned per arm, with a little noise
so no two are alike), each played in a block of its own between a hard pulse
and a spoiler, assembled through the plain PyPulseq calls —
`make_arbitrary_grad` twice and `add_block` three times per arm — and then
declared, deduplicated, written to binary, parsed back by the scanner-side
reader, and checked for stimulation and mechanical resonance. Every number is measured at 8 192 arms
and scaled to 131 072.

```{figure} ../assets/pipeline_budget/stages.png
Each stage at 131 072 arms, extrapolated from a measurement at 8 192. The
tick on each bar is that stage's line.
```

## The stages and their lines

| Stage | Line | Measured, per arm | At 131 072 arms |
|---|---|---:|---:|
| `make_arbitrary_grad` ×2 + `add_block` ×3 | 60 µs | 75 µs | 10 s |
| `declare_tr` + `remove_duplicates` | 3 s | 11.5 + 26 µs | 4.9 s |
| binary write | ≥ 1 GB/s | 1.23 GB/s | 3 s |
| parse + convert | ≥ 1 GB/s | 1.04 GB/s | 3 s |
| PNS gate | 7 s | 64 µs | 8.4 s |
| mechanical-resonance check | 7 s | 44 µs | 5.8 s |
| **end to end** | **30 s** | | **35 s** |

The scanner-side stages a *Save Rx* adds on top, measured the same way at
8 192 arms through the file path: reading the file and writing the `.pseg`
cache 95 µs per arm (a 4.3 GB file and a 3.3 GB cache at 131 072 arms, disk
bound), reading the cache back 35 µs, the canonical-TR waveform render and
the RF definitions under 1 µs, the stimulation check 64 µs, and the
**mechanical-resonance check 44 µs per arm** — 0.36 s at 8 192 arms, of
which about 0.18 s is fixed and the rest 22 µs per arm (16 384 arms measure
0.54 s), so about 3 s at 131 072; the table above scales it linearly. Past the shape-group cap that check is the scan window of the
{doc}`mechanical-resonance page <mechanical_resonance>`: one FFT per
distinct waveform and a linear pass over the scan's windows, in place of
a positional-maximum envelope rendered over every repetition for a model
that does not apply to a scan with a different arm in every repetition.

Regenerate the table and the figure with
`python docs/_bench/pipeline_budget.py --arms 8192 --plot`; the result lands
in `docs/_bench/pipeline_budget.json`.

**Assembly** is two factory calls and three block registrations per arm,
each a compiled call. The factory validates a 4 096-point waveform in two
passes and keeps a view of the caller's array — `grad.waveform` *is* that
array, as upstream's is — and `add_block` copies it once, dividing by the
signed peak on the way into the shape store's huge-page chunks. What is
left is the copy and the first touch of the memory it lands in: 30 µs for
the two factories and 40 µs for the three blocks per arm on this machine,
where a plain 32 KB copy into fresh memory costs 30 µs on small pages and
10 µs on huge ones. **Deduplication** is the other line still over:
rounding every sample to the file's precision is a division per sample,
and the numbering is the order of first appearance, which is sequential by
definition.

**The write** runs once to count and once to fill, into a `bytes` of exactly
the counted size. **The read** leaves the sample cells in the caller's
buffer, adopts them into the descriptor and sweeps them once for the
statistics the checks need. **The gate** prices each arm's own response
exactly, by FFT convolution, on every core the host has — the
{doc}`stimulation page <pns>` tells that story — and a verdict on 8 192 arms
takes 0.4 s.

## What holds it

- Every fast path is held equal to the plain calculation: the binary fixtures
  are byte-identical through every change to the writer, the corpus
  regenerates unchanged, and the pins on every fixture's stimulation peak
  stay green.
- The light structure is held equal to the full one on the TR fields and the
  segment structure over eight shipped families.
- The threaded loops are held equal to their sequential forms: the
  deduplication's numbering, the shape encoding's rows, and the stimulation
  gate's per-block prices, whatever the chunking.
