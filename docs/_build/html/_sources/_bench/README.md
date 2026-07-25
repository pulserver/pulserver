# Benchmark tooling

Documentation-only. Produces the numbers and figures quoted in
[`docs/explanations/benchmarks.md`](../explanations/benchmarks.md),
[`safety/mechanical_resonance.md`](../explanations/safety/mechanical_resonance.md),
[`safety/pns.md`](../explanations/safety/pns.md) and
[`sequence_representation/pulseg.md`](../explanations/sequence_representation/pulseg.md).
Nothing here is part of the shipped package, and nothing in the package
imports it.

## Contents

| File | What it measures |
| --- | --- |
| `bench_pipeline.c` | The on-scanner C pipeline, stage by stage: parse, convert, cache write, and the three cache loads. Wall time and exact heap. |
| `bench_alloc.c` | Allocator interposition (`-Wl,--wrap=malloc`) so the heap figures cover every allocation path the library uses, not just one. |
| `bench_zoo.py` | Drives the example plugins at several protocol sizes, times the host-side design and serialisation, then runs `bench_pipeline` on each `.seq`. |
| `bench_safety.py` | Mechanical resonance and PNS analysis across the zoo, through the same bindings the scanner path uses. |
| `segment_plots.py` | C-library-inferred maximum-energy segment views for the zoo, including multi-segment cases (MPRAGE, fat-sat EPI). |
| `waveform_plots.py` | Representative-TR and PNS (chronaxie + SAFE) plots for the safety pages. |
| `plot_benchmarks.py` | Reproducible download/load scaling plot from `results.json`. |
| `plot_safety_benchmarks.py` | Reproducible mechanical-resonance-vs-PNS cost plot from `safety.json`. |
| `seqdesc_diagram.py` | Schematic (non-data) diagram of the SEQDESC event-stream state machine. |

## Running

```bash
bash docs/_bench/build_bench.sh                    # builds bench_pipeline
python docs/_bench/bench_zoo.py --repeats=3        # -> results.json
python docs/_bench/plot_benchmarks.py              # -> explanations/assets/benchmarks/
python docs/_bench/bench_safety.py --esp <table>   # -> safety.json
python docs/_bench/plot_safety_benchmarks.py       # -> explanations/assets/benchmarks/
python docs/_bench/segment_plots.py                # -> explanations/assets/segments/
python docs/_bench/waveform_plots.py               # -> explanations/assets/{representative_tr,pns_safety}/
python docs/_bench/seqdesc_diagram.py              # -> explanations/assets/reconstruction/
```

`bench_zoo.py --c-only` re-times the C stages against `.seq` files a previous
run already produced, which is what you want after rebuilding the library: the
host-side design pass is the slow part of the sweep and does not change.

`bench_safety.py` without `--esp` uses a small synthetic band table, so it
runs without a vendor lockout file. No vendor table of any kind ships with
Pulserver.

Generated artefacts (`work/`, `results.json`, `safety.json`, the built
binary) are gitignored.

## Reading the numbers

Timings are the **minimum** of the repeat count, not the mean: scheduler noise
only ever adds, so the minimum is the better estimate of the work actually
required.

Heap figures from `bench_pipeline` are live bytes attributable to the library,
with allocator rounding included — what the process holds, not what it asked
for. The figure reported after a cache load is measured with only that
collection live, so the three loads are directly comparable.

Python memory is `tracemalloc`'s peak, which counts Python objects only.
It is measured in a pass of its own, because `tracemalloc` hooks every
allocation and inflates the timings it would otherwise be gathered alongside.
