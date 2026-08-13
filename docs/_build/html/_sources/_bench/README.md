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
| `bench_pipeline.c` | The on-scanner C pipeline, stage by stage: parse, convert, cache write, headless safety, and the three cache loads. Wall time and exact heap. |
| `bench_alloc.c` | Allocator interposition (`-Wl,--wrap=malloc`) so the heap figures cover every allocation path the library uses, not just one. |
| `bench_zoo.py` | Drives the example plugins at several protocol sizes, times the host-side design and serialisation, then runs `bench_pipeline` on each `.seq`. |
| `bench_scale.py` | The other end of the range: millions of blocks, reporting design cost *per block*, peak resident growth, and how many library entries the design pass still holds. Two families — each plugin at its largest *validating* protocol, and the design modules driven straight past those caps (512×1024×512 MPRAGE at ETL 1024, echo trains at 128/256, EPI, bSSFP, spiral and ZTE at pushed resolution). |
| `gate_bytes.py` | **Not a measurement — a gate.** SHA of every zoo plugin's `.seq` payload at 35 protocols, diffed against a committed baseline. Registration changes must not move a single byte. |
| `bench_safety.py` | Mechanical resonance and PNS analysis across the zoo, through the same bindings the scanner path uses. |
| `bench_recon_physics.py` | Direct versus compact Toeplitz normal-operator runtime and RAM/VRAM across Cartesian/non-Cartesian, subspace, and off-resonance cases. |
| `bench_cuda_streaming.py` | Host-backed one/two-stream CUDA Toeplitz execution for a representative 256³, five-coefficient volume. |
| `bench_cuda_streaming_sense.py` | Complete eight-coil streamed or full-offload subspace SENSE Toeplitz normal. |
| `profile_cuda_toeplitz.py` | Torch CPU/CUDA operator and memory profile for one packed streamed normal. |
| `cuda_streaming_sense_256_results.md` | Coil-aware eight-channel 256³ streamed-normal measurement and CPU/full-GPU scaling controls. |
| `cuda_resident_toeplitz_results.md` | Julia-style batched-bank versus compact full-CUDA controls, including FP16 packed storage. |
| `segment_plots.py` | C-library-inferred maximum-energy segment views for the zoo, including multi-segment cases (MPRAGE, fat-sat EPI). |
| `waveform_plots.py` | Representative-TR and PNS (chronaxie + SAFE) plots for the safety pages. |
| `plot_benchmarks.py` | Reproducible download/load scaling plot from `results.json`. |
| `plot_safety_benchmarks.py` | Reproducible mechanical-resonance-vs-PNS cost plot from `safety.json`. |
| `seqdesc_diagram.py` | Schematic (non-data) diagram of the SEQDESC event-stream state machine. |

## Running

```bash
python docs/_bench/gate_bytes.py check             # 35 .seq payloads, byte-identical
bash docs/_bench/build_bench.sh                    # builds bench_pipeline
docs/_bench/bench_pipeline_timing scan.seq 5 .pge  # runtime, no heap hooks
python docs/_bench/bench_zoo.py --repeats=3        # -> results.json
python docs/_bench/bench_scale.py --scale=0.25     # edge protocols, sanity size
python docs/_bench/bench_scale.py                  # edge protocols, full size (~16 GB)
python docs/_bench/bench_scale.py --family=module  # only the past-the-cap cases
python docs/_bench/plot_benchmarks.py              # -> explanations/assets/benchmarks/
python docs/_bench/bench_safety.py --esp <table>   # -> safety.json
python docs/_bench/bench_recon_physics.py          # -> recon_physics_results.{json,md}
python docs/_bench/bench_recon_physics.py --profile subspace \
  --out docs/_bench/recon_subspace_scaling_results.json
python docs/_bench/bench_cuda_streaming.py          # -> cuda_streaming_256_results.{json,md}
python docs/_bench/bench_cuda_streaming_sense.py
python docs/_bench/bench_cuda_streaming_sense.py --mode full --size 192
python docs/_bench/bench_cuda_streaming_sense.py --mode full --size 160 \
  --cuda-mode resident
python docs/_bench/bench_cuda_streaming_sense.py --mode full --size 96 \
  --rank 15 --cuda-mode resident --cuda-transfer-precision float16
python docs/_bench/profile_cuda_toeplitz.py --size 256 --rank 5
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
binary) are gitignored. `gate_bytes_baseline.json` deliberately is **not** —
the gate is worthless without a version-controlled baseline to diff against.
Re-baselining (`gate_bytes.py save`) is a decision, not a fix: take it only
when a change to the emitted file is intended, and say so in the commit,
because every `.pge` truth fixture is downstream of those bytes.

## Reading the numbers

Timings are the **minimum** of the repeat count, not the mean: scheduler noise
only ever adds, so the minimum is the better estimate of the work actually
required.

Use `bench_pipeline_timing` for optimization decisions: it does not interpose
the allocator and reports `"heap_accounting": false` (its heap fields are
zero). Use `bench_pipeline` for the memory columns; its exact allocation
accounting can perturb runtime when a stage makes many small allocations. The
optional third argument selects the cache extension, for example `.pge` in the
GE integration.

Heap figures from `bench_pipeline` are live bytes attributable to the library,
with allocator rounding included — what the process holds, not what it asked
for. The figure reported after a cache load is measured with only that
collection live, so the three loads are directly comparable.

Python memory is `tracemalloc`'s peak, which counts Python objects only.
It is measured in a pass of its own, because `tracemalloc` hooks every
allocation and inflates the timings it would otherwise be gathered alongside.

`bench_scale.py`'s `events` column is the number of rows the block-referenced
libraries still hold when the design pass finishes, before `remove_duplicates`
runs. Nothing is collapsed on the way in — a TR template claims one row per TR
per slot and assumes any of them may move — so this is roughly one row per
event in the scan, and it is what the write-time pass is asked to collapse
rather than what survives it. It is a memory figure, not a vocabulary size.

`ext rows` counts the libraries an extension chain points at, plus the chain
nodes. Those can never be shared between passes — a block orders its chain by
reference ID, so sharing IDs would reorder chains and change the emitted file —
which makes them the floor on what a labelled scan costs to build.
