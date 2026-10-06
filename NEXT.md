# What is open, and what can be done here

This repository is the whole of what a reconstruction computer and a design
service run. The scanner-side interpreter is a separate, private repository,
and the work listed under *Needs a scanner* below cannot be checked without it
and without the vendor's own tooling, which this repository neither contains
nor depends on.

Everything under *Can be done here* is Python and C in this repository, checked
by `pytest -q`, `bash scripts/format_and_lint.sh --check` and
`bash scripts/build_docs.sh`. Nothing in it needs a scanner, a vendor SDK or a
licensed image.

**A recurring failure to look for.** Almost every defect found in this area so
far has had one shape: a value crosses a boundary and loses its reader. The
conversion resolved the ADC's phase modulation and nobody played it; the
caches of one exam were written where no other process could find them; a
reconstruction's pose is intercepted and dropped. None failed a test, because
each side passed its own. When picking up an item below, establish what
already crosses before designing what to add.

## Can be done here

### The sequence description reaches no simulator

`describe()` in `src/pulserver/proxy/_seqdesc.py` reads a sequence into the
event stream a simulator needs — the pulse a block turns the magnetisation
through, the readout it samples with, each timed as a simulation needs it —
and `as_rows()` lays it out. Nothing sends it.

It belongs after the MRD XML header and before the first acquisition of a
series, so a receiver has the description before any data. The receiver is
`blochsim`, which today has no MRD surface at all: no reader, no header
handling. The interpretation of the stream is hand-written on the
reconstruction side, by picking the signal model — that is the intended use,
not something to generate.

What to do: agree how the rows ride the stream, emit them in the proxy, and
give blochsim the reader. Check by reading a written stream back into the rows
it was built from.

### Throughput at the sizes that hurt

`scripts/check_throughput.py` measures designing and checking a sequence on the
fly. At ordinary sizes the safety checks dominate: peripheral nerve stimulation
is most of the cost and grows linearly in the block count, and checking costs
far more than designing. What is not known is where that stops being linear.

What to do: measure at sequence sizes well beyond the ordinary — the block
counts a long three-dimensional acquisition reaches, not a single slice — and
report where each term's cost turns over. The answer decides whether the
checks need to move off the interactive path.

### Naming what breaks the cache location

The IR cache is written beside its sequence under the process id, recorded in
the header so other parts of a playout can find it. That is believed robust and
has never been argued either way.

What to do: name what breaks it — a process that outlives another's id, a
cleaner, a shared filesystem — and either show the scheme survives each, or
replace it. This is reasoning and a test, not a measurement.

## Needs a scanner

These are listed so nobody starts them here. Each needs the private interpreter
repository and vendor tooling that is not available in this environment.

- **Radiofrequency costing while the operator prescribes.** The design service
  replies the RF definitions of a plugin's evaluation with the listing and the
  RF layout of a protocol with its validation, when asked, and a plugin states a
  layout by returning it from `evaluate`; every shipped scanner sequence states
  one TR of its scan. Reading these blocks and costing the
  RF of a prescription from them are work in the interpreter repository, and
  neither has run on a scanner.
- **Gradient heating on a dense repetition.** The model is evaluated over a
  whole repetition. Whether that is right for a dense, short-repetition
  sequence is untested: in every fixture the squeezed core costs less than the
  whole repetition, so it never fails in isolation and the asymmetry the model
  exists for is never reached. Adding fixtures did not help — the condition is
  a property of the sequence. Designing a candidate sequence can be done here;
  confirming it needs the scanner's own model.
- **Integer sample formats in the cache.** The profile that would carry them is
  in place and carries format and scale per quantity. The conversion waits on a
  measurement of what a sequencer spends converting while it plays, which can
  only be made on one. Note that only some quantities have a fixed scale: two
  of them depend on the prescription and on the scan, so they cannot be
  pre-scaled without tying a cache to one prescription.

## Conventions worth knowing before starting

- `src/c/` is C89 and is compiled as a scanner builds it — 32-bit, warnings as
  errors — by `tests/test_ir.py`. A C99 construct fails here before it reaches
  any scanner.
- A test module is **not** importable as a package where the suite runs. Share
  a helper through `tests/conftest.py`; `from tests.<module> import ...` takes
  the whole suite down on every platform.
- The compiled extension does not change with a branch. After switching to one
  that touches `src/cpp/` or `src/c/`, reinstall with `pip install -e .` or the
  tests run against the other branch's build.
- Three `tests/test_virtual_device.py` CUDA cases fail with a Triton
  compilation error on some machines, independently of any change. Confirm a
  failure against a clean tree before chasing it.
