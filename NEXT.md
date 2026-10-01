# What is open, across the repositories

The framework is five repositories: sequence design in `pypulseqpp`, the
intermediate representation and the reconstruction computer here, the
reconstruction engine in `bartorch`, simulation in `torchsim`, and the
scanner-side interpreter in `pulserver-interpreter`, which is private. This
page is the index of what is open in each and of which machine can finish it.

Every item is an issue in the repository that owns it. This page says what the
work is for and what it depends on; the issue says how to do it.

## Which machine

Most of the work needs nothing but a checkout. Three things cannot be
*finished* anywhere else, and each is marked **(laptop)** below:

| needs | why |
|---|---|
| the vendor environment | building the scanner binary and running its simulator; no other machine has the SDK or the licensed image |
| a GPU | a reconstruction measured, rather than merely run, and the device lane of the numerical tests |
| a scanner | nothing here. Every hardware claim is checked against the simulator, not against a magnet |

An item not marked **(laptop)** is finished where it is written: `pytest -q`,
`bash scripts/format_and_lint.sh --check` and `bash scripts/build_docs.sh` are
the whole of its verification.

An item marked **(laptop)** can still be *written* anywhere. The split is
between writing it and confirming it, and the issue says where the line falls,
so the remote half is done first and the local pass is short.

## pulserver

| what | issue | machine |
|---|---|---|
| A pose a reconstruction states reaches nothing | [#160](https://github.com/pulserver/pulserver/issues/160) | anywhere |
| The sequence description reaches no simulator | [#161](https://github.com/pulserver/pulserver/issues/161) | anywhere |
| Throughput at the sizes that hurt | [#162](https://github.com/pulserver/pulserver/issues/162) | anywhere |
| Name what breaks the cache location | [#163](https://github.com/pulserver/pulserver/issues/163) | anywhere |

#160 is the one to start with: prospective motion correction does nothing
outside the tests today, because the relay drops every pose a reconstruction
states and nothing constructs the writer the sequencer reads.

## torchsim

| what | issue | machine |
|---|---|---|
| Read a sequence description off an MRD stream | [#22](https://github.com/pulserver/torchsim/issues/22) | anywhere |

The other half of pulserver#161. The description is inferred from the sequence
in the proxy and sent before any data; what the rows *mean* is hand-written
here, by picking the signal model. That is the intended use, not a gap.

## pulserver-interpreter (private)

| what | issue | machine |
|---|---|---|
| Radiofrequency costing while the operator prescribes | [#1](https://github.com/pulserver/pulserver-interpreter/issues/1) | write anywhere, **(laptop)** to confirm |
| Gradient heating on a dense repetition | [#2](https://github.com/pulserver/pulserver-interpreter/issues/2) | design anywhere, **(laptop)** to confirm |
| Integer sample formats in the cache | [#3](https://github.com/pulserver/pulserver-interpreter/issues/3) | **(laptop)**: the measurement needs a sequencer |

The costing's two halves are already checked end to end — the design service
states which control drives each pulse, and the scanner-side parser reads that
statement. What has never run is the costing built on top, because it lives in
a mode the simulator suite does not drive. Writing that lane is remote work;
running it is not.

Gradient heating is evaluated over a whole repetition, and whether that is
right for a dense, short-repetition sequence is untested: in every fixture the
squeezed core costs less than the whole repetition, so it never fails in
isolation. Designing a candidate sequence is remote work -- it is a
`pypulseqpp` sequence and nothing more; confirming it needs the vendor's own
model.

Integer formats wait on a measurement of what a sequencer spends converting
while it plays, which can only be made on one. The profile that carries format
and scale per quantity is already in place. Note that only some quantities have
a fixed scale: two depend on the prescription and on the scan, so they cannot
be pre-scaled without tying a cache to one prescription.

## pypulseqpp and bartorch

Nothing is open that this framework is waiting on. Both are engines this
repository builds on rather than extends: work that belongs to one goes
upstream into it instead of being copied here.

One known defect to be aware of rather than to fix here: a reconstruction that
batches coil maps is refused on the device lane, which is why some numerical
comparisons run on the host alone. It predates this work and blocks automatic
device selection. **(laptop)** to reproduce.

## A recurring failure, worth looking for in any of them

Almost every defect found in this area so far has had one shape: a value
crosses a boundary and loses its reader. The conversion resolved the ADC's
phase modulation and nobody played it. The caches of one exam were written
where no other process could find them. A reconstruction's pose is intercepted
and dropped. None failed a test, because each side passed its own.

When picking up an item, establish what already crosses before designing what
to add. More than once an item written as "design this" turned out to be a
defect to fix.

## Conventions worth knowing before starting

- `src/c/` is C89 and is compiled as a scanner builds it -- 32-bit, warnings as
  errors -- by `tests/test_ir.py`. A C99 construct fails here before it reaches
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
