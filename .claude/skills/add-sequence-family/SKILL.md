---
name: add-sequence-family
description: >
  Add or modify a sequence family in the zoo (examples/sequence/*), including
  its protocol contract, its reconstruction plugin, its tests and its
  fixtures. Use when asked to add a sequence, change an existing family's
  timing or encoding, or wire a new readout into a plugin.
---

# Adding a sequence family

A family in `examples/sequence/` is shipped code: scikit-build installs it as
`pulserver.app.<name>`, one flat namespace for the whole zoo. It is
not a sample, and the zoo tests hold it.

## What a plugin has to provide

Read a neighbouring family first — `gre2D_sequence.py` is the smallest
complete one, `fse3D_sequence.py` the smallest interesting multishot one.
Every plugin has the same three parts:

1. **`main(...) -> pp.Sequence`** with keyword-only design parameters after
   `plot`, `test_report`, `write_seq`, `seq_filename`. It returns the
   sequence; running the module as a script writes a `.seq`.
2. **`PLUGIN`**, a `SequencePlugin` carrying `get_default_protocol`,
   `validate_protocol` and `make_sequence`. The protocol declares the
   parameters a console renders — kind, bounds, defaults, and a `UIParam`
   slot so a console puts TE in its TE field.
3. **A docstring** stating the physics: what each repetition plays, how TE and
   TR are budgeted, what the readout traverses, and which recon plugin reads
   it back.

`validate_protocol` must answer in interactive time. Build the modules and do
arithmetic on their `duration`, `t_first_echo_s` and `esp` — never append
blocks to measure the result, or the console stalls on every keystroke.

## Build from the design toolbox

Compose `pulserver.design` modules (readout, excitation, preparation) rather
than emitting events by hand. They already solve the timing, place the
labels, and carry the durations `validate_protocol` needs. If the readout you
need does not exist, add it to `pulserver/design/_readout/` with its own tests
before writing the plugin.

Encoding counters are written as the loop goes — the loop knows which line,
partition, slice and echo it is on, so it plays `LABELSET` events rather than
having them recovered later.

## Checks that must pass

```bash
bash scripts/run_tests.sh -k <family>
```

The zoo asserts, per family: the file round-trips; the derived segmentation
satisfies the PulSeg rules (`tests/python/test_pulseg_oracle.py`); the
authored counters match those recovered from the gradients by `auto_label`;
the TR is detected and bounds every instance; and the paired recon plugin
turns the stream back into images.

If the family changes any checked-in fixture:

```bash
bash scripts/regenerate_fixtures.sh
git diff --stat        # review what moved, and why
```

## Common traps

- **The TR must be detectable.** A zero-duration counter block at the top of a
  repetition, or a first repetition that differs from the rest, reads as one
  long TR. Materialise a prologue with `expand_repeats()` rather than leaving
  detection to guess.
- **Rotated readouts defer their FOV shift.** A rotating trajectory applies
  the offset at the consumer, not in the ADC — follow what the neighbouring
  non-Cartesian family does.
- **Gradients must be continuous across block boundaries.** Blocks run back to
  back; a waveform that ends at a non-zero amplitude next to one that starts
  at zero is an infinite slew, and `check_hardware_limits()` will say so.
- **Do not disable write deduplication.** It triples the file and the
  scanner's parse.

## Finish by

1. `bash scripts/format_and_lint.sh`
2. `bash scripts/run_tests.sh`
3. Giving the family a size sweep in `docs/_bench/bench_full.py` if it is
   new, and regenerating
   `docs/explanations/performance/full_benchmark` with
   `python docs/_bench/bench_full.py --only=<family>`.
