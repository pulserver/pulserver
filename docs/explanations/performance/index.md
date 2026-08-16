# Throughput and footprint

A clinical protocol is not a demonstration. A 3D acquisition is hundreds of
thousands of blocks, and it has to be built while an operator waits, written
to a file, parsed by a scanner that has a few seconds and a few tens of
megabytes to spare, and gated against the hardware before it plays. This page
is what that costs, measured on the current tree.

## Measured

A 3D gradient echo, 256 × 256 × 192 with the zoo's default acceleration and
calibration region — **154 220 blocks**, a 5.2 MB `.seq`, a four-block TR
repeated 38 555 times. Single core of an i7-13700H, Python 3.11.

| Stage | Where | Cost |
|---|---|---|
| design the sequence | Python | 0.31 s |
| write the `.seq` | Python | 23 s |
| write the binary form | C++ | 0.32 s |
| parse + detect the structure | C | **0.10 s** |
| consistency check | C | 3 ms |
| gradient, slew, continuity | C | 7 ms |
| mechanical-resonance analysis | C | 16 ms |
| resident scan | C | **12.7 MB** |

The asymmetry is the point. Everything the *scanner* has to do — parse the
file, work out what repeats, check it against the hardware — is a tenth of a
second and thirteen megabytes for a protocol of this size. Everything the
*designer* does happens once per protocol edit.

## At protocol scale

The sizes a clinical 3D protocol actually reaches, design only:

| Protocol | Blocks | Design | Peak RSS |
|---|---|---|---|
| GRE, 512 × 512 × 512 | 823 436 | 1.3 s | 347 MB |
| MPRAGE, 512 × 1024 × 512 | 1 652 300 | 3.0 s | 563 MB |

Both are under **2 µs per block**, so design is linear in the block count
across two orders of magnitude of protocol size.

## The counters are written, not recovered

A design loop knows which line, partition, slice and echo every acquisition
is — they are what it iterates over — so it plays them as `LABELSET` events
while it builds. Nothing has to read them back.

Recovering them from the trajectory is still worth doing, and
{meth}`~pulserver.pypulseq.Sequence.auto_label` does exactly that: for a `.seq`
written elsewhere it is the only way, and for one built here it is an
independent check that the gradients encode what the loop believed. It costs a
full k-space evaluation of the scan, so it belongs in a test rather than in a
protocol build — and that is where the zoo runs it, asserting the authored
counters against the derived ones sequence by sequence.

The first/last boundary flags are not in the file at all. They follow from the
counters and the encoding limits, and the ISMRMRD client sets them from those
when it builds the acquisition — so writing them would put an extension row on
every acquisition to state something the reader already knows.

## Why the interpreter side is cheap

Three properties, none of them accidental:

**The scan is stored as ids, not as events.** A block is a row of indices
into definition libraries plus its per-instance parameters; 154 220 blocks
weigh 12.7 MB because the waveforms behind them are stored once. The
{doc}`../background/pulseg` static/dynamic split *is* this saving.

**The structure is detected, not searched for.** Periodicity is tested at
candidate periods derived from the content, and the answer is confirmed by
comparing normalized block identities that were already computed during
deduplication. Detecting a four-block TR in a 154 220-block table costs a
fraction of the parse.

**The checks walk the instance table.** Amplitude, slew and continuity are a
pass over rows with the waveform library resident — no re-expansion of the
timeline. The two expensive analyses are expensive per *shape*, not per
block, which is the next section.

## Where the analyses got their speed

Both opt-in checks were, naively implemented, too slow to run while an
operator edits a parameter. Both were made affordable by exploiting the same
structural fact rather than by approximating:

- **PNS** convolves $dG/dt$ with a nerve kernel. Evaluated over the timeline
  of a 30-minute protocol that is hundreds of millions of samples. Evaluated
  per *shape*, with templates reused across repetitions and the kernel
  truncated where it has decayed, it is tens of milliseconds — and the fast
  path is asserted equal to the exact one, per sequence family, in the test
  suite.
- **Mechanical resonance** needs the drive spectrum at the TR harmonics
  inside a band. Computing those chosen frequencies with a chirp-z transform,
  rather than an FFT of the whole scan, took a Cartesian protocol's analysis
  from about 5.5 s to a quarter of a second; at the scale measured above it
  is 16 ms.

## Where the design side spends its time

A design loop runs Python once per repetition, so what it costs is decided by
what each call it makes does — not by the loop. Everything on that path is
compiled:

- **Events are C++ objects, not namespaces.** `make_trapezoid` and friends
  hand back a slotted event whose fields are `PyMemberDef` offsets, so
  reading one is a load rather than a dictionary lookup. `scale_grad` — the
  call a phase-encode loop makes twice per shot — copies the struct and
  multiplies an amplitude in C++, which is why a trapezoid's `area` and an
  arbitrary gradient's `waveform` both follow without touching a sample.
- **A block is one call.** `add_block(*events)` unpacks and registers every
  event in one C function, reached through `METH_FASTCALL` so the arguments
  arrive as an array rather than a tuple.
- **Deduplication.** Finding the distinct events in a million rows is a
  rounding pass and a unique pass over a large float matrix — compiled
  kernels, with the NumPy implementations kept beside them and asserted
  equal.
- **Serialization.** Writing the text form is dominated by number formatting;
  it is the reason a 5 MB `.seq` takes tens of seconds while the binary form
  of the same scan takes a third of a second.

```{note}
Deduplication is not optional if the file is going to a scanner. Writing with
`remove_duplicates=False` produces a file roughly three times larger whose C
parse is roughly three times slower, because every repeated event is a
separate library entry.
```

## Reproducing it

```python
from pulserver.seqzoo import gre_3d

seq = gre_3d.main(n_x=256, n_y=256, n_z=192, slab_thickness=0.19, n_dummy=0)
seq.write("scan.seq")
```

then load it through the interpreter bindings and time the parse. The
numbers above come from that measurement, taken on the tree this
documentation was built from; re-measure rather than quote them when the
question is whether a change made something slower.
