# Amplitude, slew and continuity

The two cheap checks — {doc}`amplitude and slew <../safety/gradient_slew>`,
and gradient continuity across block boundaries — are the ones a scanner
always runs, because they need nothing a scanner does not already know. They
are also the ones a reference toolbox pays for on every `add_block`, which is
what makes a million-block sequence expensive to build.

Pulserver moves both off the building path and onto `write()`, under
`check_gradients=True`. What that costs, and why the two have different cost
models, is the whole page.

## Amplitude and slew walk the library, not the scan

`check_hardware_limits` iterates the **gradient library** — the distinct event
rows the file will store — and asks each one for its peak amplitude and its
steepest ramp. A trapezoid answers from three numbers; an arbitrary waveform
answers from one pass over its samples. However many times the scan plays a
waveform, it is examined once.

That only pays off once the library *is* small, and the library collapses to
its distinct rows when the sequence is deduplicated. Before that, every shot's
phase-encode can have its own row: same shape, different amplitude, separate
entry.

Which of those the check sees depends on the design layer. Where a plugin
publishes its events by identity, the library arrives at `write()` already
near its distinct size and the check is milliseconds on a protocol. Where it
does not, the check walks instances — and `write()` runs it *before*
deduplicating, so that is the cost it pays.

An EPI protocol at four scan lengths, with the repetition itself held fixed —
96 × 96, one blipped shot per slice, 99 blocks per TR:

| Blocks | Gradient library | After dedup | `check_hardware_limits` | …deduplicated |
|---:|---:|---:|---:|---:|
| 297 | 594 | 11 | 8.5 ms | 0.3 ms |
| 1 188 | 2 376 | 11 | 32.5 ms | 0.3 ms |
| 4 752 | 9 504 | 11 | 131.7 ms | 0.3 ms |
| 14 256 | 28 512 | 11 | 318.2 ms | 0.3 ms |

The right-hand column is the point: eleven distinct gradients is what this
sequence *is*, whatever its scan length, so the deduplicated check does not
move. The left-hand column is what the same check costs when it is asked to
look at instances instead — the cost model a per-`add_block` check is stuck
with.

## Continuity walks the block table, in C

Continuity cannot be a library question. Whether a waveform ends where its
neighbour begins is a property of the *pair*, so the pass is over block
instances, in scan order, and it is linear in the scan by construction.

It is also rotation-aware: both endpoints are rotated before they are
compared, so a block carrying a `ROTATIONS` extension is judged in the
physical frame the amplifiers actually slew in rather than in its own logical
one.

Linear, but in the C safety core rather than in Python:

| Blocks | `check_gradient_continuity` | …deduplicated |
|---:|---:|---:|
| 297 | 3.7 ms | 2.9 ms |
| 1 188 | 4.8 ms | 3.1 ms |
| 4 752 | 11.4 ms | 4.3 ms |
| 14 256 | 21.4 ms | 4.8 ms |

Deduplication helps here too, for a different reason: the pass reads each
block's endpoints through its library rows, and a library that fits in cache
is a faster table to walk than one that does not.

```{note}
Timings above are the checks alone. The first call that needs the C
representation also builds it — a serialise and a parse of the whole scan —
and on the writing path that bill falls to continuity, because amplitude and
slew do not need the C representation and it does.
{doc}`sequence_creation` reports it that way, the way `write()` actually pays
it.
```

## Which one runs where

Both are the same C arithmetic wherever they are invoked, so design time and
predownload cannot disagree — they are one implementation called from two
places.

- `write()` runs both, because building the sequence ran neither.
- The interpreter runs amplitude, slew, continuity and timing at predownload
  against the scanner's *real* rasters and limits, not the ones the script
  declared. That is the authoritative pass, which is why the scanner path
  writes without checking: running it twice buys nothing.
- A file written for anywhere else — a bench, a foreign toolbox, a colleague
  — gets every check at design time instead, because nothing downstream will
  run them.

See {doc}`../safety/index` for what each check refuses and why.
