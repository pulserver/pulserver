# Amplitude, slew and continuity

```{admonition} TL;DR
:class: tip

- Both checks move off `add_block` and onto `write()`, under
  `check_gradients=True`.
- Slew and continuity are {doc}`one criterion <../safety/gradient_slew>` asked
  about two different sample pairs, and the pair is what sets the cost.
- **Interior slew is a shape question**: the normalised shape's slew times an
  amplitude, so an EPI protocol costs 0.3 ms at any scan length once
  deduplicated — against 8.5→318 ms when the same check walks instances.
- **The seam is a pair question** — two endpoints at their own amplitudes and
  rotations — so it walks the block table and is linear in the scan by
  construction: 3.7→21.4 ms over the same range, in the C safety core.
- Continuity is **rotation-aware**: both endpoints are rotated before they are
  compared.
- Same C arithmetic at design time and at predownload, so the two cannot
  disagree.
```

{doc}`Amplitude and slew <../safety/gradient_slew>` and gradient continuity are
the checks a scanner always runs, because they need nothing a scanner does not
already know. They are also the ones a reference toolbox pays for on every
`add_block`.

Slew and continuity are the same inequality — the seam is just the sample pair
that straddles a block boundary. They are two passes here because of what each
pair needs to be evaluated, and that is the whole of this page.

```{figure} ../assets/gradient_checks/library_vs_scan.png
The two checks read the same sequence through different structures. Amplitude
and slew are properties of a waveform, so they are answered once per library
entry; continuity is a property of a pair of neighbours, so it has no choice
but to walk the block table.
```

## Amplitude and slew

`check_hardware_limits` iterates the **gradient library** — the distinct event
rows the file will store — and asks each one for its peak amplitude and its
steepest ramp. A trapezoid answers from three numbers; an arbitrary waveform
answers from one pass over its samples, and what that pass yields is the
*normalised* slew of the shape, in 1/s: the per-block answer is then that number
times the amplitude the instance plays it at. However many times the scan plays
a waveform, its samples are examined once.

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

## Continuity

The seam cannot be a library question. Its two samples come from different
waveforms, and each has to be scaled by *its own* instance amplitude before they
can be subtracted — so the pass is over block instances, in scan order, and it
is linear in the scan by construction.

Rotation is the same story: both endpoints are turned by their own block's
rotation before they are compared, so a block carrying a `ROTATIONS` extension
is judged in the physical frame the amplifiers actually slew in. Neither the
amplitude nor the rotation is a property of the shape, which is exactly why this
half cannot collapse onto the library the way the interior half does.

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

## Where each check runs

Both are the same C arithmetic wherever they are invoked, so design time and
predownload cannot disagree — they are one implementation called from two
places.

- `write()` runs both, because building the sequence ran neither.
- The interpreter runs amplitude, slew, continuity and timing at predownload
  against the scanner's *real* rasters and limits, not the ones the script
  declared. That is the authoritative pass, which is why the scanner path
  writes without checking: running it twice buys nothing.
- A file written for anywhere else is checked at design time instead, since
  nothing downstream will check it — see {doc}`../safety/index`.

See {doc}`../safety/index` for what each check refuses and why.
