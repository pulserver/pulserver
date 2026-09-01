# Conversion

```{admonition} TL;DR
:class: tip

- **The parse is the whole cost.** 1.2–1.5 s from text on a two-million-block
  scan, 0.6–0.8 s from binary. Detection by magic bytes, so either form reads
  through the same call.
- **Structure detection is tens of microseconds** — it compares normalised block
  identities the conversion already computed. Segmentation is a millisecond,
  consistency about 60 ms.
- **The cache pays for itself on the second read**: writing it costs about what
  the conversion did, reading it back costs 23–90 ms.
- The cache is sectioned, so a consumer loads only what it reads. Its version
  triple must match exactly — a stale cache is rejected, never repaired.
```

Between the file arriving and the first block playing, the scanner turns a text
or binary Pulseq file into the
{doc}`PulSeg representation <../sequence_model/pulseg_representation>` it
executes: parse it, resolve every event reference, find the repeating unit, cut
the segments, and check the result is consistent — in a few seconds, on a host
that is also running everything else on the console.

```{figure} ../assets/conversion/stages.png
What the scanner does between the file arriving and the first block playing,
and what sets the cost of each stage. The safety check sits in that interval
too, and on the timings below it is the largest thing in it — the stages this
page measures all happen in its shadow. See {doc}`full_benchmark`.
```

The three MPRAGE protocols of the {doc}`previous page <sequence_creation>`,
2 103 300 blocks each:

| Stage | Cartesian | Spirals, rotated | Spirals, written out |
|---|---:|---:|---:|
| parse + convert, from `.seq` text | 1.34 s | 1.20 s | 1.51 s |
| parse + convert, from binary | 0.78 s | 0.62 s | 0.67 s |
| structural TR detection | 0.01 ms | 0.02 ms | 0.02 ms |
| segmentation | 0.9 ms | 1.1 ms | 1.0 ms |
| consistency | 56 ms | 63 ms | 51 ms |
| cache write | 1.27 s | 1.13 s | 1.50 s |
| cache read | 79 ms | 23 ms | 90 ms |
| cache size | 137 MB | 133 MB | 137 MB |

## Parsing

Everything after parsing works on structures already in memory, so the parse
is the one stage whose cost is set by the file rather than by the sequence. In
text, most of it is `sscanf` over a hundred megabytes of formatted numbers.
The binary form of the same sequence skips both the formatting on the writing
side and the scanning on the reading side, and the whole read is about twice as
fast.

The format is upstream Pulseq's own, not invented here, and **detection is by
content**: an eight-byte magic decides how a file is read, so a `.seq` holding
binary still reads, and every existing caller works on either form with no new
argument. That is why the scanner path writes binary and the offline path
writes text — see `write_sequence` on the
{doc}`previous page <sequence_creation>`.

## Structure detection

Detecting the structural TR of a two-million-block scan takes **tens of
microseconds**, which is not a typo: the search runs over normalized block
identities the conversion has already computed, so asking "is block $n$ the
same structure as block $n+P$?" is an integer comparison, not a re-derivation.
Segmentation is a millisecond on top.

Speed is not why the TR is derived rather than annotated — an annotation is a
second source of truth that can disagree with the sequence, see
{doc}`../sequence_model/tr_and_segmentation`. That deriving it is also free is a
bonus, not the argument.

Consistency checking, at about 60 ms, is the largest of the three and still a
rounding error against the parse.

## The cache

Writing the converted representation beside the `.seq` costs about as much as
the conversion itself. Reading it back costs **23–90 ms** — one to two orders
of magnitude less than re-parsing and re-converting.

The file is split into independently loadable sections, so a consumer pays only
for what it reads: the pulse-generation pass takes the common framing and the
shapes; the scan loop additionally takes the per-instance tables, the rotations
and the scan loop itself. All fields are four bytes and the header records
endianness, because the target processor is big-endian and the host that wrote
the cache is not.

The version triple must match exactly on read. A cache at any other revision is
rejected outright and the `.seq` re-parsed — never partially read, never
heuristically repaired.

## Memory footprint

The cache is about 137 MB for these protocols, against roughly 100 MB of `.seq`.
It is larger than the file it derives from and that is the trade: resolved
tables cost bytes and save a parse. What keeps both numbers in the tens rather
than the thousands of megabytes is the same property everything else here rests
on — every waveform is stored once, and the scan is rows of indices into the
libraries that hold them.
