# Conversion

Between the file arriving and the first block playing, the scanner has to turn
a text or binary Pulseq file into the
{doc}`PulSeg representation <../sequence_model/pulseg_representation>` it
executes: parse it, resolve every event reference, find the repeating unit,
cut the segments, and check the result is consistent. It has a few seconds to
do it in, on a host that is also running everything else on the console.

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

## Parsing is the floor, and binary halves it

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
writes text — see `write_sequence` on the {doc}`previous page <sequence_creation>`.

## Structure comes almost free

Detecting the structural TR of a two-million-block scan takes **tens of
microseconds**, which is not a typo: the search runs over normalized block
identities the conversion has already computed, so asking "is block $n$ the
same structure as block $n+P$?" is an integer comparison, not a re-derivation.
Segmentation is a millisecond on top.

That is worth stating plainly because the alternative was rejected on other
grounds: the TR is derived from the content rather than annotated in the file,
because an annotation is a second source of truth that can disagree with the
sequence — see {doc}`../sequence_model/tr_and_segmentation`. It happens to
also be free.

Consistency checking, at about 60 ms, is the largest of the three and still a
rounding error against the parse.

## The cache: pay once

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

## Where the footprint goes

The cache is about 137 MB for these protocols, against roughly 100 MB of `.seq`.
It is larger than the file it derives from and that is the trade: resolved
tables cost bytes and save a parse. What keeps both numbers in the tens rather
than the thousands of megabytes is the same property everything else here rests
on — every waveform is stored once, and the scan is rows of indices into the
libraries that hold them.
