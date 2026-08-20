# Creating the sequence

`pulserver.pypulseq.Sequence` is a drop-in for the reference toolbox's, over a
compiled core. A design script that ran against PyPulseq runs here unchanged;
what differs is what each call it makes actually does.

That distinction is the whole cost model. A design loop runs Python once per
repetition — there is no way around that, and no reason to want one — so what
a protocol-scale sequence costs is decided by the calls inside the loop, not
by the loop.

## Everything on the loop's path is compiled

- **Events are compact compiled objects.** `make_trapezoid` and friends hand
  back an object whose fields read at native speed, and `scale_grad` — the
  call a phase-encode loop makes twice per shot — copies the event and
  rescales it in compiled code, without touching a waveform sample.
- **A block is one call.** `add_block(*events)` registers every event of a
  block in a single compiled call.
- **Finding what repeats is compiled.** Collapsing the events of a
  million-block scan to its distinct rows — the step that turns a scan into a
  small library of shapes plus instance tables — is a rounding pass and a
  uniqueness pass over large numeric arrays, in compiled kernels kept beside
  plain NumPy implementations the tests hold them equal to.
- **Writing is compiled, and has a binary form.** Serialising `.seq` text is
  dominated by number formatting; the binary form of the same sequence skips
  it, and a reader tells the two apart by content rather than by file name.

## What a protocol-scale build costs

Three MPRAGE protocols at one encoding size — 512 partitions, 1024 views per
inversion train, **2 103 300 blocks** each — so the only thing that varies is
how the in-plane shot is held. Single core:

| Case | Design | Rate | Distinct shapes | Peak memory |
|---|---:|---:|---:|---:|
| Cartesian, 512² in-plane | 3.8 s | 1.8 µs/block | 12 | 1.4 GB |
| Stack of spirals, arms rotated | 5.3 s | 2.5 µs/block | 15 | 3.0 GB |
| Stack of spirals, arms written out | 7.9 s | 3.7 µs/block | 5 130 | 2.6 GB |

A few microseconds per block, holding across two orders of magnitude of
protocol size, is the whole design story: the console asks, the answer comes
back on the download clock, and it does so from an ordinary Python program.

The shapes column is the representation at work — a two-million-block scan is
a dozen distinct waveforms plus instance tables. The rotated-versus-written-out
pair shows why the `ROTATIONS` extension matters at scale: a rotated arm is one
row against a stored shape, where a written-out arm is a new waveform to
deduplicate, write and parse — 5 130 shapes instead of 15, and half again as
much design time. It is the same quantity that sets the cost of {doc}`pns` and
{doc}`mechanical_resonance` later.

```{note}
These are scale cases, not clinically prescribed protocols: they exist to
measure what the path costs per block at protocol size.
```

## Moving the FOV

An off-isocentre prescription is applied by `TransformFOV` at the end of the
plugin, over a scan that is by then millions of blocks — so what it costs is a
property of the scan size rather than of the sequence family, and it is
invisible inside a single end-to-end number. Measured on its own, it is
**1.2–1.7 s** on all three cases above, or well under a microsecond per block.

It is applied once, to the finished sequence, precisely because it is cheaper
there than in the loop: a shift is a phase, and the phase of a whole scan is a
vector operation.

## Writing

`write()` is not only serialisation. In order, it checks, declares the
structural TR, deduplicates, and then serialises:

| | Cartesian | Spirals, rotated | Spirals, written out |
|---|---:|---:|---:|
| `remove_duplicates` | 0.57 s | 0.77 s | 0.35 s |
| `.seq` text | 0.84 s | 0.15 s | 0.59 s |
| binary | 0.22 s | 0.19 s | 0.40 s |
| text size | 101 MB | 106 MB | 98 MB |
| binary size | 98 MB | 93 MB | 80 MB |

Deduplication is folded in here on purpose: it is part of what writing a file
*is*, not an optimisation a caller should have to remember. The sequence
itself is left alone — a deduplicated copy is written — and a sequence that
has not been touched since its last deduplication skips the pass entirely.

Text against binary is not a size story at this scale; both files are
dominated by the block table, and the block table is a row per block either
way. It is a *parsing* story, and it is told on the {doc}`conversion` page:
the scanner reads the binary form about twice as fast, and loses none of the
precision the text form rounds away.

## What the checks cost

`write()` runs three checks before it serialises — amplitude and slew,
continuity, and timing — and they have three different cost models:

| | Cartesian | Spirals, rotated | Spirals, written out |
|---|---:|---:|---:|
| `check_hardware_limits` — amplitude and slew | 9 ms | 2 ms | 105 ms |
| `check_gradient_continuity` | 69 ms | 80 ms | 77 ms |
| …including the structure build it pays for | 1.15 s | 1.65 s | 1.20 s |
| `check_timing`, per block | 77 µs | 637 µs | 84 µs |

**Amplitude and slew** are a pass over the gradient library, so they cost one
evaluation per distinct waveform — milliseconds on a two-million-block scan,
and proportional to the library rather than to the block count. They run
before deduplication, so what they see is the library as the design layer left
it; {doc}`gradient_checks` measures both sides of that.

**Continuity** is a pass over block instances, because whether a waveform ends
where its neighbour begins is a property of the pair. Linear in the scan, but
in the C safety core: about 70–80 ms for two million blocks. The first caller to
need the C representation also builds it — a serialise and a parse of the whole
scan — and on the writing path that bill falls here, which is the second row.

**Timing** is the one check whose cost is genuinely per block in Python: it is
upstream's own checker, and running it means materialising an upstream
PyPulseq sequence over the window it is given. At tens to hundreds of
microseconds per block, that is minutes on a protocol — and gigabytes, which
is why the table reports a rate measured on a 20 000-block window rather than
a whole-scan time. The rotated case is the expensive one because upstream has
no vocabulary for a `ROTATIONS` extension, so every block must be replayed with
the rotation resolved into its events.

So the two forms of writing are not arbitrary:

```python
write_sequence(seq, path, offline=False)  # to the scanner: binary, unchecked
write_sequence(seq, path, offline=True)   # anywhere else: .seq text, checked, signed
```

Going to the scanner, the interpreter checks timing and gradients at
predownload against its *real* rasters and limits, so checking again here buys
nothing. Going anywhere else — a bench, a foreign toolbox, a colleague —
nothing downstream will check, so everything is checked here.

## Counters are written, not recovered

The design loop knows which line, partition, slice and echo every acquisition
is — they are what it iterates over — so it writes them into the file as labels
while it builds, for free.
{meth}`~pulserver.pypulseq.Sequence.auto_label` can recover them from the
gradients instead, at the cost of a full k-space evaluation of the scan. The
zoo runs it as an independent check that the gradients encode what the loop
believed, which is where a whole-scan cost belongs.
