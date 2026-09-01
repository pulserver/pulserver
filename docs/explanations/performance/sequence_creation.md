# Creating the sequence

```{admonition} TL;DR
:class: tip

- A design loop runs Python once per repetition. What a protocol-scale sequence
  costs is decided by the **calls inside the loop**, and every one of them is
  compiled.
- **2 103 300-block MPRAGE protocols design in 3.8–7.9 s** — 1.8–3.7 µs per
  block, from an ordinary Python program.
- A two-million-block scan is **a dozen distinct waveforms** plus instance
  tables. Writing arms out instead of turning them with a `ROTATIONS` extension
  turns 15 shapes into 5 130, and that number sets the cost of everything after.
- The **binary write** skips number formatting on both ends; a reader tells the
  forms apart by content, not by file name.
```

`pulserver.pypulseq.Sequence` is a drop-in for the reference toolbox's, over a
compiled core. A design script that ran against PyPulseq runs here unchanged;
what differs is what each call it makes actually does.

## The compiled core

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

## Cost of a protocol-scale build

Three MPRAGE protocols at one encoding size — 512 partitions, 1024 views per
inversion train, **2 103 300 blocks** each — so the only thing that varies is
how the in-plane shot is held. Single core:

| Case | Design | Rate | Distinct shapes | Peak memory |
|---|---:|---:|---:|---:|
| Cartesian, 512² in-plane | 3.8 s | 1.8 µs/block | 12 | 1.4 GB |
| Stack of spirals, arms rotated | 5.3 s | 2.5 µs/block | 15 | 3.0 GB |
| Stack of spirals, arms written out | 7.9 s | 3.7 µs/block | 5 130 | 2.6 GB |

A few microseconds per block, holding across two orders of magnitude of
protocol size, is the design story.

The shapes column is the representation at work. The rotated-versus-written-out
pair shows why the `ROTATIONS` extension matters at scale: a rotated arm is one
row against a stored shape, where a written-out arm is a new waveform to
deduplicate, write and parse — 5 130 shapes instead of 15, and half again as
much design time. It is the same quantity that sets the cost of {doc}`pns` and
{doc}`mechanical_resonance` later.

```{note}
These are scale cases, not clinically prescribed protocols: they exist to
measure what the path costs per block at protocol size.
```

## Off-isocentre prescription

A prescription is applied once, by `TransformFOV`, to the finished sequence —
not inside the design loop. The reason, the cost and what a non-Cartesian
readout ends up storing are {doc}`transform_fov`.

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

Deduplication is folded in on purpose: it is part of what writing a file *is*,
not an optimisation a caller should have to remember. The sequence itself is
left alone — a deduplicated copy is written — and a sequence untouched since its
last deduplication skips the pass. `remove_duplicates(in_place=True)` is the
form for a sequence too large to hold twice.

**Declaring the TR reads no samples.** `declare_tr` — and `tr_size`,
`num_trs`, `num_segments` — derive from a structure-only conversion: the
writer emits every readout shape of an RF-free block as a six-value stub
(first sample, last sample, peak magnitude, behind a marker no decoder
accepts) and the scanner-side conversion detects the TR and the segments from
the block table and those edges. On 4 096 written-out arms of 4 096 points
the light write is 0.2 µs per arm and the light conversion 2.2 µs; the full
round trip it replaces is 60 µs. The structure a light conversion reports is
held equal to the full one's over eight shipped fixtures, and a light
collection refuses any waveform or safety request.

**Where the time goes at scale.** A written-out library is a few gigabytes
of samples, and every stage below is a pass over them; the discipline is that
each stage touches each sample about once, at memory speed:

- *Registering a waveform* copies it once into a **chunked** shape store — 32 MB
  chunks allocated once, a row never spanning two — so the library never
  re-copies itself as it grows.
- *Deduplicating* rounds every sample to the file's nine significant digits
  and hashes each row on every core, then walks the rows once in order to
  number first appearances; a duplicate drops out of the index and its bytes
  stay where they are, so nothing moves. 28 µs per arm of two 4 096-point
  shapes.
- *Writing* runs the writer twice over a *sink*: once counting, to learn the
  file's size, then filling a buffer of exactly that size — no zero-fill and
  no copy on the way to the caller's `bytes`. 19 µs per arm, 1.2 GB/s.

The {doc}`pipeline budget <pipeline_budget>` page puts these stages end to end
at 131 072 distinct arms.

Text against binary is not a size story at this scale; both files are dominated
by the block table, which is a row per block either way. It is a *parsing*
story, told on {doc}`conversion`: the scanner reads the binary form about twice
as fast and loses none of the precision the text form rounds away.

## Cost of the checks

`write()` runs three checks before it serialises — amplitude and slew,
continuity, and timing — and what each costs is set by what it is a property
of: a waveform, a pair of neighbours, or an event. On the same
2 103 300-block protocols:

| | Cartesian | Spirals, rotated | Spirals, written out |
|---|---:|---:|---:|
| `check_hardware_limits` — amplitude and slew | 9 ms | 2 ms | 105 ms |
| `check_gradient_continuity`, first call | 1.15 s | 1.65 s | 1.20 s |
| `check_gradient_continuity`, called again | 69 ms | 80 ms | 77 ms |
| `check_timing` | 49 ms | 38 ms | 53 ms |

**Amplitude and slew** are a pass over the gradient library: one evaluation per
distinct waveform, so milliseconds on a two-million-block scan. They run before
deduplication, so what they see is the library as the design layer left it —
{doc}`gradient_checks` measures both sides of that.

**Continuity** is a pass over block instances, because whether a waveform ends
where its neighbour begins is a property of the pair. Linear in the scan, but
in the C safety core: about 70–80 ms for two million blocks. That core works on
its own copy of the scan, built by serialising and parsing the whole thing —
the first caller to need it builds it, and on the writing path that caller is
this check, which is the difference between the two rows. Everything asked
afterwards, here and on the safety pages, finds the copy ready.

**Timing** asks of every block: does each delay, duration, dwell and ramp land on
its raster; does the block's stored duration agree with what its events take;
does the RF start after the coil's dead time and finish a ringdown before the
block ends; does the ADC leave its dead time at both ends; do soft delays
sharing an ID agree. The rasters and dead times it judges against are the
system's, not the ones the file records.

Most of that list is a property of the event, so it is decided once per
*distinct* event and attributed to each block that uses it — the same shape of
cost as the amplitude and slew pass. What is genuinely per block is arithmetic
over precomputed extents with no waveform decoded: does anything end after its
block does, and do the dead times fit. Two million blocks come to tens of
milliseconds.

The report is upstream PyPulseq's entry for entry, held there by a differential
test on deliberately broken files as well as clean ones — a check that disagrees
with the one it replaces is a different check, not a quicker one.

Rotation does not enter into it: resolving a `ROTATIONS` extension changes
gradient *values* and no time in the list above, which is why the rotated column
is the cheapest of the three.

All three are switchable on `write()`: `check_timing`, on by default exactly
as it is upstream, and `check_gradients`, covering both gradient passes. So
the two forms of writing are not arbitrary:

```python
write_sequence(seq, path, offline=False)  # to the scanner: binary, unchecked
write_sequence(seq, path, offline=True)   # anywhere else: .seq text, checked, signed
```

Which of the two a plugin wants follows from where the file is going, and
{doc}`../safety/index` sets out why: the scanner re-checks against its own
rasters and limits, and nothing else does.
