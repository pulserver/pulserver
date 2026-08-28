# Throughput and footprint

A clinical protocol is not a demonstration. A 3D acquisition is hundreds of
thousands to millions of blocks, and it has to be built while an operator
waits, checked against the hardware, and parsed by a scanner with a few
seconds and a few tens of megabytes to spare. These pages are how Pulserver
makes that affordable, without any of the speed changing an answer: every
fast path is held equal to the plain calculation it replaces.

Two sequences carry the section, chosen because each is the informative
extreme of its question. **MPRAGE** carries the throughput story: at
512 × 1024 × 512 it is the largest of the shipped families, and its Cartesian and
stack-of-spirals variants between them exercise every case the design path
has — a line readout rescaled per shot, an arm rotated per shot, and arms
written out as distinct waveforms. **EPI** carries the safety story: a blipped
echo train is the case where events genuinely interact — through the nerve's
memory and through coherent spectral summation — so it is where a shortcut
would be easiest to get wrong, and where the equivalence tests bite hardest.

## The five stages

A sequence is built, written, parsed, and then judged three times. Each stage
has its own cost model, and each page below is one of them.

| Page | What it is | What sets its cost |
|---|---|---|
| {doc}`sequence_creation` | the Python design loop, and `write()` | one compiled call per block |
| {doc}`conversion` | the scanner reading the file into the PulSeg representation | the block table, once, plus one pass per distinct waveform |
| {doc}`gradient_checks` | amplitude, slew, gradient continuity | the number of *distinct waveforms*, not blocks |
| {doc}`pns` | the peripheral-nerve-stimulation estimate | one window, and the shapes inside it |
| {doc}`mechanical_resonance` | the acoustic drive spectrum | one window, and the frequencies actually asked about |

Two further pages sit beside them. One is the cost of a prescription change,
which is not part of building a sequence but is what an operator waits on
most often; the other puts the stages back together.

| Page | What it is |
|---|---|
| {doc}`transform_fov` | Moving, turning and resizing a written sequence: one pass instead of a block walk, and what a non-Cartesian readout stores. |
| {doc}`full_benchmark` | Every shipped plugin at four prescribable sizes, on the two clocks an operator feels: the parameter round trip, and one press of *Save Rx*. |

```{toctree}
:hidden:
:maxdepth: 1

sequence_creation
conversion
gradient_checks
pns
mechanical_resonance
transform_fov
full_benchmark
```

## Two properties of the representation

Everything on the five pages follows from two properties of the
{doc}`representation <../sequence_model/pulseg_representation>`, neither of
them accidental.

**The scan is stored as references into libraries, not as events.** A block is
a row of indices plus its per-instance parameters, so a million-block scan is
a small library of waveforms plus instance tables — tens of megabytes with
every waveform stored once. Any pass that can be phrased over the library
instead of over the blocks immediately stops scaling with the scan.

**Every expensive analysis runs over one window.** The
{doc}`structural TR <../sequence_model/tr_and_segmentation>` is the repeating
unit, detected from the block content; the scan is that window repeated, so
the window is where the work is. Detection itself is nearly free, because it
runs on normalized block identities the conversion computed anyway.

## The canonical TR

"The TR" is not one waveform: it is *built*, differently, for what each
consumer needs, and always so that it bounds the scan.

- **For the gradient-side checks** — amplitude, slew, PNS, mechanical
  resonance — the window is the **worst-case envelope**: at every block
  position, the largest amplitude that position takes across all instances of
  the TR, with its sign kept. No single repetition of the real scan looks
  like this window, deliberately: it is constructed so that a pass of the
  envelope is a pass of every instance.
- **For the RF checks** — SAR, coil heating — the worst instance is not the
  envelope but the worst *repetition*: a train whose flip angle ramps has its
  worst B1rms somewhere in the middle, so the check walks the TR instances
  and takes the worst one, not the first and not the mean.
- **For looking**, `seq.plot(tr=...)` draws the same windows the checks use —
  the envelope, or any actual instance by index.

The bound is checkable on any sequence, not only asserted here: an integer
`tr=` selects one real instance, so the same analysis can be run on the window
the gate uses and on any repetition it is supposed to cover.

```python
ok, norm, *_ = seq.calculate_pns(hardware, tr="worst_case")  # the gate's window
ok0, norm0, *_ = seq.calculate_pns(hardware, tr=0)           # one real instance
```

Because a TR plays back to back with copies of itself, both windows are
evaluated *periodically*: the history a nerve model or a spectrum needs at
the start of the window is wrapped around from its end, so the boundary
between repetitions is handled rather than ignored, and the peak found inside
one window is the peak of the steady-state scan.

