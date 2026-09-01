# Throughput and footprint

```{admonition} TL;DR
:class: tip

- A clinical 3D acquisition is hundreds of thousands to millions of blocks,
  built while an operator waits and parsed by a scanner with seconds and tens of
  megabytes to spare.
- **No fast path changes an answer.** Each is held equal to the plain
  calculation it replaces, by a test that computes both.
- Everything here follows from two properties of the representation: the scan is
  **references into libraries**, not events; and every expensive analysis runs
  over **one window**, the structural TR.
- **MPRAGE** carries the throughput story, **EPI** the safety story.
```

A clinical protocol is not a demonstration. These pages are how Pulserver makes
it affordable, without any of the speed changing an answer.

Two sequences carry the section. **MPRAGE** carries the throughput story: at
512 × 1024 × 512 it is the largest of the shipped families, and its Cartesian and
stack-of-spirals variants between them exercise every case the design path
has — a line readout rescaled per shot, an arm rotated per shot, and arms
written out as distinct waveforms. **EPI** carries the safety story: a blipped
echo train is the case where events genuinely interact — through the nerve's
memory and through coherent spectral summation — so it is where a shortcut
would be easiest to get wrong, and where the equivalence tests bite hardest.

## Building, prescribing, checking

A sequence is built, moved to the prescription, written, parsed, and then
judged three times. Each stage has its own cost model, and each page below is
one of them.

| Page | What it is | What sets its cost |
|---|---|---|
| {doc}`sequence_creation` | the Python design loop, and `write()` | one compiled call per block |
| {doc}`transform_fov` | moving, turning and resizing the finished scan | the libraries, not the scan — and what a non-Cartesian readout stores |
| {doc}`conversion` | the scanner reading the file into the PulSeg representation | the block table, once, plus one pass per distinct waveform |
| {doc}`gradient_checks` | amplitude, slew, gradient continuity | the number of *distinct waveforms*, not blocks |
| {doc}`pns` | the peripheral-nerve-stimulation estimate | one window, and the shapes inside it |
| {doc}`mechanical_resonance` | the acoustic drive spectrum | one window, and the frequencies actually asked about |

The prescription sits second because that is where it happens: it is applied
to the finished scan, before the file is written, and it is the one of these
an operator triggers over and over on a sequence that is otherwise unchanged.
A last page puts the stages back together.

| Page | What it is |
|---|---|
| {doc}`full_benchmark` | Every shipped plugin at four prescribable sizes, on the two clocks an operator feels: the parameter round trip, and one press of *Save Rx*. |

```{toctree}
:hidden:
:maxdepth: 1

sequence_creation
transform_fov
conversion
gradient_checks
pns
mechanical_resonance
full_benchmark
```

## Two properties of the representation

Everything on the pages above follows from two properties of the
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

Every analysis below runs over one window and applies its verdict to the scan.
*What* goes in that window differs by check — a constructed envelope, a
per-position spectral bound, or a real repetition selected by a closed-form
score — and which one each check takes, why it bounds the scan, and how it is
found without evaluating every repetition is
{doc}`../safety/canonical_tr`.

The bound is checkable on any sequence, not only asserted: an integer `tr=`
selects one real repetition, so the same analysis runs on the window the gate
uses and on any repetition it is supposed to cover. `seq.plot(tr=...)` draws
either.

```python
ok, norm, *_ = seq.calculate_pns(hardware, tr="worst_case")  # the gate's window
ok0, norm0, *_ = seq.calculate_pns(hardware, tr=0)           # one real instance
```

Because a TR plays back to back with copies of itself, the window is evaluated
*periodically*: the history a nerve model or a spectrum needs at its start is
wrapped around from its end, so the peak found inside one window is the peak of
the steady-state scan.
