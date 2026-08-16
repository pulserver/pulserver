# Pulseq: the portable description

[Pulseq](https://pulseq.github.io/) is the open, vendor-neutral file format
Pulserver accepts as input. A `.seq` file is a flat, time-ordered list of
**blocks**. Each block is a row of indices into separately-stored event
tables — an RF pulse, up to three gradients, an ADC window, an extension
(trigger, label, rotation) — plus a block duration. Repeated waveform shapes
may share a table entry, but every *use* of a shape is still its own block
in the file: the format does not compress by construction, it only avoids
storing a shape twice.

This is deliberately minimal. Pulseq has:

- **No mandatory TR.** A sequence is just the blocks in file order; nothing
  declares a repeating unit.
- **No segment or memory-region concept.** Nothing in the format says which
  blocks a scanner could load and replay as one contiguous unit.
- **No notion of "the same shape played again".** A ky step at a different
  amplitude and a completely unrelated gradient are both just a new gradient
  event unless the writer happened to reuse a table entry.

That is the right design for an *interchange* format — portability wants the
least possible structure — but it leaves exactly the questions a scanner
needs answered: what is reusable, what has to be resident in instruction
memory, and what varies from one repetition to the next. Pulseq intentionally
answers none of them; Pulserver has to.

## What Pulserver reads from it

Parsing walks the block list once and classifies every block by its event
table entries: RF-carrying (excitation), gradient-only (readout, spoiler,
crusher, rewinder), ADC-carrying (acquisition), or empty (a pure delay).
Rotation and label extensions attach per-block metadata without changing this
classification. Nothing about TR structure, segment boundaries or repetition
is decided during this pass — those are properties Pulserver *infers*
afterwards, in the [PulSeg intermediate representation](pulseg), not
properties Pulseq encodes.

For the conformance mapping between Pulseq's object model and what Pulserver
builds from it, see {doc}`../../pulseg_conformance`.
