# Explanations

Why Pulserver is built the way it is, and what it does that a Pulseq writer
alone does not. Each page is a concept, with the sequence zoo as the worked
evidence.

Read the background first if Pulseq or ISMRMRD are new to you; the rest
stands on its own.

## Background

Enough of the formats and tools Pulserver sits between to follow everything
else: what a Pulseq file says, what the PulSeg representation adds to it,
what MRD carries back, and the GUI that stands in when a console cannot ask
for a sequence itself.

```{toctree}
:maxdepth: 1

background/pulseq
background/pulseg
background/ismrmrd
background/nimpulseqgui
```

## The sequence model

Why the sequence is a service rather than a file, and what it *is* on each
side of the download: a set of modules and a loop on the design side, a
segmented block table with a detected TR on the scanner side, and a
description on the reconstruction side.

```{toctree}
:maxdepth: 1

sequence_model/interactive_design
sequence_model/modules_and_loops
sequence_model/protocol_ui
sequence_model/tr_and_segmentation
sequence_model/pulseg_ir
```

## Safety

The checks a scanner runs before it will download a sequence, and what
Pulserver runs at design time so a rejection happens at the desk instead of
at the magnet.

```{toctree}
:maxdepth: 1

safety/index
safety/gradient_slew
safety/pns
safety/mechanical_resonance
```

## Performance

A clinical protocol is a million blocks. What it costs to design it, convert
it, check it and write it — and where the cost went.

```{toctree}
:maxdepth: 1

performance/index
```

## Validation

```{toctree}
:maxdepth: 1

validation/sequence_zoo
```
