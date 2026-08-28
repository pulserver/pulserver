# The sequence model

A sequence is looked at from three places — the console that designs it, the
scanner that plays it, and the reconstruction that interprets what comes back
— and all three are looking at one representation. These pages are that
representation from each side.

| Page | What it covers |
|---|---|
| {doc}`pulseg_representation` | The scanner's reading: a segmented block table of definitions and instance rows, and the one structure Pulserver adds above the segment. |
| {doc}`tr_and_segmentation` | The repeating unit and the segments inside it, both derived from the block content rather than annotated by the designer. |
| {doc}`design_service` | The design side: a Python program on a compiled sequence core, answering the console's protocol edits in interactive time. |
| {doc}`mrd_architecture` | The reconstruction side: an MRD stream that arrives already carrying its counters, trajectory and sequence description, and a server whose unit of context is the exam. |

```{toctree}
:hidden:
:maxdepth: 1

pulseg_representation
tr_and_segmentation
design_service
mrd_architecture
```
