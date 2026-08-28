# Explanations

Why Pulserver is built the way it is, and what it does that a Pulseq writer
alone does not. Each page is a concept, with the shipped plugins as the worked
evidence.

Read the background first if Pulseq or ISMRMRD are new to you; the rest stands
on its own.

| Section | What it covers |
|---|---|
| {doc}`background/index` | The formats and tools Pulserver sits between: Pulseq and how a written sequence is moved to a prescription, PulSeg, Nimpulseq, and MRD. A recap of published work, with no Pulserver specifics in it. |
| {doc}`sequence_model/index` | What a sequence *is* on each side of the download — design, scanner, reconstruction — and how the three views stay one representation. |
| {doc}`safety/index` | The checks a scanner runs before it will play a sequence, the physics behind each, and which of them Pulserver owns. |
| {doc}`performance/index` | Designing, converting, repositioning and checking a million-block scan while an operator waits, and how the fast paths stay equal to the plain ones. |

```{toctree}
:hidden:
:maxdepth: 2

background/index
sequence_model/index
safety/index
performance/index
```
