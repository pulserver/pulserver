# Explanations

Why Pulserver is built the way it is, and what it does that a Pulseq writer
alone does not. Each page is a concept, with the sequence zoo as the worked
evidence.

Read the background first if Pulseq or ISMRMRD are new to you; the rest
stands on its own.

```{toctree}
:maxdepth: 2

background/index
sequence_model/index
safety/index
performance/index
validation/sequence_zoo
```

| | |
|---|---|
| {doc}`background/index` | The formats and tools Pulserver sits between: Pulseq, PulSeg, Nimpulseq, and MRD. |
| {doc}`sequence_model/index` | What a sequence *is* on each side of the download — design, scanner, reconstruction — and how the three views stay one representation. |
| {doc}`safety/index` | The checks a scanner runs before it will play a sequence, and the physics behind each. |
| {doc}`performance/index` | Designing, converting and checking a million-block scan while an operator waits — and how the fast paths stay exactly equal to the plain ones. |
| {doc}`validation/sequence_zoo` | Twenty sequence families, built and checked end to end, as evidence that all of the above holds. |
