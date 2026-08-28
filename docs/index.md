# Pulserver

Pulserver takes a vendor-neutral [Pulseq](https://pulseq.github.io) sequence
from design to acquisition: it builds the sequence in Python, checks it
against the hardware limits a scanner enforces before it will play anything,
hands an interpreter the structure it needs to actually play it, and gives
the reconstruction the description it needs to interpret what came back.

Three interfaces, one representation:

| | |
|---|---|
| **Python** — design sequences from readout, excitation and preparation modules; write `.seq` files; run the safety checks; reconstruct. | {doc}`api/python/index` · {doc}`examples/index` |
| **C** — the scanner-side library: parse a `.seq`, derive its structure, gate it against the hardware, replay it block by block. C89, no dependencies. | {doc}`api/c/index` · {doc}`examples/c/index` |
| **C++** — the same library with RAII types, plus the reconstruction-side reader that turns a `.seq` chain into trajectories, labels and a sequence description. | {doc}`api/cpp/index` · {doc}`examples/cpp/index` |

## Where to start

- **New here?** {doc}`getting_started/index` builds a gradient echo, checks
  it, and looks at what the interpreter makes of it — about ten minutes.
- **Want to know why it is built this way?** {doc}`explanations/index` covers
  the structural TR, segmentation, the safety model, and the performance
  budget, with a short review of Pulseq and ISMRMRD for context.
- **Integrating an existing interpreter?** The easiest thing to adopt on its
  own is {doc}`the safety checks <examples/c/safety_gate>` — in particular the
  acoustic and nerve-stimulation ones, judged over the canonical TR rather
  than the whole scan.

## What is here that is not elsewhere

Pulseq describes *what to play*. A scanner needs rather more before it will
play it, and a reconstruction needs rather more before it can use the result.
Pulserver supplies the missing pieces, and each is documented as a concept
rather than as a function list:

| | |
|---|---|
| {doc}`explanations/sequence_model/design_service` | The sequence is a service the console asks, not a file someone copies — the motivation everything else follows from. |
| {doc}`explanations/sequence_model/tr_and_segmentation` | The repeating unit is *detected from content*, not annotated — which is what makes safety checks and hardware playout tractable. |
| {doc}`explanations/sequence_model/mrd_architecture` | The acquired data reach the reconstruction as an MRD stream that already knows its trajectory, its counters and its FOV shift — and a server that reconstructs it, caching per exam and never dropping a scan. |
| {doc}`explanations/safety/index` | Gradient, PNS and mechanical-resonance gating that runs before download, against the same engine the scanner runs. |
| {doc}`explanations/performance/index` | Designing, converting and checking a million-block scan, and what it costs in time and memory. |

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
explanations/index
examples/index
api/index
```
