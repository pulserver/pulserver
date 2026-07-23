# Getting started

Install the sequence-authoring extra:

```bash
python -m pip install "pulserver[sequence]"
```

Authoring uses two namespaces. `pulserver.pypulseq` is a drop-in replacement
for PyPulseq — the whole upstream namespace, plus Pulserver's replacements for
a few of its objects:

```python
import pulserver.pypulseq as pp

seq = pp.Sequence()
seq.add_block(pp.make_delay(1e-3))
```

`pulserver.design` is the toolbox on top of it: one factory per RF pulse,
readout, sampling scheme and phase schedule Pulserver knows how to build. Each
returns a `SequenceModule` (a reusable multi-block fragment) or a `ScanLoop`
(a table of encoding positions):

```python
import pulserver.design as design
import pulserver.pypulseq as pp

system = pp.Opts()
excitation = design.make_slice_selective_pulse(0.35, 5e-3, system=system)
readout = design.make_line_readout(system, (0.22, 0.22), (128, 128))
loop = design.make_cartesian_sampling((128, 128), acceleration=2)

seq = pp.Sequence(system)
for shot in loop:
    for block in excitation:
        seq.add_block(*block)
    readout.set_state(lin_idx=int(shot[0, 0]))
    for block in readout:
        seq.add_block(*block)
```

The loop nesting stays yours: nothing in `pulserver.design` iterates the
sequence for you, which is what keeps a preparation, a trigger or a dummy TR
insertable at any level.

Next, {doc}`build a complete plugin <tutorials/build_a_sequence_plugin>` — one
sitting, ending with a file you could hand to a scanner. After that,
{doc}`how_to` has the task-shaped recipes, {doc}`api/index` the grouped public
interface, and {doc}`explanation` the reasoning underneath both.
