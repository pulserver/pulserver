# Getting started

Install the sequence-authoring extra:

```bash
python -m pip install "pulserver[sequence]"
```

Authoring uses two namespaces. `pulserver.pypulseq` is a drop-in replacement
for PyPulseq — the whole upstream namespace, plus Pulserver's replacements for
a few of its objects, including a `Sequence` whose libraries and file formats
live in C++:

```python
import pulserver.pypulseq as pp

seq = pp.Sequence(pp.Opts())
seq.add_block(pp.make_delay(1e-3))
```

`pulserver.design` is the toolbox on top of it: reusable modules — an
excitation, a preparation, one whole readout TR — that design their waveforms
once and hand them back under the names their constructor gave them.

```python
import pulserver.design as design
import pulserver.pypulseq as pp

system = pp.Opts()
excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
readout = design.LineReadout2D(
    system, excitation.rf, excitation.gz_select,
    fov_m=0.22, matrix=128, te=4e-3, tr=10e-3,
)

phases = pp.make_rf_spoiling_schedule(128)

seq = pp.Sequence(system)
for line in range(128):
    readout.rf.phase_offset = readout.adc.phase_offset = phases[line]
    seq.add_block(readout.rf, readout.gz_select)
    seq.add_block(readout.wait_te)
    seq.add_block(readout.gx_pre, pp.scale_grad(readout.gy_phase, (line - 64) / 64))
    seq.add_block(readout.gx_read, readout.adc)
    seq.add_block(readout.gx_rew, pp.scale_grad(readout.gy_rew, (line - 64) / 64))
    seq.add_block(readout.wait_tr)

seq.write("gre.seq")
```

Nothing in `pulserver.design` iterates the sequence for you. The loop nesting,
the encoding plan and the order of the shots stay yours, which is what keeps a
preparation, a trigger or a dummy TR insertable at any level — and a module is
a convenience, never a requirement. Per-shot variation is ordinary PyPulseq:
a phase is an attribute write, an encode is `scale_grad`, an orientation is a
rotation event.

The masks, orderings and angle generators an encoding plan is built from live
in `pulserver.pypulseq` rather than here, because they return plain arrays:
`make_uniform_mask`, `make_poisson_disc_mask`, `calc_traversal_order`,
`calc_golden_angles`. Plain NumPy does just as well.

Next, {doc}`how_to` has the task-shaped recipes, {doc}`api/index` the grouped
public interface, and {doc}`explanation` the reasoning underneath both.
