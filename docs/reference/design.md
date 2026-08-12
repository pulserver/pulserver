# `pulserver.design` — sequence-building blocks

`pulserver.design` holds reusable sequence modules: the handfuls of Pulseq
blocks that always travel together, designed once and named. Authoring is
split across two namespaces by role. `pulserver.design` holds every
`SequenceModule`; `pulserver.pypulseq` is the event layer beneath it —
upstream PyPulseq re-exported whole, plus Pulserver's replacements for a few
of its objects (`Sequence`, `Opts`, `make_label`, `make_rotation`,
`make_rf_shim`) and the plain-array helpers an encoding plan is built from.
The two share no names. The `pulserver` root namespace carries the plugin
contract (base classes, typed protocol parameters, protocol serialisation,
`run_cli`) and nothing else.

Requires the optional `pypulseq` dependency (same tier as
`pulserver.pypulseq` / `pulserver.io`).

## What a module is, and what it is not

A module owns a **design**: it solves its gradients, budgets its TE and TR,
lands its ADC on both time rasters, and publishes the resulting events under
the names its constructor gave them. It owns nothing else. It does not iterate,
does not hold state, and never sees a sampling pattern — which shots to play,
in what order, with which encodes, is the plugin's own `for` statement.

That division is why a module is a convenience rather than a requirement. A
plugin that builds its own events by hand loses the design help and keeps
everything else.

```python
readout = design.LineReadout2D(
    system, excitation.rf, excitation.gz_select,
    fov_m=0.22, matrix=128, te=4e-3, tr=10e-3,
)

for line in lines:
    readout.rf.phase_offset = phases[line]                   # a phase is a write
    seq.add_block(readout.rf, readout.gz_select)
    seq.add_block(readout.gx_pre, pp.scale_grad(readout.gy_phase, ky[line]))
    seq.add_block(readout.gx_read, readout.adc)
```

Per-shot variation is ordinary PyPulseq. The events a module publishes are the
very objects its blocks hold, so a write shows through immediately; an encode
is `scale_grad`; an orientation is a rotation event.

## How events get their names

Everything a module adds to a block is recorded, and at the end of
construction the recording is matched against the local variables of
`init_module`. A name is published as:

- **the object itself**, when only one distinct object ever wore it — a pulse
  replayed once per arm is still one pulse;
- **a list**, when several did — one gradient per interleave stays one per
  interleave.

The rule is identity, not count, which is what makes it right in the case that
matters: a per-arm list built by repeating a single waveform collapses back to
that waveform, because a trajectory whose arms are rotations of a base arm has
one waveform however many times it is played.

Two escape hatches. `self.publish()` reads the caller's frame, for events a
helper function built where `init_module` never saw them. `self.register(...)`
publishes the structure it is *given* rather than deducing one, so a one-entry
list stays a list — which is what label lists need, since a loop should index
them the same way whatever was asked for.

An event whose name collides with one the module itself answers to (`duration`,
`center`, `seq`) warns, and stays reachable as `module.events.<name>`.

## What a module answers

`blocks` is the structural view: one tuple of events per block, in the order
they were added.

`duration` and `center` are the module's timing — its length, and the point it
is timed against (an RF isodelay, a readout's echo). A readout also reports
`echo_time` and the `bandwidth_hz` it actually achieved, which is generally not
the one requested: `calc_adc_timing` solves the ADC and gradient rasters
together, and the dwell that satisfies both is the dwell you get.

A module also forwards the sequence-level analyses to the sequence it built
itself in — `plot`, `plot_kspace`, `calculate_kspace`, `calculate_pns`,
`calculate_gradient_spectrum`, `check_timing`, `test_report` — so a design can
be inspected without reaching for `.seq`. PNS and the gradient spectrum answer
for the module played **once** from rest; for a whole-TR readout that is the
meaningful single-shot answer. An RF module adds `sim_rf`, which is the one
view a sequence-level analysis cannot give.

## Labels and triggers

Counters — `LIN`, `PAR`, `SLC`, `ECO`, `PHS`, `REP`, `SET`, `AVG`, `SEG` — say
*where an acquisition belongs*, one ISMRMRD `EncodingCounters` field each.
A module builds a slot per name it is given and the loop writes the values:

```python
readout = design.LineReadout3D(..., labels=("LIN", "PAR"))
lin, par = readout.adc_labels
...
lin.value, par.value = line, partition
seq.add_block(readout.gx_read, readout.adc, *readout.adc_labels)
```

The module makes the slot; it cannot invent one the loop did not ask for, and
it does not decide what goes in it.

Flags — `NOROT`, `NOPOS`, `NOSCL`, `PMC`, `NAV`, `REV`, `SMS`, `REF`, `IMA`,
`NOISE`, `OFF`, `ONCE`, `TRID` — say *how a block is played or classified*.
Pulseq labels are sticky: a value set at one block persists until some later
block sets it again, so a flag that should not outlive its blocks has to be
cleared explicitly by the loop that set it. `pp.make_label` builds them and
`STICKY_FLAGS` lists the two that deliberately do outlive their module —
`ONCE`, which delimits a whole preparation or cooldown section, and `TRID`,
which names a repeating unit and is the group the safety model checks SAR
over.

Triggers and digital outputs are ordinary block events; a readout takes one
through `trigger=` because which block it belongs on is a property of the
design — a cardiac trigger gates the block that opens a shot.

### First/last-in-axis MRD flags

`FIRST_IN_ENCODE_STEP1`, `LAST_IN_SLICE`, `LAST_IN_REPETITION` and the rest are
not written by the sequence. The interpreter derives them by comparing each
acquisition's counter against the *observed* range of that counter over the
scan. Emitting the counters is therefore the whole mechanism: a dimension that
is looped but never labelled collapses to a single index, and both its flags
fire on every acquisition.

## Encoding plans

A plan is plain data, so it lives one layer down, in `pulserver.pypulseq`:
`make_uniform_mask`, `make_poisson_disc_mask`, `make_caipirinha_mask`,
`calc_sampled_lines` for what to sample; `make_linear_order`,
`make_centric_order`, `make_radial_order`, `make_radial_adaptive_order`,
`make_shuffling_order`, `calc_traversal_order` for the order to visit it in;
`calc_golden_angles`, `calc_tiny_golden_angles`, `calc_raga_angles`,
`calc_uniform_angles` for non-Cartesian orientations; and
`make_rf_spoiling_schedule`, `make_phase_cycling_schedule`,
`make_traps_schedule` for the per-repetition RF lists. All return arrays, and
plain NumPy does just as well.

References for the ordering schemes: Buonincontri et al., *Doubling the
repetition time without paying the price: 3D TSE with individually
parameterized echo trains*, ISMRM 566-05-007 for the linear / radial /
adaptive schemes; Tamir et al., *T2 Shuffling*, Magn Reson Med
2017;77:180–195 for random shuffling.

## API documentation rendering

The Sphinx API reference is configured in `docs/conf.py`; build it with
`sphinx-build -E -W -b html docs docs/_build/html`.
