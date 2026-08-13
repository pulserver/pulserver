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
    system, excitation.rf, excitation.gz, excitation.gz_reph,
    fov=0.22, matrix=128, te=4e-3, tr=10e-3,
)

for line in lines:
    readout.rf.phase_offset = phases[line]                   # a phase is a write
    seq.add_block(readout.rf, readout.gz)
    seq.add_block(readout.wait_te, readout.gz_reph)
    seq.add_block(readout.gx_pre, pp.scale_grad(readout.gy_pre, ky[line]))
    seq.add_block(readout.gx, readout.adc)
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

Every `init_module` in the class chain is read, not only the innermost, so a
subclass that narrows a family — a spiral readout over the general
non-Cartesian one — publishes what it built *and* what its base built, with no
call of its own. None of the shipped modules asks for publication.

Two escape hatches remain for what the automatic path cannot see.
`self.publish()` reads the caller's frame, for events a helper function built
where no `init_module` saw them. `self.register(...)` publishes the structure
it is *given* rather than deducing one, so a one-entry list stays a list.

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

## The slice rephaser

A 2D excitation hands back its rephaser as a separate event, and a readout
takes it as its third positional argument. It is not simply appended: a
rephaser has to run straight off the selection lobe, so the readout puts it at
the **head of the first block after the pulse** — the TE wait when there is
one, the prewinder block otherwise. Where a prephaser is already running there,
the two overlap and the rephaser costs no echo time at all; a radial spoke,
whose prephaser is folded into the spoke itself, carries it inside the
acquisition block for the same reason.

Two placements are refused rather than fudged. A partition encode already
occupies the selection axis, so a 3D readout wants the merged form instead
(`is_slab=True`). And a non-Cartesian readout is oriented by rotating its
blocks: a rephaser on an axis the rotation mixes would be turned with the
interleave, silently dephasing the slice by an amount that changes shot to
shot. An in-plane acquisition therefore takes a rephaser on z and a projection
takes none.

## Echo trains

An FSE readout spans a CPMG train rather than a line, and everything about its
layout follows from one rule: every interval between an RF centre and the echo
it makes is half an echo spacing.

Nothing inside the train varies. One refocusing pulse at one amplitude, one
readout lobe, one encode per axis, replayed `etl` times — because what varies
between echoes is a *number*, and numbers belong to the scan loop. A variable
flip schedule is `rf_ref.amplitude` moved before each `add_block`; a line is
`pp.scale_grad(gy_pre, ky)`. Neither needs the module rebuilt, and both are the
same mechanism a single-line readout already uses for RF spoiling.

Phase encoding is a trapezoid of its own on every axis, including the partition
encode of a 3D train, which shares the slab axis with the refocusing pulse's
gradients but not a block with them. Scaling a trapezoid gives one registered
waveform played at many amplitudes; summing an encode onto a crusher gives an
arbitrary waveform, and then every line of the scan is a distinct shape — a
cost the interpreter pays per view and a trajectory it can no longer recognise
as one repeating unit.

The read axis carries the crushers instead. Between the refocusing pulse and
the acquisition it has to reach the readout plateau and come back, and the
refocusing pulse negates the accumulated moment in between, so an echo's own
dephasing prephases the next one and those two lobes need carry only the
plateau's ramps. That is the minimum, and it is what `spoiling_cycles = 0`
builds. Raising it lengthens both lobes symmetrically, which is what displaces
the FID that an imperfect refocusing pulse leaves behind out of the sampled
window — the phase-encode axes are rewound over the spacing and displace
nothing, so a train whose flip angles fall well below 180° wants some.

The first echo spacing is its own number. A slab-selective excitation is far
longer than a hard refocusing pulse, and the excitation-to-first-pulse interval
is by definition half a spacing, so a long pulse would otherwise stretch every
spacing in the train; `esp_first` covers the excitation and `esp` governs the
rest. Stimulated echoes from the longer first spacing do not rephase with the
rest of the train, which is why the first refocusing flip angle should stay
near 180° and keep its crushers.

## Balanced steady state

A bSSFP repetition is three blocks -- rewind, excite, read -- and two things
shape all of it.

Every axis returns to k = 0 between one pulse centre and the next, which is why
the module builds the slice rephasers itself instead of taking the excitation's:
their areas are fixed by the balance condition, a pair that cancels the
selection lobe outright, not by the pulse alone. Pass the excitation with
`rephase=False`.

The echo sits at TR/2, and that is one equation relating the two windows either
side of the excitation block. The module solves it by sizing those windows, so
a longer `tr` is padding split evenly and TE follows it. It also sets a floor:
half the acquisition block has to cover what follows the pulse centre in the
excitation block, or no half-flip pulse can sit half a repetition before the
first excitation. `half_flip_prep=False` lifts the floor for a loop that ramps
in with dummy repetitions instead.

The read axis never falls to zero between repetitions -- the closing lobe
leaves the plateau amplitude and the opening one arrives back at it -- so the
two share ramps a separate rewinder and prephaser would each have paid for. The
plateau runs to the end of its block, past the last sample by the receiver's
dead time, because otherwise there is nothing for the next lobe to leave from.

In 3D the partition encode shares the rephasers' window and is added onto them.
That stays a single trapezoid, and so a single registered shape at many
amplitudes, only because both are built on the same ramp time -- `make_trapezoid`
solves a shorter ramp for a smaller area otherwise, and the sum of two lobes
with different vertex times is an arbitrary waveform.

The half-flip pulse must be **opposite in phase** to the first excitation. It
leaves the magnetisation on one side of the steady-state cone and the first
full pulse has to carry it to the other side, which is a rotation the other
way; the same phase is worse than no preparation at all. `ONCE` marks the three
regimes for the interpreter: 1 on the entry transient, 0 on the steady state,
2 on the ramp-down.

## Zero echo time

A ZTE view inverts the usual order: the readout gradient is already at full
amplitude when the pulse fires, so encoding starts at the pulse and the spoke
runs from the centre of k-space outward. Between views the gradient does not
come down — it slews straight from one direction to the next — so a whole shell
costs one ramp up and one ramp down however many views it holds.

That is only affordable if the views share a waveform, and they can: view *k*
is the designed view under a rotation, provided consecutive views subtend the
**same angle**. This is much weaker than a circle — any constant-step walk on
the sphere qualifies, a pole-to-pole spiral included — but it does rule out
orderings whose step wanders, phyllotaxis among them.

**A ZTE view is two blocks** — pulse and hold, then acquire and turn. A block
carries at most one of an RF and an ADC, so the split is forced, and it costs
nothing: gradients are not obliged to reach zero at a block boundary, so the
plateau runs straight through and the two blocks play as one continuous
segment. The boundary is placed exactly where the acquisition starts, past
ringdown and receiver dead time, which leaves `adc.delay` at zero — an ADC
delayed by `rf.center + gap` would generally miss the RF raster and fail
`check_timing`. It also puts the excitation at a block boundary, where the
trajectory core takes its k-space reset; k then carries a fixed offset of half
the pulse duration, because the reset lands at the pulse's start rather than
its centre.

The centre of k-space is not acquired. Transmit ringdown and receiver dead time
run into the spoke and the samples inside them are dropped rather than
squeezed, so what survives stays on the radial grid — `n_missing` says how many
went, and filling the gap (PETRA, WASPI, algebraic) is a reconstruction's job.

### The shell and its shots

The ordering is the one piece of sampling a readout has to own, because the
transition is solved from its step. `ZteReadout` measures the step of a
supplied `directions=` and refuses one that wanders; left alone, it asks
`calc_projection_shell` for a shell that qualifies.

A generated shell runs **pole to pole**, `+z` to `−z`, visiting every polar
ring once, and a shot is that shell turned about `z` by `2π/n_shots`. Each ring
then carries `n_shots` spokes at even azimuth — full spherical coverage, and
every shot congruent with every other, so one set of waveforms and one set of
`view_rotations` serves them all. The two poles lie on the rotation axis and
are the one thing the shots share.

`n_views` defaults to a Nyquist-matched sphere, `ceil(π·matrix²)`, divided by
`n_shots`, so `n_shots` sets segment length without changing the total spoke
count. Playing a subset of the shots is angular undersampling; playing every
`n`-th keeps it uniform, playing a contiguous run leaves a wedge.

**`n_shots` and `n_views` are not independent, and they set the angular step
between them.** The shell holds the rings and the shots fill each one, so the
two spacings are reciprocal — ring to ring is `2/((n_views−1)·sin θ)`, within a
ring is `2π·sin θ/n_shots`. They agree at the equator only when

```
n_shots = π · n_views
```

which, against a Nyquist-matched total, puts `n_views` at the matrix size and
`n_shots` at `π·matrix`. That is the default. Move away from it and the
sampling goes anisotropic in one direction or the other: fewer shots
undersamples azimuth, more shots undersamples polar. Nothing forbids it —
`n_shots` is exactly the knob for trading segment length and angular
acceleration — but it is a trade, not a free choice.

The step follows, at `arccos(1 − 2/(n_views−1)) ≈ 2/√n_views`, so there is no
separate knob for it: it is the polar gap *at the pole*, where `sin θ = 0` and
no amount of azimuth buys any angle. Every other gap has slack, and that is
what the azimuth increments are solved to absorb — which is why one view to the
next strides across many rings' worth of azimuth while the shots, not the
shell, sample a ring.

`scheme='meridian'` is a half great circle at equal polar steps instead:
simpler, and it oversamples the poles.

## Labels and triggers

Counters — `LIN`, `PAR`, `SLC`, `ECO`, `PHS`, `REP`, `SET`, `AVG`, `SEG` — say
*where an acquisition belongs*, one ISMRMRD `EncodingCounters` field each.
A module builds a slot per name it is given and the loop writes the values.
One label publishes as the event itself and several as a list, following the
identity rule above:

```python
readout = design.LineReadout3D(..., labels=("LIN", "PAR"))
lin, par = readout.adc_labels
...
lin.value, par.value = line, partition
seq.add_block(readout.gx, readout.adc, *readout.adc_labels)
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
