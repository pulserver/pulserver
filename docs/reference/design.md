# `pulserver.design` — sequence-building blocks

`pulserver.design` collects the reusable building blocks shared by the
example sequence plugins under [`examples/sequences/`](../../examples/sequences/).
It replaces the old per-family `_*_common.py` helpers (loaded via a
`spec_from_file_location` hack) with an installed, importable package: a
plugin imports module factories and sampling helpers directly from
`pulserver.design` even when the plugin file itself is exec-loaded
standalone by the bridge or tests.

Authoring is split across two namespaces by role. `pulserver.design` holds
every factory that returns a `SequenceModule` or a `ScanLoop`;
`pulserver.pypulseq` is the event layer beneath it — upstream PyPulseq
re-exported whole, plus Pulserver's replacements for a few of its objects
(`Sequence`, `Opts`, `make_label`, `make_rotation`, `make_rf_shim`). The two
share no names. The `pulserver` root namespace carries the plugin contract
(base classes, typed protocol parameters, protocol serialisation, `run_cli`)
and nothing else.

Requires the optional `pypulseq` dependency (same tier as
`pulserver.pypulseq` / `pulserver.io`).

## Public surface

| API group | Contents |
| --- | --- |
| `pulserver.params` | Protocol-dict getters/setters, phase-FOV and ACS resolution, readout/phase axis resolution. |
| `pulserver.design.make_*_pulse` | RF excitation and magnetization-preparation module factories. |
| `pulserver.design.make_*_readout` | Cartesian, EPI, FSE/CPMG, non-Cartesian, and ZTE readout module factories. |
| `pulserver.design.make_crusher`, `make_phase_encoding`, `make_phase_blip` | Gradient factories. Call `make_crusher` independently on each required axis. |
| `pulserver.design.make_*_schedule` | RF phase, phase-cycling, and refocusing-flip schedules. |
| `pulserver.design.make_*_sampling`, `make_slice_loop`, `make_counter_loop` | [Scan-loop factories](sampling.md) for Cartesian, EPI, radial/spherical non-Cartesian, slice, SMS, and counter (frame/contrast/average) loops. |
| `pulserver.pypulseq.make_label`, `COUNTER_LABELS`, `FLAG_LABELS`, `STICKY_FLAGS` | The Pulseq label set Pulserver understands, split into counters and flags. |
| `pulserver.run_cli` | Declarative offline CLI shared by every plugin. |
| `pulserver.SequenceModule`, `pulserver.ScanLoop`, `pulserver.EncodingAxis` | The abstract types those factories return. |

The corresponding implementation modules under `pulserver.design` use
leading underscores and are not public import locations.

## Counters, flags and triggers

The Pulseq label set splits in two, and the split is the toolbox's division of
labour.

Both halves go through the module's one setter, `set_state`, which tells them
apart by name: **UPPERCASE** keywords are labels, lowercase ones are the numbers
the waveforms are rebuilt from.

**Counters** — `LIN`, `PAR`, `SLC`, `ECO`, `PHS`, `REP`, `SET`, `AVG`, `SEG`,
`ACQ` — say *where an acquisition belongs*. They are one ISMRMRD
`EncodingCounters` field each, and they come from a scan loop's `EncodingAxis`:
`loop.label_state(shot)` returns `{counter: value}`, ready to spread into
`set_state`. `LIN`/`PAR` reach the readout as `lin_idx`/`par_idx` and it emits
them itself; `ECO` is the readout's own, because a multiecho train's echoes are
blocks of one shot rather than iterations of a loop.

```python
excitation.set_state(freq_offset_hz=offsets[s], **slices.label_state(s), **frames.label_state(f))
```

**Flags** — `NOROT`, `NOPOS`, `NOSCL`, `PMC`, `NAV`, `REV`, `SMS`, `REF`,
`IMA`, `NOISE`, `OFF`, `ONCE`, `TRID` — say *how a block is played or
classified*:

```python
readout.set_state(lin_idx=ky, adc_flag=False)   # play the ADC, discard the data
excitation.set_state(once=1, TRID=2)            # a preparation TR, in safety group 2
fatsat.set_state(NOPOS=1, NOROT=1)              # exempt from the FOV transform
```

`adc_flag=False` is `OFF=1` and `once=` is `ONCE`, spelled the way a loop reads.

Pulseq labels are sticky — a value set at one block persists until some later
block sets it again — so a flag has to be *scoped* or it leaks into whatever
the sequence plays next. Scoping is by default: the value is emitted on the
module's first block and `0` on its last. The two flags that deliberately
outlive their module are exempt, and listed in `STICKY_FLAGS`: `ONCE` delimits
a whole preparation or cooldown *section*, and `TRID` names a repeating unit —
a TR, or a whole contrast — which is also the group the safety model checks
SAR over. Pass `flag_scope="sticky"` or `flag_scope="module"` at construction
to override.

A label is **structure** from the moment it is first named: the event carrying
it is built once and only its value moves afterwards, which is what lets a TR
template recognise the block as its own. So label values *persist* across
`set_state` calls — pass `0` to clear one, and a call naming only labels leaves
the numbers alone. They are emitted in the order they were declared. Declare
them with the module to have them there from the start:

```python
readout = design.make_line_readout(system, fov, matrix, labels=("SLC", "REP"))
```

**Triggers and digital outputs** are ordinary block events, but which block
they belong on is a property of the module's *design* — a cardiac trigger gates
the excitation that opens a shot, a scope sync pulse marks the readout that must
be captured — so they are declared with it rather than set per shot:

```python
readout = design.make_line_readout(
    system, fov, matrix,
    triggers=(pp.make_trigger("physio1", duration=100e-6),),           # first block
)
scoped = design.make_line_readout(
    system, fov, matrix,
    triggers={-1: (pp.make_digital_output_pulse("osc0", duration=100e-6),)},   # last block
)
```

`labels=`, `flags=`, `flag_scope=` and `triggers=` reach a readout through any
`make_*_readout` factory. The RF and preparation factories do not forward them
yet — declare counters and flags on those with `set_state` before the sequence
is built, which is where the template records them.

### First/last-in-axis MRD flags

`FIRST_IN_ENCODE_STEP1`, `LAST_IN_SLICE`, `LAST_IN_REPETITION` and the rest are
not written by the sequence. The interpreter derives them by comparing each
acquisition's counter against the *observed* range of that counter over the
scan. Emitting the counters is therefore the whole mechanism: a dimension that
is looped but never labelled collapses to a single index, and both its flags
fire on every acquisition. `ScanLoop.label_limits()` reports the ranges those
flags will be derived from.

## View orderings

`ScanLoop` separates the positions visited from the order they are visited in,
and an `EncodingAxis` says what the visited numbers mean. See the [scan-loop
reference](sampling.md) for absolute FSE trains, relative EPI shifts,
non-Cartesian tilts, slice/SMS grouping, and frame/contrast counters.

The echo-train / segment view-ordering helpers apply to any acquisition
with an outer loop and an inner echo train or MPRAGE segment — 2D/3D FSE and
segmented GRE alike. They take phase-encode locations in the (ky, kz) plane
and return a list of shots (each an ordered list of view indices; the
echo/segment index is the position within the shot):

A plugin reaches them through the `ordering=` argument of a scan-loop
factory — or of {meth}`~pulserver.ScanLoop.from_mask`, when the support is
hand-built — rather than by calling them:

- `make_linear_order` — raster (kz-major) linear reordering.
- `make_radial_order` — center-out radial (wedge) reordering.
- `make_radial_adaptive_order` — adaptive radial reordering with per-shot
  parameter support.
- `make_shuffling_order` — randomly shuffled (T2-Shuffling) reordering with
  spatial clustering to limit gradient switching.

References: Buonincontri et al., *Doubling the repetition time without paying
the price: 3D TSE with individually parameterized echo trains*, ISMRM
566-05-007 ([`refcode/Abstract 566-05-007.pdf`](../../../../refcode/)) for the
linear / radial / adaptive schemes; Tamir et al., *T2 Shuffling*, Magn Reson
Med 2017;77:180–195 ([`refcode/nihms804984.pdf`](../../../../refcode/)) for
random shuffling.

## Advanced excitation

Beyond the slice-selective and hard builders,
`make_frequency_selective_pulse` creates spectrally selective pulses and the
spatial factories accept existing or generated multidimensional trajectories.
All return the common `pulserver.SequenceModule` protocol.

## API documentation rendering

The Sphinx API reference is configured in `docs/conf.py`; build it with
`sphinx-build -E -W -b html docs docs/_build/html`.
