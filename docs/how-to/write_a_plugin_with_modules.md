# How to write a sequence plugin with the design toolbox

This guide builds a complete 2D multi-slice GRE plugin — RF-spoiled,
slice-selective, optionally accelerated with ACS lines — out of shipped
modules and sampling objects. It assumes you can already write the same
sequence in plain PyPulseq; if you cannot, start from
[`examples/sequences/gre_2d.py`](https://github.com/INFN-MRI/pulserver/tree/main/examples/sequences/gre_2d.py),
which does exactly that.

**Nothing here replaces the way you already write PyPulseq.** `seq` is still a
`Sequence`, blocks still go in with `seq.add_block`, and the loop nesting is
still your own `for` statements — which is what keeps a preparation, a trigger
or a dummy TR insertable at any level. Two things change: the events you would
have built and mutated by hand come pre-assembled as modules, and the index
bookkeeping comes from a loop object instead of `range()`.

| Object | Answers | Replaces |
| --- | --- | --- |
| `SequenceModule` (from any `make_*_pulse` / `make_*_readout`) | *what do I play?* | the group of events you built once and mutated per shot |
| `ScanLoop` (from any `make_*_sampling` / `make_slice_loop` / `make_counter_loop`) | *what varies, in what order?* | `range(Ny)`, `phase_areas[i]`, the slice-offset arithmetic, the label bookkeeping |

There is one loop type, not one per axis: a `ScanLoop` is a table of
**positions**, a grouping of them into **shots**, and one counter label per
column. That describes k-space, slices, frames and contrasts equally well —
only the units differ, and the axis declaration says which they are.

## 1. Design every module once

Modules build their waveforms in the constructor. Build them outside the loop
and never rebuild them:

```python
import numpy as np
import pulserver.design as design
import pulserver.pypulseq as pp

excitation = design.make_slice_selective_pulse(np.deg2rad(flip_deg), thickness_m, system=opts)
readout = design.make_line_readout(
    opts, (fov_m, phase_fov_m), (nx, ny), spoil_position="post", spoil_cycles=1.0
)
```

`spoil_position="post"` puts the gradient spoiler inside the readout, on the
readout axis, after the last echo. Together with the 117° RF spoiling below
that is the whole spoiling scheme — no separate slice-axis crusher is needed.
Add one only if you have a reason to dephase along `z` as well; it would be a
bare Pulseq event, not a module, so `seq.add_block(crusher)` works directly.

### Give the readout the slice rephaser

The excitation ends with a rephasing block that cancels the selection moment
accrued after the RF isocentre. The readout is about to play a prewinder
anyway, so hand the rephaser over and save its block from every TR:

```python
readout = design.make_line_readout(
    opts, (fov_m, phase_fov_m), (nx, ny),
    spoil_position="post", spoil_cycles=1.0,
    slice_rephasing=excitation.rephasers,
)
excitation = excitation.without_rephasers()
```

Both lines matter. `slice_rephasing` moves the moment into the prewinder;
`without_rephasers()` stops the excitation playing it as well. Skip the second
and the slice is rephased twice — every shot ends with a net selection moment,
which costs signal without failing any timing check. The rephasers stay
readable on the returned module, so the two lines can be written in either
order.

Do this before the TE budget in step 3: `excitation.duration` and
`readout.t_prephase_s` both change, and the whole point is that the pair is
shorter than it was.

`make_line_readout`, `make_epi_readout` and every non-Cartesian factory take
`slice_rephasing`. On an axis the readout already encodes — kz on a stack, the
partition on 3D EPI — the moment is summed into that encode rather than played
beside it, since one channel carries one gradient per block; only the encoding
half is unwound afterwards, because a rephaser is a one-way moment. A rephaser
on the readout axis itself is rejected: that waveform is fixed by the
acquisition window.

Two families do not take it. `make_bssfp_readout` and `make_zte_readout` take
the whole `excitation` module instead and handle the rephaser themselves;
`make_fse_readout` accepts no rephaser at all, because its prewinder sits after
the first refocusing pulse, too late to rephase the excitation.

## 2. Plan k-space and slices separately

```python
sampling = design.make_cartesian_sampling((nx, ny), acceleration=ry, calibration=acs_lines)
slices = design.make_slice_loop(n_slices, spacing_m)
```

`make_cartesian_sampling` knows nothing about slices, and `make_slice_loop`
knows nothing about k-space. Neither knows how they are nested — that is your
`for` statement, which is what lets you put a preparation or a trigger at
whichever level you need.

Frames, contrasts and averages are the same type again, from
`make_counter_loop`; this plugin has none, but adding one is one more line and
one more `for`:

```python
frames = design.make_counter_loop(n_frames, label="REP")
```

Slice positions are in metres. In plain PyPulseq you would write the RF offset
of slice `s` as

```python
rf.freq_offset = gz.amplitude * (s - (n_slices - 1) / 2) * spacing_m
```

`to_frequencies` is that same multiplication for the whole loop at once, so
the layout arithmetic stops being yours:

```python
offsets_hz = slices.to_frequencies(excitation.gradients[0].amplitude)
```

It returns **one entry per physical slice**, indexed by slice number — not by
the order they are played in. That distinction matters as soon as you leave
`order="sequential"`, and step 4 says how to use it.

## 3. Budget TE and TR from module durations

Every module reports `duration` for its current block snapshot, and readouts
additionally publish the timing landmarks a TE budget needs
(`t_first_echo_s`, `esp`, `t_prephase_s`):

```python
raster = opts.block_duration_raster

rf_center_s = pp.calc_rf_center(excitation.rf)[0] + excitation.rf.delay
min_te_s = (excitation.duration - rf_center_s) + readout.t_first_echo_s
te_delay_s = round((te_s - min_te_s) / raster) * raster

shot_s = excitation.duration + max(te_delay_s, 0.0) + readout.duration
tr_delay_s = round((tr_s - n_slices * shot_s) / raster) * raster
```

A negative `te_delay_s` or `tr_delay_s` is the answer `validate_protocol`
returns to the scanner UI. Do this arithmetic in one helper and call it from
both `validate_protocol` and `make_sequence` so the two can never disagree.

## 4. Write the loop

The loop is ordinary Python, and it has the same shape as the PyPulseq one you
would have written by hand. Here is the PyPulseq original, from `gre_2d.py`:

```python
rf_phase, rf_inc = 0.0, 0.0
for i in range(n_pe):
    for s in range(n_slices):
        rf.freq_offset = gz.amplitude * (s - (n_slices - 1) / 2) * spacing_m
        rf.phase_offset = adc.phase_offset = np.deg2rad(rf_phase)
        rf_inc = (rf_inc + 117.0) % 360.0
        rf_phase = (rf_phase + rf_inc) % 360.0

        seq.add_block(rf, gz)
        seq.add_block(gx_pre, pp.scale_grad(gy_pre, scales[i]), gz_reph)
        seq.add_block(pp.make_delay(te_delay_s))
        seq.add_block(gx, adc)
    seq.add_block(pp.make_delay(tr_delay_s))
```

and here is the same loop with modules. Two nested `for`s over the same two
counters; each event group becomes one `set_state` plus one iteration:

```python
phases = design.make_rf_spoiling_schedule(len(sampling) * len(slices), increment=np.deg2rad(117.0))
i = 0

for lin_idx in sampling.flatten()[:, 0].astype(int):
    for s in range(len(slices)):
        phase = float(phases[i])
        i += 1

        excitation.set_state(freq_offset_hz=float(offsets_hz[s]), phase_offset_rad=phase)
        excitation.set_labels(SLC=s)
        for block in excitation:
            seq.add_block(*block)

        if te_delay is not None:
            seq.add_block(te_delay)

        readout.set_state(lin_idx=lin_idx, phase_offset_rad=phase)
        for block in readout:
            seq.add_block(*block)

    if tr_delay is not None:
        seq.add_block(tr_delay)
```

Line for line against the PyPulseq version:

| PyPulseq | Here |
| --- | --- |
| `for i in range(n_pe)` | `for lin_idx in sampling...` — the loop knows which lines are sampled, so acceleration and ACS need no new code |
| `rf.freq_offset = gz.amplitude * ...` | `freq_offset_hz=offsets_hz[s]` — the same product, computed once in step 2 |
| `rf_phase = (rf_phase + rf_inc) % 360` | `phases[i]` — the same quadratic schedule, precomputed |
| `pp.scale_grad(gy_pre, scales[i])` | folded into `readout.set_state(lin_idx=...)` |
| `seq.add_block(rf, gz)` | `for block in excitation: seq.add_block(*block)` |

Three things differ from PyPulseq, and each buys something:

- **`set_state` replaces the module's whole dynamic state.** Anything you leave
  out reverts to its default. PyPulseq's `rf.freq_offset = ...` mutates an
  object that survives the iteration, so a value set three shots ago can still
  be in effect; `set_state` makes that impossible.
- **`set_labels` writes the counters the *loop* owns** — `SLC` here, and `REP`,
  `SET`, `PHS`, `AVG` when those loops exist. It is the module-level spelling
  of `seq.add_block(..., pp.make_label(type="SET", label="SLC", value=s))`,
  and it exists because the label has to land on the module's *first* block,
  which you would otherwise have to unpack the snapshot to reach. The keyword
  form above is the short one; `set_labels(*slices.labels(s))` is the same
  thing sourced from the loop, which is what you want as soon as the slice
  order is not `sequential`.
- **`lin_idx` doubles as the `LIN` label.** The readout emits `SET LIN` from
  the index you pass, so the reconstruction sees exactly what was played and
  you never write a `LIN` label yourself.

When a module needs no per-block handling, `add_to` collapses its two lines
into one:

```python
readout.set_state(lin_idx=lin_idx, phase_offset_rad=phase).add_to(seq)
```

Modules are deliberately **not** callable. `set_state` then iterate is the only
path, so the names you read in `set_state` are the names you use.

### Label every loop, even the ones you index yourself

The MRD `FIRST_IN_*` / `LAST_IN_*` flags are derived downstream from the
*observed range* of each counter. An unlabelled frame loop therefore reports
one repetition, and every acquisition comes back flagged both first and last in
it. `slices.label_limits()` shows what the reconstruction will see.

### Two idioms you will meet in the shipped plugins

**Shot objects instead of bare indices.** Iterating a `ScanLoop` yields each
shot as an array of positions rather than an integer, because one shot is one
excitation's worth of the loop — a single line for GRE, a whole echo train for
FSE, an SMS band group for a multiband slice loop. For the single-position case
above, `int(shot[0, 0])` is the index:

```python
for shot in sampling:
    readout.set_state(lin_idx=int(shot[0, 0]))
```

and `slices.shots[s]` is the *slice numbers* of excitation `s`, which is what
`offsets_hz[...]` must be indexed by when the slice order is interleaved, or
when `sms_factor > 1` puts several bands in one shot:

```python
for s, band in enumerate(slices.shots):
    excitation.set_state(freq_offset_hz=offsets_hz[band])   # one entry, or one per band
    excitation.set_labels(*slices.labels(s))
```

**`iter()` / `next()` instead of an index.** `phases[i]` above needs you to
carry `i`. Consuming a schedule as an iterator drops the counter and, more
importantly, keeps *one* chronology running across every nesting level — which
is what a golden-angle acquisition needs, so that every frame sees fresh
angles rather than replaying the same set:

```python
phases = iter(design.make_rf_spoiling_schedule(total_shots, increment=np.deg2rad(117.0)))
rotations = iter(design.make_noncartesian_2d_sampling(matrix, views=total_shots).to_rotations())

for frame in range(n_frames):
    for s in range(len(slices)):
        excitation.set_state(phase_offset_rad=float(next(phases))).add_to(seq)
        readout.set_state(lin_idx=s, rotation=next(rotations)).add_to(seq)
```

Re-`iter()` inside a loop instead, and each outer position replays the same
set. Use whichever reads better: `phases[i]` when the schedule is per-shot and
the index is already in hand, `next(phases)` when the chronology must span
loops you would otherwise have to flatten by hand.

## 5. Set definitions and write

```python
seq.set_definition("Name", "gre_2d")
seq.set_definition("FOV", [fov_m, phase_fov_m, thickness_m * n_slices])
seq.set_definition("TE", te_s)
seq.set_definition("TR", tr_s)
seq.set_definition("Nx", nx)
seq.set_definition("Ny", ny)
seq.set_definition("NySampled", len(sampling))
seq.set_definition("NumSlices", n_slices)

pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)
```

`len(sampling)` is the shot count, so `TR * len(sampling)` is the scan time
`validate_protocol` reports.

## Check it before you ship it

Two checks catch most mistakes without a scanner.

Inspect one shot of any module on its own — Pulserver's `Sequence` is a
write-only fast builder, so `plot()` replays the snapshot into an upstream
sequence for you:

```python
readout.set_state(lin_idx=0).plot()
```

Then verify the k-space you *planned* is the k-space you *played*, by
replaying the loop into an upstream `pypulseq.Sequence` — the only one that
can be read back:

```python
import pypulseq

check = pypulseq.Sequence(system=opts)
...                                          # same loop, small matrix
assert check.check_timing()[0]

k_adc, *_ = check.calculate_kspace()
played = np.round(np.unique(np.round(k_adc[1] * phase_fov_m, 6)), 3)
planned = np.array(sorted(int(v) - ny // 2 for v in sampling.flatten()[:, 0]), dtype=float)
assert np.array_equal(played, planned)
```

This is worth the few lines: it is the one check that catches a sign error, a
double-applied rephaser or a mis-indexed encode, none of which fail
`check_timing`.

## The complete plugin

```python
"""Cartesian 2D GRE, built from Pulserver modules and scan loops."""

from __future__ import annotations

import numpy as np
import pulserver.design as design
import pulserver.io as pio
import pulserver.pypulseq as pp
from pulserver import (
    DropdownFloatParam,
    DropdownIntParam,
    Sequence,
    UIParam,
    dict_to_protocol,
    params,
    protocol_to_dict,
)

RF_SPOIL_INCREMENT_RAD = np.deg2rad(117.0)


def _build(opts, prot):
    """Design every module and sampling object once, from a protocol."""
    fov_m = params.param_float(prot, UIParam.FOV) * 1e-3
    phase_fov_m = params.phase_fov_mm_from_protocol(prot) * 1e-3
    thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    nx = params.param_int(prot, UIParam.NX)
    ny = params.param_int(prot, UIParam.NY)
    n_slices = params.param_int(prot, UIParam.NSLICES)
    flip_rad = np.deg2rad(params.param_float(prot, UIParam.FLIP))
    ry = max(1, int(round(params.param_float_optional(prot, UIParam.RY, 1.0))))
    acs = params.acs_lines_from_protocol(prot, ny, 0)

    excitation = design.make_slice_selective_pulse(flip_rad, thickness_m, system=opts)
    readout = design.make_line_readout(
        opts,
        (fov_m, phase_fov_m),
        (nx, ny),
        spoil_position="post",
        spoil_cycles=1.0,
        slice_rephasing=excitation.rephasers,
    )
    excitation = excitation.without_rephasers()

    return {
        "excitation": excitation,
        "readout": readout,
        "sampling": design.make_cartesian_sampling((nx, ny), acceleration=ry, calibration=acs),
        "slices": design.make_slice_loop(n_slices, spacing_m or thickness_m, order="sequential"),
        "fov": (fov_m, phase_fov_m, thickness_m),
        "n_slices": n_slices,
    }


def _timing(opts, parts, te_s, tr_s):
    """Turn requested TE/TR into the two delays that realise them."""
    excitation, readout = parts["excitation"], parts["readout"]
    raster = opts.block_duration_raster

    rf_center_s = pp.calc_rf_center(excitation.rf)[0] + excitation.rf.delay
    min_te_s = (excitation.duration - rf_center_s) + readout.t_first_echo_s
    te_delay_s = round((te_s - min_te_s) / raster) * raster

    shot_s = excitation.duration + max(te_delay_s, 0.0) + readout.duration
    tr_delay_s = round((tr_s - parts["n_slices"] * shot_s) / raster) * raster

    return te_delay_s, tr_delay_s, shot_s


class GreSequence(Sequence):
    def get_default_protocol(self, opts):
        del opts
        return protocol_to_dict(
            {
                UIParam.TE: DropdownFloatParam(value=8.0, min=2.0, max=80.0, incr=0.1, unit="ms"),
                UIParam.TR: DropdownFloatParam(value=250.0, min=20.0, max=5000.0, incr=0.1, unit="ms"),
                UIParam.FLIP: DropdownFloatParam(value=12.0, min=1.0, max=90.0, incr=1.0, unit="deg"),
                UIParam.FOV: DropdownFloatParam(value=220.0, min=80.0, max=500.0, incr=1.0, unit="mm"),
                UIParam.PHASE_FOV: DropdownFloatParam(value=220.0, min=80.0, max=500.0, incr=1.0, unit="mm"),
                UIParam.SLICE_THICKNESS: DropdownFloatParam(value=5.0, min=1.0, max=20.0, incr=0.5, unit="mm"),
                UIParam.SLICE_SPACING: DropdownFloatParam(value=5.0, min=1.0, max=20.0, incr=0.5, unit="mm"),
                UIParam.NX: DropdownIntParam(value=64, min=16, max=512, incr=1),
                UIParam.NY: DropdownIntParam(value=64, min=8, max=512, incr=1),
                UIParam.NSLICES: DropdownIntParam(value=5, min=1, max=128, incr=1),
            }
        )

    def validate_protocol(self, opts, protocol):
        prot = dict_to_protocol(protocol)
        te_s = params.param_float(prot, UIParam.TE) * 1e-3
        tr_s = params.param_float(prot, UIParam.TR) * 1e-3

        parts = _build(opts, prot)
        te_delay_s, tr_delay_s, shot_s = _timing(opts, parts, te_s, tr_s)
        if te_delay_s < 0.0:
            return {"valid": False, "duration": None, "info": "TE too short for this readout"}
        if tr_delay_s < 0.0:
            n_max = int(tr_s / shot_s)
            return {"valid": False, "duration": None, "info": f"TR fits only {n_max} slice(s)"}

        duration_s = tr_s * len(parts["sampling"])
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.1f} s"}

    def make_sequence(self, opts, protocol, output_path):
        prot = dict_to_protocol(protocol)
        te_s = params.param_float(prot, UIParam.TE) * 1e-3
        tr_s = params.param_float(prot, UIParam.TR) * 1e-3

        parts = _build(opts, prot)
        excitation, readout = parts["excitation"], parts["readout"]
        sampling, slices = parts["sampling"], parts["slices"]
        te_delay_s, tr_delay_s, _ = _timing(opts, parts, te_s, tr_s)

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0 else None
        offsets_hz = slices.to_frequencies(excitation.gradients[0].amplitude)
        phases = design.make_rf_spoiling_schedule(
            len(sampling) * len(slices), increment=RF_SPOIL_INCREMENT_RAD
        )

        seq = pp.Sequence(opts)
        i = 0
        for lin_idx in sampling.flatten()[:, 0].astype(int):
            for s in range(len(slices)):
                phase = float(phases[i])
                i += 1

                excitation.set_state(freq_offset_hz=float(offsets_hz[s]), phase_offset_rad=phase)
                excitation.set_labels(SLC=s)
                excitation.add_to(seq)

                if te_delay is not None:
                    seq.add_block(te_delay)

                readout.set_state(lin_idx=int(lin_idx), phase_offset_rad=phase).add_to(seq)

            if tr_delay is not None:
                seq.add_block(tr_delay)

        fov_m, phase_fov_m, thickness_m = parts["fov"]
        seq.set_definition("Name", "gre_2d")
        seq.set_definition("FOV", [fov_m, phase_fov_m, thickness_m * parts["n_slices"]])
        seq.set_definition("TE", te_s)
        seq.set_definition("TR", tr_s)
        seq.set_definition("Nx", params.param_int(prot, UIParam.NX))
        seq.set_definition("Ny", params.param_int(prot, UIParam.NY))
        seq.set_definition("NySampled", len(sampling))
        seq.set_definition("NumSlices", parts["n_slices"])
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


PLUGIN = GreSequence()


def get_default_protocol(opts):
    return PLUGIN.get_default_protocol(opts)


def validate_protocol(opts, protocol):
    return PLUGIN.validate_protocol(opts, protocol)


def make_sequence(opts, protocol, output_path):
    return PLUGIN.make_sequence(opts, protocol, output_path)
```

## Next

- [Customise a plugin](customise_a_plugin.md) — swap the excitation, add a
  preparation, change the sampling.
- [Write a new module or loop structure](write_a_new_module.md) — when the
  shipped families do not cover your sequence.
- [Scan-loop reference](../reference/sampling.md) — every loop factory and
  what its shots contain.
