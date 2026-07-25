# Sequence description and the state-machine simulator extension

`TRAJECTORY` ({doc}`trajectory`) answers "where in k-space is this sample."
It does not answer "what was the magnetisation state when it was acquired" —
that needs the RF history, not just the gradient history. `SEQDESC` is the
second, independent cache section that carries exactly that: a compact event
list plus an RF shape library, small enough to ship in full rather than
sampled.

## Why an event list instead of a sampled waveform

A sampled RF/gradient waveform at the raster is precise but large, and most
of it is redundant with information the cache already has elsewhere (block
definitions, instance tables). `SEQDESC` instead stores, per subsequence
(`SequenceDescription` in `cxx/recon/trajectory_cache_reader.h`, emitted by
`csrc/src/structure/pulseg_seqdesc.c`):

- **`events`** — one `SeqEvent` per WAIT, RF or ADC occurrence, each a type
  tag, a `timestamp_us`, and up to seven `params`: for RF, the definition id,
  RF use, actual amplitude, phase offset, frequency offset, shim id and
  slice-select gradient amplitude; for ADC, the role (single echo, echo
  centre, non-centre, non-acquired) and phase offset.
- **`rf_defs`** — a per-subsequence RF-definition library (`RfDef`): each
  entry's bandwidth, multi-band frequency offsets, total $B_1^2$ power, and
  its magnitude/phase/time samples, still delta-RLE compressed exactly as
  `SEQDESC` stores them.
- **`rf_shape_tuples`** — deduplicated (definition, shim, slice-select
  gradient) triplets, carrying the derived slice thickness and a
  slice-selectivity flag, so a consumer does not have to re-derive slice
  geometry from bandwidth and gradient amplitude itself.

This is the same economy of representation as everywhere else in the stack:
an event references a definition by id rather than re-embedding its shape,
so the list scales with the number of RF/ADC/wait *occurrences* in one
subsequence, not with the scan's sample count.

## Custom Waveform types: how it travels

`SEQDESC` is not a new ISMRMRD message type. It is emitted as **custom
`ISMRMRD::Waveform` payloads**, alongside the session's ordinary metadata
(see {doc}`ismrmrd_session`), by four dedicated builders in
`cxx/recon/trajectory_cache_reader.h`: `make_seqdesc_header_waveform`,
`make_seqdesc_events_waveform`, `make_seqdesc_rf_shapes_waveform` and
`make_seqdesc_shims_waveform`. Reusing the `Waveform` payload type — rather
than inventing a new ISMRMRD message — means any receiver already able to
read physiological waveforms can carry `SEQDESC` through unmodified; only a
handler that wants to *interpret* the payload needs to know its schema, and
that handler must treat it as versioned sequence-description data, not as
physiological samples (`add_waveform_information` tags each waveform so the
two are never confused on the wire).

## The state machine: what a consumer would do with it

The event list is deliberately shaped for one specific kind of consumer: a
state-machine walk that advances a per-isochromat (or per-voxel) signal
state event by event — free precession across a `WAIT`, an instantaneous (or
shaped) rotation at an `RF`, a state readout at an `ADC` — using each RF
event's amplitude/phase/frequency/shim and the matching `RfDef` shape. That
walk is a Bloch simulation driven by the compact event stream instead of by
a raster-sampled pulse program, which is what makes it vendor-neutral: the
event list carries no scanner-specific pulse-sequence representation, only
the physics-relevant parameters.

```{image} ../assets/reconstruction/seqdesc_state_machine.png
:alt: Schematic of the SEQDESC event stream and per-event state advance
:width: 100%
```

*Schematic, not measured data* — there is deliberately no simulated signal
plotted here. **This consumer does not exist yet.** `SEQDESC` emission
(the C structures and the four `Waveform` builders above) is implemented and
shipped; a Python-side decoder for those payloads, and a state-machine
simulator that steps through the resulting event stream, are not — neither
`external/pulserver` nor `pulserver-interpreter` currently contains one. The
data contract above is what such a consumer would read; building it is
future work, not a documented-but-hidden feature.
