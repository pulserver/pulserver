---
orphan: true
---

# Internal ISMRMRD session contract

This page records the current private transport implementation. It is not a
supported public API; the scientific reconstruction surface is documented in
{doc}`../../api/recon`.

Pulseq records the events required to play a scan; a reconstruction stream
also needs encoding geometry, acquisition labels, trajectories and enough RF
context to interpret contrast. Pulserver derives those products while the
sequence is converted (see {doc}`../sequence_representation/pulserver`) and
preserves them in the cache, rather than asking the reconstruction side to
rediscover them from raw data.

## What the session looks like

A reconstruction connection is an ISMRMRD session in the literal, protocol
sense: `pulserver.recon._mrd.connection.Connection` opens with an XML header,
streams a sequence of acquisitions and optional waveforms, then closes.

- **The header** carries encoding spaces (encoded/recon matrix and FOV per
  space), sequence parameters and vendor/protocol metadata. It is built once,
  before the first acquisition, from the cache's `COMMON`/`DEFINITIONS`
  sections plus derived trajectory metadata (see
  {doc}`trajectory`) — not from the raw data stream.
- **Each acquisition** carries its labels (encoding indices, slice, average,
  segment...), flags (first/last in slice, measurement, etc.), sample timing,
  centre sample and — for non-Cartesian encoding — its trajectory, attached
  per-readout rather than left for the reconstruction to recompute.
- **Waveforms** are the extension channel: physiological traces when
  present, and — as a deliberate reuse of the same ISMRMRD payload type —
  the `SEQDESC` event list and RF library described in
  {doc}`simulator`.

This makes the receiving handler independent of the scanner's native raw-data
layout: `Connection` and the private reader/writer classes speak ISMRMRD
messages (`MessageType` in `pulserver/recon/_mrd/connection.py`), not a vendor frame
format, so a handler written against this contract does not change when the
frame format on the wire does.

## Why the contract is enforced, not advisory

The public cache reader (`cxx/recon/trajectory_cache_reader.*`) is the
boundary: a reader consumes `DEFINITIONS`, `TRAJECTORY` and `SEQDESC`
together, and both `TRAJECTORY` and `SEQDESC` are **mandatory** cache
sections (see {doc}`../sequence_representation/pulserver`). A missing or
malformed section is therefore a failed conversion at predownload time, not
a degraded reconstruction mode discovered later — the same "reject early,
scanner stays authoritative" posture the safety gates take
({doc}`../safety/gradient_slew`).
