# Reconstruction metadata

Pulseq records the events required to play a scan, but a reconstruction stream
also needs encoding geometry, acquisition labels, trajectories and enough RF
context to interpret contrast. Pulserver derives those products while the
sequence is converted, then preserves them in the cache rather than attempting
to rediscover them from raw data.

## ISMRMRD session contract

The reconstruction connection opens with an ISMRMRD XML header, streams
acquisitions and optional waveforms, then closes. The header carries encoding
spaces and sequence parameters. Each acquisition carries its labels, flags,
sample timing, centre sample and—when non-Cartesian—its trajectory. This makes
the receiving handler independent of the scanner's native raw-data layout.

## Enrichment from the sequence cache

`TRAJECTORY` contains canonical k-space shots, instance amplitudes, rotations,
ADC labels and encoding-space membership. The recon reader precomputes the
per-encoding-space trajectories, applies the stored rotation, and attaches the
result to every applicable ISMRMRD acquisition. Cartesian acquisitions retain
their labels and geometry but need no explicit trajectory array.

`SEQDESC` contains per-subsequence RF definitions and a compact event list of
wait, RF and ADC events. It is emitted as custom ISMRMRD waveform payloads
alongside the standard session metadata. A Python-side decoder is required for
handlers that consume this extension; it should treat these waveform payloads
as versioned sequence-description data, not as physiological samples.

## State-machine simulation

The event list is deliberately smaller than a sampled waveform. A Bloch
state-machine simulator advances through waits, RF events and acquired ADC
events, using the RF shape library and the event's amplitude, phase, frequency
and shim state. This permits sequence-aware signal simulation without a
vendor-specific pulse-program representation.

The public cache reader is the contract boundary: readers must consume
`DEFINITIONS`, `TRAJECTORY` and `SEQDESC` together. `TRAJECTORY` and `SEQDESC`
are mandatory cache sections, so a missing or malformed section is a failed
conversion, not a degraded reconstruction mode.
