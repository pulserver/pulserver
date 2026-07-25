# Pulseq enrichment and automatic trajectory calculation

Pulseq's `.seq` file, and PulSeg's block/segment representation built from
it, describe how to *play* a sequence. Neither says what an acquired sample
means for reconstruction: which encoding space it belongs to, where it sits
in k-space, or which ISMRMRD flags it should carry. `cxx/recon/trajectory_cache_reader.*`
derives all of that from the cache the same conversion pass already
produced, and attaches it to the ISMRMRD session described in
{doc}`ismrmrd_session` — hence *enrichment*: the acquisition stream is
Pulseq's, decorated with what reconstruction needs, not recomputed from
scratch or authored separately.

## The same definition/instance split, applied to k-space

The trajectory representation mirrors
{doc}`../sequence_representation/pulseg`'s base-block deduplication, for the
same reason: k-space shots repeat far more than they differ.

- **`Kshot`** is a canonical trajectory — one shot's k-space samples
  (Hz·s/m), computed once from a gradient definition's shape.
- **`TrajTableEntry`** is one ADC occurrence: which `Kshot` id each axis
  uses, that occurrence's per-instance gradient amplitude and rotation id,
  its full label set (slice, phase, repetition, average, segment, set, echo,
  partition, line, acquisition), its ISMRMRD flags, centre sample, dwell
  time, and which encoding space it belongs to.

`pre_compute_trajectories(cache)` walks the cache once and produces exactly
these two tables. Reconstructing sample $n$'s physical k-space location is
then "look up its `Kshot`, apply its instance's scale and rotation" — the
same table-lookup-plus-transform pattern the safety checks use
({doc}`../safety/gradient_slew`), not a re-expansion of the gradient
waveform per acquisition.

**Encoding-space membership is explicit, not inferred from geometry.** An
`EncodingSpace` entry ties a subsequence (and, for navigators, a
`nav_subseq_offset`) to a `geometry_tag` (primary vs. navigator) and its
label limits; FOV and matrix size are read from the cache's own
`DEFINITIONS` rather than duplicated into the trajectory tables. A
collection with a navigator interleaved into the main acquisition — or
several distinct subsequences, as in {doc}`../sequence_representation/pulseg`'s
segment-dedup discussion — therefore reports multiple encoding spaces
cleanly, each with its own matrix/FOV, instead of forcing one global guess.

**Cartesian acquisitions need no trajectory array at all.** `Kshot`/
`TrajTableEntry` exist to serve non-Cartesian encoding — radial, spiral,
rosette — where sample position is not derivable from label indices alone.
A Cartesian acquisition retains its labels and geometry through the same
`TrajTableEntry` row, but `enrich_ismrmrd_acquisition` leaves the trajectory
field empty for it: the reconstruction already has everything it needs from
`ky`/`kz`/`slice` labels and the encoding space's matrix size.

## What gets attached, and when

`enrich_ismrmrd_header(hdr, cache)` runs once, before the first acquisition:
it populates the ISMRMRD XML header's encoding spaces, sequence parameters
and resource paths (`add_sequence_resource_paths`) from the cache's
`COMMON`/`DEFINITIONS` sections and the precomputed trajectory tables above.
`enrich_ismrmrd_acquisition(...)` then runs per acquisition, attaching that
occurrence's labels, flags, centre sample and (when applicable) its
rotated/scaled trajectory. Both draw only from the `TRAJECTORY` cache
section and the tables derived from it — never from re-expanding the
gradient waveform — so enrichment cost is a table lookup per acquisition,
not a per-sample recomputation.
