# The reconstruction side

The scanner's reconstruction client streams each series as MRD: a
configuration text, the XML header, and the acquisitions. What a vendor
reconstruction client puts in them does not describe a Pulseq sequence: the
encoding counters, the trajectory and the encoding spaces are those of the
interpreter's own acquisition loop. The reconstruction proxy replaces them with
what the sequence states.

## Enrichment

The header names the revision the series was played from (see
{doc}`sessions`). The proxy tabulates the readouts of that revision's sequence
chain in play order ({class}`~pulserver.vre.SequenceTable`) and applies the
table to the stream:

- the header receives an encoding space for the imaging readouts of each
  subsequence, and one for its navigator readouts when it has any, with the
  matrix and field of view the sequence defines and encoding limits from the
  counters its readouts reach;
- each acquisition, matched to its row by position in the stream, receives the
  encoding counters, the MRD flags, the dwell time and the encoding space
  reference, and the k-space trajectory when k moves across the readout.

The flags are those the sequence's labels set, such as `IS_NAVIGATION_DATA`,
and the first-and-last flags (`FIRST_IN_SLICE`, `LAST_IN_SLICE` and the
others), derived from the counters. A reconstruction plugin routes on them (see
{doc}`../user-guide/reconstruction-plugins`).

When the header carries the `pulserver_fov_offset_mm` user parameter, the
prescription centre in mm along the sequence's gradient axes, each readout is
multiplied by `exp(+i 2π d·k)`, which moves an object at `d` to the centre of
the field of view. The trajectory is left unchanged.

## Routing

Each series runs on a worker process with the reconstruction plugin the
sequence names, or the one the client's configuration names when the sequence
names none. A worker serves one series and exits, so the memory a
reconstruction holds on the host and on a GPU is released with it. Importing a
reconstruction stack takes seconds, so the proxy keeps spare workers that have
already imported it waiting for a series.

The number of series reconstructed at once is bounded by slots, derived from
the available memory unless given. A series that finds every slot busy is
written to disk as it arrives, with its enrichment applied, and replayed to a
worker once a slot frees; the client stays connected, and the images return
through its connection.

Images, DICOM datasets and text from the worker are relayed to the client as
they arrive. Closing either end closes the other.
