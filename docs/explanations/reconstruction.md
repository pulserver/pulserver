# Raw-data enrichment and routing

The scanner's reconstruction client streams each series as MRD: a configuration
text, the XML header and the acquisitions. The encoding counters, flags,
encoding spaces and trajectory a vendor client records describe the
interpreter's playout loop, not the Pulseq sequence: the counters of a
non-Cartesian or segmented acquisition, the navigator readouts and the k-space
location of each sample are defined by the sequence. The reconstruction proxy
therefore replaces them with the values the sequence states, before any
reconstruction code reads the stream.

## Enrichment

The header names the revision the series was acquired with
({doc}`sessions`). The proxy reads that revision's sequence chain and tabulates
its readouts in play order ({class}`~pulserver.vre.SequenceTable`). The table
is applied to the stream as follows.

- The header receives one encoding space for the imaging readouts of each
  subsequence, and one for its navigator readouts when it has any. Each space
  carries the matrix size and field of view the sequence defines and encoding
  limits from the counters its readouts reach.
- Each acquisition is matched to a table row by its position in the stream and
  receives the encoding counters, the MRD flags, the dwell time and the
  encoding space reference, and the k-space trajectory when the k-space
  location changes across the readout.

The flags are those the sequence's labels set, such as `IS_NAVIGATION_DATA`,
and the first-and-last flags (`FIRST_IN_SLICE`, `LAST_IN_SLICE` and the
others), derived from the counters. A reconstruction plugin selects the data a
reconstruction runs on by these flags.

## Demodulation to the prescription centre

A Pulseq sequence is designed in logical coordinates about the isocentre. When
the header records a prescription centre $\mathbf{d}$, in mm along the
sequence's gradient axes, in the `pulserver_fov_offset_mm` user parameter,
every channel of each readout is multiplied by

$$
e^{+i 2\pi \mathbf{d} \cdot \mathbf{k}},
$$

where $\mathbf{k}$ is the k-space location of each sample in 1/m, along the
same axes with the sequence's block rotations applied. This moves an
object at $\mathbf{d}$ to the centre of the reconstructed field of view. The
trajectory is not changed.

## Workers

Each series is reconstructed in its own worker process, with the reconstruction
plugin the scanner sequence names, or the one named by the client's
configuration when the sequence names none. A worker reconstructs one series
and exits, which releases the host and GPU memory the reconstruction allocated.
Importing a reconstruction engine takes seconds, so the proxy keeps spare
worker processes that have already imported it.

The number of series reconstructed concurrently is bounded by a number of
slots, derived from the available memory unless it is specified. A series that
arrives when every slot is occupied is written to disk, enriched, as it
arrives, and replayed to a worker once a slot is released. The client remains
connected meanwhile, and the images are returned through its connection.

Images, DICOM datasets and text produced by the worker are relayed to the
client as they are produced. Closing either connection closes the other.

## See also

* {doc}`../user-guide/reconstruction-plugins` — writing a reconstruction.
* {doc}`../api/vre` — the proxy and the enrichment interface.
* {doc}`../api/recon` — the reconstruction plugin interface.
* {doc}`/generated/gallery/03-reconstruction/01_enrichment` — enrichment and reconstruction of a simulated series.
