# Raw-data enrichment and routing

The scanner's reconstruction client streams each series as MRD: a configuration
text, the XML header and the acquisitions. The encoding counters, flags,
encoding spaces and trajectory a vendor client records describe the
interpreter's playout loop, not the Pulseq sequence: the counters of a
non-Cartesian or segmented acquisition, the navigator readouts and the k-space
location of each sample are defined by the sequence. The reconstruction proxy
therefore replaces them with the values the sequence states, before any
reconstruction code reads the stream.

The readouts arrive demodulated to the prescribed field-of-view centre, which
is applied to the sequence when its IR is built ({doc}`ir-cache`). The proxy
leaves the samples as received, and the trajectory it attaches is that of the
sequence as designed, in the logical frame.

## Enrichment

The header names the design the series was acquired with
({doc}`designs`). The proxy reads that design's sequence chain and tabulates
its readouts in play order ({class}`~pulserver.proxy.SequenceTable`). The table
is applied to the stream as follows.

- The header receives one encoding space for the imaging readouts of each
  subsequence, and one for its navigator readouts when it has any. Each space
  carries the matrix size and field of view the sequence defines and encoding
  limits from the counters its readouts reach.
- The header's sequence parameters are the TR, TE, TI and flip angles the
  sequence defines. The TR, TE and flip angles it does not define are
  measured by pypulseqpp's `Sequence.test_report_dict`: TE from the excitation
  before the closest approach to the k-space centre, TR between the
  excitations around it, and every distinct flip angle the sequence plays.
- Each acquisition is matched to a table row by its position in the stream and
  receives the encoding counters, the MRD flags, the dwell time and the
  encoding space reference, and the k-space trajectory when the k-space
  location changes across the readout. When the client numbers its
  acquisitions, each `scan_counter` must follow the previous one by one; a gap
  or a repeat stops the series before it is reconstructed, since every later
  row would be shifted.

The flags are those the sequence's labels set, such as `IS_NAVIGATION_DATA`,
and the first-and-last flags (`FIRST_IN_SLICE`, `LAST_IN_SLICE` and the
others), derived from the counters. A reconstruction plugin selects the data a
reconstruction runs on by these flags.

## Trajectory

The k-space location of each sample is the integral of the sequence's
gradients, with block rotations applied, in 1/m. The table does not hold it for
the whole scan: it is integrated a run of consecutive readouts at a time, by
pypulseqpp's `Sequence.adc_kspace`, from the last excitation before the run.
An excitation resets k to zero at the pulse centre, so the samples that follow
it do not depend on the gradients played before it, and a run integrated from
there gives the k-space locations of the scan integrated from its first block.
A refocusing pulse reverses k rather than resetting it and does not begin an
integration, nor does an excitation in a block that also holds a readout.

An acquisition carries as its trajectory every axis its encoding space varies
along, up to the last, whether or not its own k moves along it: the line of a
PROPELLER blade that is played unrotated keeps its phase encoding in ky. The
partitions of a stack of spokes or spirals are Cartesian along z and placed by
their `kspace_encode_step_2` counter, so their kz is not part of the
trajectory. A reconstruction takes the trajectory in grid units, k times the
reconstructed field of view, through
{meth}`~pulserver.recon.ReconBuffer.grid_trajectory`.

Tabulating a design integrates every range once, for the echo sample of each
readout and the k-space axes its trajectory spans, which decide the header's
trajectory type. An acquisition's trajectory is integrated again when it is
enriched, from the range holding it, and the table keeps the ranges it
integrated last. The k-space locations the proxy holds therefore do not grow
with the length of the scan.

## Workers

Each series is reconstructed in its own worker process, with the reconstruction
plugin the scanner sequence names, or the one named by the client's
configuration when the sequence names none. A worker reconstructs one series
and exits, which releases the host and GPU memory the reconstruction allocated.
Importing a reconstruction engine takes seconds, so the proxy keeps spare
worker processes that have already imported it.

The series of one exam share a directory on the reconstruction computer. What
a series stores in its exam cache is written there, and a later series of the
exam reads it back, such as a coil calibration it need not compute again. The
exam is the one the header names; the directory is deleted once a header names
another exam and no series of the first is still reconstructed.

The number of series reconstructed concurrently is bounded by a number of
slots, derived from the available memory unless it is specified. On a host with
GPUs, which the proxy finds from `CUDA_VISIBLE_DEVICES` or `nvidia-smi` without
importing a GPU library, each slot holds one of them, one series per GPU unless
more are allowed, and the reconstruction finds its GPU in `context.device`. A
worker is an ordinary process, so a reconstruction may start processes of its
own. A series that
arrives when every slot is occupied is written to disk, enriched, as it
arrives, and replayed to a worker once a slot is released. The client remains
connected meanwhile, and the images are returned through its connection.

Images, DICOM datasets and text produced by the worker are relayed to the
client as they are produced. Closing either connection closes the other. Once
the client's stream ends, the proxy waits for the worker to close however long
the reconstruction takes, unless it was started with a reconstruction timeout,
past which the worker is terminated and the client told so.

## Reconstruction server

A proxy given a server to forward to reconstructs no series itself. It
enriches each series as it arrives and sends it on over TCP to that MRD server:
a reconstruction server ({class}`~pulserver.proxy.ReconServer`) on another
computer, or any server that reads the MRD streaming protocol. The server
receives a config file message naming the reconstruction plugin of the series,
or a name the proxy is configured with, then the enriched header and
acquisitions. What it returns is relayed to the client as a worker's output
is, and a reconstruction timeout closes the connection to it. With DICOM
conversion enabled, the proxy converts each image the server returns to DICOM
from the enriched header before relaying it.

A reconstruction server runs each series it receives with the plugin its
config names, in workers, slots and a queue of its own, and enriches nothing:
the series a proxy forwards arrive enriched.

An MRD message carries no length by which a reader can skip a message type it
does not know. A message the proxy has no reader for therefore ends what it
relays, and the client receives a text naming the message's type.

## See also

* {doc}`../user-guide/reconstruction-plugins` — writing a reconstruction.
* {doc}`../user-guide/reconstruction-client` — the MRD stream a reconstruction client sends.
* {doc}`../api/proxy` — the proxy, the reconstruction server and the enrichment interface.
* {doc}`../api/recon` — the reconstruction plugin interface.
* {doc}`/generated/gallery/03-reconstruction/01_enrichment` — enrichment and reconstruction of a simulated series.
