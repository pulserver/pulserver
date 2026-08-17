# The MRD client

{doc}`MRD <../background/ismrmrd>` defines what a reconstruction service
receives. Something still has to *produce* that stream on a real scanner:
vendor acquisition systems emit their own native raw-data packets, and a
Pulseq sequence player knows things about the data — trajectory, echo
position, what each readout is for — that no vendor field carries. Pulserver
closes this gap with a C++ client that runs beside the vendor's own
reconstruction pipeline and speaks the {doc}`MRD session protocol
<../background/ismrmrd>` to any MRD server: Gadgetron, the Python MRD
services, or {doc}`Pulserver's own reconstruction server
<../../examples/python/reconstruction>`.

Three jobs, in order.

## Convert on the fly

As acquisitions arrive from the spectrometer, the client converts each native
packet to an MRD acquisition and sends it immediately — streaming while the
scan runs, not exporting after it ends, which is what makes real-time and
long-scan reconstructions possible at all.

The vendor-specific half of this is deliberately confined to one component.
On GE the client is built on the Orchestra SDK and uses GE's own
`ge_to_ismrmrd` converter for the packet translation; the converter sits
behind an interface, so a different vendor's converter — or a different
GE-side one — slots in without touching the enrichment or the session logic
below.

## Enrich from the Pulseq file

A converted packet is samples with a vendor header — correct, but ignorant:
it does not know its k-space trajectory, where its echo sits, or which
encoding space it belongs to. Everything it is missing is knowledge the
*sequence* has, so the client reads the sequence: the same `.seq` chain the
scanner played, parsed in C++ (`read_sequence_files()` walks the
`NextSequence` chain, text or binary). From it, the client fills in:

- **The MRD header**: the encoding spaces with their matrices, fields of
  view and encoding limits, derived per subsequence — a calibration prescan
  and its imaging scan each get their own encoding space.
- **Per-acquisition counters and flags**: the design side writes its loop
  counters into the file as labels, so each acquisition is stamped with the
  `idx` counters and role flags MRD expects, plus its echo position
  (`center_sample`) and dwell time. The first/last boundary flags are derived
  here, from the counters and arrival order.
- **The k-space trajectory**, computed from the gradient waveforms and
  attached per readout for non-Cartesian encoding — the reconstruction never
  re-derives it from vendor parameters.
- **The sequence description**, streamed in band as MRD waveforms: the RF
  definitions with their shapes and shim vectors, the event timeline of one
  TR, and the echo layout — enough for parameter fitting or subspace
  reconstruction to know what the data are. It rides the standard waveform
  message, so a server that does not need it simply ignores it, and
  physiological traces travel the same channel beside it.

## Undo the FOV shift

An off-isocentre prescription is applied at design time as a phase — the
sequence is played as if at isocentre, and every sample carries
$e^{-i\,2\pi\,\Delta r \cdot k}$ for the prescribed offset $\Delta r$. The
client undoes it: after the trajectory is attached, each acquisition is
demodulated by the matching phase, $\Delta r \cdot k$ evaluated per sample.

Both the shift and the trajectory live in the *logical* frame — readout,
phase, slice — and the product $\Delta r \cdot k$ does not change when both
are rotated together, so neither the client nor the reconstruction ever needs
to know the prescribed orientation, a per-block rotation, or a motion
correction's updates. The sign convention matches the design side exactly:
a sequence shifted at design time and one shifted here reconstruct
identically.

The result is that an MRD server downstream sees a stream with no vendor
semantics left in it — the constraint the {doc}`session protocol
<../background/ismrmrd>` imposes, met by construction. A reconstruction
written against this stream runs unchanged against a file recorded from it,
or against another scanner producing the same stream. See
{doc}`../../examples/cpp/gadgetron_client` for the client in use.
