# The MRD path

```{admonition} TL;DR
:class: tip

- Two halves connected by nothing but the MRD session protocol: a C++ **client**
  that turns the scan into a vendor-neutral stream, and a Python **server** that
  reconstructs it. Either is replaceable.
- The client **converts** packets as they arrive, **enriches** them from the
  Pulseq file the scanner played — header, counters, flags, trajectory, sequence
  description — and **finishes** any FOV shift the design side deferred.
- The trajectory is **composed, never re-integrated**: the file already holds
  the readout's normalised base, and the client scales, offsets and turns it.
- The server's unit of context is the **exam**, not the scan: a keyed cache whose
  generation retires when the exam identity changes, with outstanding leases kept
  alive.
- Concurrency is derived from **available RAM**, and overflow is **queued to
  disk and replayed**, never dropped and never stalled.
- Real-time feedback gets its own port and a strictly synchronous one-out
  one-back exchange with a tagged payload.
```

{doc}`MRD <../background/ismrmrd>` defines what a reconstruction service
receives. On a real scanner both ends of that definition are missing: the vendor
acquisition system emits native raw-data packets rather than MRD, and the vendor
reconstruction pipeline is not a place a Python reconstruction can run.

Each half is replaceable. The client speaks to any MRD server — Gadgetron, the
Python MRD services, or Pulserver's own — and the server accepts any MRD client,
reconstructing a recorded file just as it reconstructs a live scan.

```{figure} ../assets/mrd_path/mrd_path.png
The path from the spectrometer to the images: the scanner's native packets
and the sequence enter the C++ client, which converts, enriches and
demodulates them; the MRD session protocol carries the stream to the Python
server, whose reconstruction slots share an exam cache and whose overflow is
drained to disk and replayed.
```

## The client: producing the stream

The client runs beside the vendor's own reconstruction pipeline and does
three jobs, in order.

### Conversion

As acquisitions arrive from the spectrometer, the client converts each native
packet to an MRD acquisition and sends it immediately — streaming while the scan
runs, not exporting after it ends.

The vendor-specific half is confined to one component. On GE the client is built
on the Orchestra SDK and uses GE's own `ge_to_ismrmrd` converter, behind an
interface, so a different converter slots in without touching the enrichment or
the session logic.

### Enrichment from the Pulseq file

A converted packet is samples with a vendor header — correct, but ignorant of
its k-space trajectory, where its echo sits, and which encoding space it belongs
to. All of that is knowledge the *sequence* has, so the client reads the same
`.seq` chain the scanner played, parsed in C++ (`read_sequence_files()` walks
the `NextSequence` chain, text or binary). From it it fills in:

- **The MRD header**: the encoding spaces with their matrices, fields of
  view and encoding limits, derived per subsequence — a calibration prescan
  and its imaging scan each get their own encoding space.
- **Per-acquisition counters and flags**: the design side writes its loop
  counters into the file as labels, so each acquisition is stamped with the
  `idx` counters and role flags MRD expects, plus its echo position
  (`center_sample`) and dwell time. The first/last boundary flags are derived
  here, from the counters and arrival order.
- **The k-space trajectory**, attached per readout wherever k does not advance
  at a constant rate across the ADC window — the test is per axis, so a
  Cartesian readout that samples its ramps gets a one-dimensional trajectory
  along the readout while its flat axes stay in the counters, and a readout
  played entirely on a plateau gets none because there is nothing a counter
  cannot say. It is *composed*, not integrated: the design side already stored
  the readout's normalised base in the file, and the client scales it by the
  instance amplitude, adds the block's k origin and turns it by the block's
  rotation — see {doc}`../performance/transform_fov`. The reconstruction never
  re-derives it from vendor parameters, and the client never re-integrates a
  gradient.
- **The sequence description**, streamed in band as MRD waveforms: the RF
  definitions with their shapes and shim vectors, the event timeline of one
  TR, and the echo layout — enough for parameter fitting or subspace
  reconstruction to know what the data are. It rides the standard waveform
  message, so a server that does not need it simply ignores it, and
  physiological traces travel the same channel beside it.

### Finishing a deferred FOV shift

An off-isocentre prescription is a phase, and most of it is already in the
file: a readout whose gradient is flat across its ADC window, on a block that
is not rotated, had its shift baked at design time as a frequency and a phase
offset, and nothing downstream has to know it happened.

The rest cannot be baked. A rotated block plays its gradients at a physical
angle, and a gradient that moves across the window has a phase no frequency
describes — so those readouts are written carrying a base trajectory and
marked `defers_fov_shift`, and the client finishes them. After the trajectory
has been attached, and only for a marked acquisition, each sample is multiplied
by the rotor for $\Delta r \cdot k$ at that sample — with $k$ **absolute**,
carrying the block's k origin, so the readouts stay referenced to one point and
not each to its own entry. Which readouts those are, why the file stores a
trajectory rather than the phase itself, and where the origin comes from, is
{doc}`../performance/transform_fov`.

Both the shift and the trajectory live in the *logical* frame — readout,
phase, slice — and the product $\Delta r \cdot k$ does not change when both
are rotated together, so neither the client nor the reconstruction ever needs
to know the prescribed orientation, a per-block rotation, or a motion
correction's updates. The sign convention is the design side's: on a readout
that *can* be baked, the two routes are held to produce the same phase, and
that phase is held equal to $\Delta r \cdot k$ sample by sample.

What leaves the client is a stream with no vendor semantics left in it. See
{doc}`../../api/cpp/recon` for its surface.

## The server: consuming the stream

Pulserver's own server is an MRD server like any other: it listens on a TCP
port, accepts a connection, reads the configuration message and the header,
and reconstructs what follows. The configuration message names the
reconstruction — a module name, or the absolute path of the recon script
staged with the sequence, which is what the scanner sends, so a scan and its
reconstruction are downloaded together and no server-side registration step
exists. A stream naming nothing recognisable falls back to a handler that
records the data and returns.

Each connection reconstructs through its own copy of the plugin
(`plugin.spawn()`), so the configured `PLUGIN` stays a template: an expensive
thing it holds — a loaded network, a compiled operator — is shared across
concurrent scans, while everything the lifecycle hooks assign stays private to
one stream. Acquisitions are handed to the plugin as they arrive; waveforms are
collected and given to every reconstruction of the scan, because they describe
the measurement rather than the image. What the plugin returns goes back up the
same socket as MRD images, or as DICOM if it asked for that.

Three things beyond that are the server's own, each because a scanner is not a
workstation.

### The exam is the unit of context

Sensitivity maps, a subspace basis, a gridding plan are expensive, and within
one exam they are often the *same* scan after scan. So each reconstruction
receives an **exam cache** — a keyed store whose `get_or_create` runs its
factory at most once per key, with an optional cleanup for values holding GPU
memory.

The exam identity comes from the header: an `ExamID` user parameter if the
exporter writes one, otherwise the standard study identity. The measurement
identity is deliberately *not* used, because it changes between the sequences
of a single exam, which is exactly the boundary the cache is meant to cross.
A header with no exam identity at all gets a private generation, so nothing
is ever shared by accident.

Observing a header with a different exam identity retires the current
generation immediately — the next patient never inherits the previous
patient's calibration. Retiring is not freeing: a reconstruction that already leased the outgoing
generation keeps it alive until it finishes, and the artifacts are released on
the last lease returned — a scanner can begin the next exam while the previous
scan is still reconstructing.

Keys should carry what an artifact actually depends on — geometry, coil
configuration, trajectory or basis identity. Sharing an exam does not by
itself make two sensitivity maps interchangeable.

### Concurrency is derived from available RAM

An iterative or deep-learning reconstruction is measured in tens of gigabytes,
and two of them on a box that fits one do not run half as fast — they swap, or
the kernel kills one. So the server admits a bounded number of simultaneous
reconstructions, derived from the machine:

$$
\text{slots} = \max\left(1,\ \left\lfloor \frac{\text{available RAM} \times 0.8}{\text{RAM per reconstruction}} \right\rfloor\right)
$$

with 48 GiB per reconstruction as the default estimate. On a 156 GiB machine
with ~142 GiB available, that is two slots and a 96 GiB working set. Both the
estimate and the resulting limit are overridable by flag or environment
variable.

### Overflow: queued, never dropped

Refusing the connection is not an option — the scan has already happened, and
the client has nowhere to put the data. Stalling the client is not one
either: the acquisition system needs its buffers back.

So when every slot is busy, the server still consumes the stream in full,
writing it to an MRD file as it goes, and drops a small JSON file beside it
naming the reconstruction that was requested. The client sees an ordinary,
complete session. A background worker then picks up the oldest of those,
waits for a slot, and runs the requested reconstruction against the file. Its
name changes as it goes — queued, processing, then processed or failed —
which is both the record of what happened and what stops two workers taking
the same session; one still marked `processing` at startup belonged to a
process that died, and is retried.

The replayed scan runs through the same plugin lifecycle, exam lease and output
handling as a live one; only the source of the acquisitions and the destination
of the images differ, and both sit behind the same connection interface. There
is no second reconstruction path to keep in step with the first — which is why
reconstructing a recorded file needs no server at all:

```python
from pulserver.app import cartesian2D_recon

images = cartesian2D_recon("scan.h5")
```

opens the file, hands the header to the plugin as its context, and drives the
ordinary hooks over the acquisitions in acquisition order — in process, with
no socket and no port.

## The real-time round trip

Reconstruction consumes a scan; a real-time process *changes* it. That needs a
different contract — not throughput but a bounded round trip — so it gets its
own port, with the same MRD framing and a strictly synchronous exchange: one
acquisition out as it is measured, exactly one result back for it, before the
interpreter reaches the blocks the result bears on.

```{figure} ../assets/mrd_path/mrd_feedback.png
The real-time round trip: the interpreter sends one acquisition as it is
measured and receives one tagged result before the blocks it bears on.
```

The channel is uncommitted about what a result *means*: it carries a tagged
payload, and the tag is what the interpreter dispatches on. Which tags it
recognises, and what each does to the blocks ahead of the cursor — a geometry
update, a rejected TR replayed, a parameter moved — is decided on the
interpreter side and nowhere else. Feedback of a new kind is a new tag; the
transport, which only has to be fast and deterministic, does not change.
