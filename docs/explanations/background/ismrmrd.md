# ISMRMRD in one page

[ISMRMRD](https://ismrmrd.readthedocs.io) (MRD) is the vendor-neutral format
for raw MR data and the streaming protocol that carries it. Pulseq describes
what the scanner plays; MRD describes what comes back, and how it reaches the
reconstruction. Pulserver's reconstruction side speaks it, so this is the
part that matters here.

## The data model

**A header, then a stream of messages.** The XML header carries everything
constant for the acquisition: the encoded and reconstructed matrices and
field of view, the encoding limits (how far each counter runs and where its
centre is), the trajectory type, sequence parameters (TR, TE, TI, flip angle)
and the patient/system frame.

**Acquisitions** are the readouts. Each carries a fixed-size header plus
complex samples shaped `(coils, samples)`, and optionally the k-space
trajectory for those samples. The header is the interesting part:

- `idx` — the encoding counters: `kspace_encode_step_1` (the in-plane phase
  encode), `kspace_encode_step_2` (partition), `slice`, `contrast`,
  `repetition`, `average`, `set`, `segment`, `phase`. These are exactly the
  Pulseq `LABELSET` counters, which is why Pulserver's design side writes
  them: an acquisition arrives already knowing where it belongs.
- `flags` — a bit field: first/last in slice, in encode step, noise
  measurement, navigator, parallel-imaging calibration, and so on.
- `center_sample`, `discard_pre`, `discard_post` — where the echo sits in the
  readout, and what to trim.
- `position`, `read_dir`, `phase_dir`, `slice_dir` — the geometry, which is
  what turns counters into physical space.

**Waveforms** are the other message type: uniformly sampled, non-image data
with their own id, sampling interval and time stamp — physiological traces
(ECG, respiration), gradient waveforms, or anything a site defines. Pulserver
uses a waveform stream to deliver the sequence description alongside the
data, so a reconstruction that needs the RF and echo layout gets it in band
rather than out of it. See {doc}`../../examples/cpp/gadgetron_client`.

**Images** close the loop: the reconstruction sends images back with their
own header and meta-attributes.

## The session protocol

The streaming form is what a reconstruction service actually sees. A session
is a framed message exchange over a socket:

1. the client connects and sends the **XML header**;
2. it streams **acquisitions** (and waveforms) as they are measured;
3. it sends a **close** message;
4. the service streams **images** back, and closes in turn.

Each message is length-prefixed and typed, so a service can consume a scan it
never sees the end of — reconstructing while the acquisition continues. This
is the shape both Gadgetron and the Python MRD services expect, and the shape
of Pulserver's own reconstruction server: a plugin receives a
`ReconContext`, is handed acquisitions as they arrive, and yields images.

The protocol carries no vendor semantics. Anything a specific system needs —
a calibration flavour, a shot ordering, an FOV shift the reconstruction has
to undo — must be *in* the header, the counters, or a waveform. That
constraint is why the sequence description travels as a waveform rather than
as a side file: the stream is the only thing guaranteed to arrive.

## What Pulserver adds on this side

- **Counters that already mean something.** `auto_label` derives the encoding
  counters from the k-space the sequence actually traverses, so the MRD `idx`
  fields are the sequence's own geometry rather than a convention the
  reconstruction has to be told.
- **A description of the sequence, streamed.** RF definitions with their
  bandwidths and shim vectors, ADC roles, echo positions and the TR duration
  — enough for parameter fitting or subspace reconstruction to know what the
  data are, delivered as MRD waveforms.
- **Recon-side FOV shift.** A shift applied in the logical frame at design
  time is undone by demodulating the acquired samples, which needs the
  trajectory and the echo anchor — both of which Pulserver derives from the
  `.seq` chain rather than from a vendor field.

For the mapping in the other direction — how a Pulseq label becomes an MRD
counter, flag by flag — see {doc}`../sequence_model/modules_and_loops`.
