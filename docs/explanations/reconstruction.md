# Reconstruction units and routing

```{admonition} TL;DR
:class: tldr

- A reconstruction unit is the readouts of one branch and encoding space that
  share their image-selecting counters; it closes on the flag its branch
  declares.
- Its images carry the values the plugin returned, with no scaling.
- A proxy can forward each enriched series to a reconstruction server instead
  of reconstructing it.
```

The proxy completes each series from its design ({doc}`raw-data`) and then
groups its readouts into the units a reconstruction plugin receives.

## Reconstruction units

A reconstruction takes the readouts of one image, or of several images
reconstructed together, such as the echoes of a slice. The stream states where
those readouts end only through the encoding counters and flags that
enrichment supplies. A *reconstruction unit* is the set of readouts of one
branch and one encoding space that share their slice, contrast, cardiac phase,
repetition, set and average counters; a branch is a kind of readout a plugin
reconstructs separately, such as `imaging` or `navigator`. The `segment` and
user counters name positions within a readout train and do not separate units.
A plugin names in its `axes` the counters it takes as axes of the unit's
k-space instead, such as the echoes of a multi-echo acquisition, and these do
not separate units either.

A unit allocates its imaging k-space when its first imaging readout arrives,
from the encoding space the header gives it, and places each readout as
described under {ref}`reconstruction-placement`. The flags of a readout select
its buffer as Gadgetron's acquisition bucket divides readouts: parallel-imaging
calibration readouts are placed in `ref`, readouts flagged as calibration and
imaging in `ref` and `data`, phase-correction readouts in neither, and any
other readout in `data`. The calibration k-space is allocated when the unit
closes, over the lines its readouts cover.

A unit closes when the flag its branch declares has arrived at every position
along its axes. Enrichment sets the last-in flag of a counter within each
combination of the encoding space and the other image-selecting counters, so
`LAST_IN_SLICE` arrives once per echo or average of a slice, and a plugin with
those counters as axes waits for all of them. Readouts of different units may
be interleaved: slices acquired line by line each close at their own flagged
readout.

A closed unit leaves the plugin before its reconstruction runs, and nothing in
the runtime holds it afterwards, so the memory a series holds is that of the
units open at once, not that of the series. A unit still open at the last
readout of the measurement or at the end of the stream is reconstructed then,
under its own branch, in the order the units opened.

## Images

A {class}`~pulserver.recon.ReconResult` becomes an MRD image whose header the
runtime builds from the unit that produced it. The values of the image are those
of the array the plugin returned.

### Header

The field of view is that of the reconstruction space of the unit's encoding
space, `(x, y, z)` in mm, so the images of a unit in the second encoding space
carry the second space's field of view. The matrix is the shape of the returned
array, which is {attr}`~pulserver.recon.ReconBuffer.image_shape`, the
reconstruction matrix of the unit's space, for a plugin that uses it.

The position, the orientation (`read_dir`, `phase_dir`, `slice_dir`), the
patient table position, the physiology time stamps and the user fields are those
of the unit's *reference acquisition*, the imaging acquisition placed nearest the
k-space centre, as Gadgetron selects it for an image header. The centre lies at
`extent // 2` along the phase-encoding axis and, where the encoded space has more
than one partition, the partition axis, as described under
{ref}`reconstruction-placement`; nearness is the Euclidean distance from it in
grid positions. Of acquisitions at the same distance, the first placed is the
reference. The acquisition time stamp of the image is the earliest of the unit's
imaging acquisitions.

A plugin reads the reference acquisition as
{attr}`~pulserver.recon.ReconBuffer.reference`, and takes the geometry from
another acquisition by passing it as `ReconResult.reference`.

### Values

The MRD image holds the values the plugin returned: `float32` for real
floating-point data, `complex64` for complex data, wider types being converted
to these, and integer data in its own type. No scale or normalisation is
applied, so the ratios between the images of a series, such as the decay across
the echoes of a multi-echo acquisition or the ratios between frames, are those
of the reconstruction. The units are the reconstruction's own.

A `(z, y, x)` result becomes one image per partition. Each image states the
smallest and largest finite value of the result, of its magnitude where it is
complex, in the meta attributes `ArrayMinimum` and `ArrayMaximum`.

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

* {doc}`raw-data` — what the proxy adds to the stream before a unit is formed.
* {doc}`../user-guide/reconstruction-plugins` — writing a reconstruction.
* {doc}`../user-guide/reconstruction-client` — the MRD stream a reconstruction client sends.
* {doc}`calibration` — coil sensitivity maps, prewhitening and coil compression.
* {doc}`../api/proxy` — the proxy, the reconstruction server and the enrichment interface.
* {doc}`../api/recon` — the reconstruction plugin interface.
* {doc}`/generated/gallery/02-tours/03_fov_offset_enrichment` — enrichment and reconstruction of a simulated series.
