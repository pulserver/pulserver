# Virtual scanner

The stand-ins for the scanner: the cache played, an analytic phantom acquired
along the trajectory it plays or the blocks it plays simulated on isochromats,
and the raw data sent as the scanner's reconstruction client sends them; and
the blocks the cache plays, written as a Pulseq file for a simulator that reads
one.

```{eval-rst}
.. currentmodule:: pulserver.virtual
```

The design calls, the IR, the reconstruction proxy and the reconstruction
plugins are the production code; what each stand-in exercises, and the signal
model, are described in {doc}`../explanations/virtual-scanner`. Positions are
in metres and k-space locations in 1/m, along the physical axes: the logical
axes turned by the prescription's rotation, the identity unless one is given.
Chemical shifts are in ppm from water, and frequencies in Hz from the
scanner's centre frequency.

## Acquisition

| Object | Description |
| --- | --- |
| {obj}`~pulserver.virtual.trajectory` | The k-space location of every ADC sample the cache beside a sequence file plays. |
| {obj}`~pulserver.virtual.acquire` | The samples the cache beside a sequence file acquires of a phantom, demodulated as the playout demodulates. |
| {obj}`~pulserver.virtual.simulate` | The samples the cache beside a sequence file acquires of isochromats, the blocks it plays played on them in turn. |
| {obj}`~pulserver.virtual.excited` | The slabs the excitation pulses of the cache beside a sequence file excite, found by playing each on a line of isochromats along its gradient; none where one excites the whole phantom. |
| {obj}`~pulserver.virtual.Slabs` | Slabs of the physical frame, each the isochromats that see a field within its bounds during a pulse, so that an isochromat's own frequency moves its slab; the region a phantom is sampled in. |

## Bloch simulation

The Bloch equation with relaxation, integrated on isochromats in the frame
rotating at the reference frequency, as {doc}`../explanations/bloch-simulation`
states it. Times are in seconds, gradients in Hz/m and RF fields in Hz.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.virtual.Isochromats` | Isochromats, and the magnetisation the Bloch equation carries from one block to the next; one block's events played on them. |
| {obj}`~pulserver.virtual.Repetitions` | Repetitions of a sequence of blocks that differ in their phase offsets, phase encodings and readout directions, played in turn from the affine map one repetition applies to each isochromat. |

## Scan clock and sound

| Object | Description |
| --- | --- |
| {obj}`~pulserver.virtual.Scan` | The cache beside a sequence file played on isochromats as {func}`~pulserver.virtual.simulate` plays it, against a scan clock, in spans carrying their readouts and the sound of their gradients. |
| {obj}`~pulserver.virtual.Chunk` | A span of a scan between two block boundaries: its bounds in scan time, its readouts and its sound. |
| {obj}`~pulserver.virtual.SAMPLE_RATE` | MATLAB Pulseq's audio sample rate, which a scan's sound takes unless given another. |

## External simulators

| Object | Description |
| --- | --- |
| {obj}`~pulserver.virtual.export` | Write the blocks the cache beside a sequence file plays as one Pulseq 1.5.1 file, and return the receiver phase of every readout. |

## Phantom

| Object | Description |
| --- | --- |
| {obj}`~pulserver.virtual.Phantom` | Ellipses whose signals add, received by one coil or several of analytic sensitivity, placed in the physical frame by a rotation and a position; sampled as isochromats for the Bloch simulation. |
| {obj}`~pulserver.virtual.BrainWeb` | BrainWeb's normal brain, downloaded on first use, placed head first and supine with its tissues' T1, T2 and proton density, in the field its susceptibility adds to B0; sampled as isochromats for the Bloch simulation. |
| {obj}`~pulserver.virtual.Ellipse` | An ellipse of uniform magnetization in a plane of constant z, of one chemical shift and one pair of relaxation times. |
| {obj}`~pulserver.virtual.localizer` | The axial, coronal and sagittal images of a phantom's proton density through a point, as DICOM, drawn from its ground truth with no sequence played. |

## Coils

| Object | Description |
| --- | --- |
| {obj}`~pulserver.virtual.Coil` | The RF coils of an exam, fixed in the physical frame: the transmit and receive sensitivities of BART's coil models, sampled by bartorch, or of each coil's field maps, the default shim a single-channel pulse plays through, and the VOP limits of mapped transmit coils. |
| {obj}`~pulserver.virtual.COILS` | The virtual scanner's coils by name, `transmit/receive`: the body coil both ways, the body coil with a 48-channel receive head array, and an 8-channel parallel-transmit head coil with a 32-channel receive head array. |

## Reconstruction client

| Object | Description |
| --- | --- |
| {obj}`~pulserver.virtual.send` | Send one series to a reconstruction proxy as the scanner's client sends it, each readout as it is acquired; return what comes back. |
| {obj}`~pulserver.virtual.record` | Write one series to an ISMRMRD file as `send` sends it. |
| {obj}`~pulserver.virtual.Console` | A scanner console's calls answered in process: the design calls on the interpreter's text blocks, an exam's coil and localizer, and a scan of a stored design whose clock and DICOM images stream back, reconstructed by a proxy or in process; `pulserver console` serves them over a WebSocket. |
