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
| {obj}`~pulserver.virtual.simulate` | The samples the cache beside a sequence file acquires of isochromats, in pypulseqpp's Bloch simulation of the blocks it plays. |

## Scan clock and sound

| Object | Description |
| --- | --- |
| {obj}`~pulserver.virtual.Scan` | The cache beside a sequence file played on isochromats block by block against a scan clock, in spans carrying their readouts and the sound of their gradients. |
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
| {obj}`~pulserver.virtual.Ellipse` | An ellipse of uniform magnetization in a plane of constant z, of one chemical shift and one pair of relaxation times. |

## Reconstruction client

| Object | Description |
| --- | --- |
| {obj}`~pulserver.virtual.send` | Send one series to a reconstruction proxy as the scanner's client sends it, each readout as it is acquired; return what comes back. |
