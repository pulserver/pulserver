# Reconstruction plugins

The interface a reconstruction is written against, and the runtime that drives
it over an MRD stream.

```{eval-rst}
.. currentmodule:: pulserver.recon
```

A plugin file defines a {class}`ReconPlugin` subclass and a module-level
`PLUGIN` instance of it. The same hooks run over a live MRD stream in a worker
of the reconstruction proxy, over an ISMRMRD HDF5 file
({meth}`ReconPlugin.run`) and over an assembled
{class}`~pulserver.mrd.AcquisitionBucket` (calling the instance). The plugin
reconstructs one unit of readouts at a time, as {class}`ReconData`. Writing one
is described in {doc}`../user-guide/reconstruction-plugins`. The image a
{class}`ReconResult` becomes takes its geometry from its unit and its values
from its array, as {doc}`../explanations/reconstruction` states. Coil
sensitivity maps, noise prewhitening and coil compression are the subject of
{doc}`../developer-guide/internals/calibration`.

## Plugin

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.ReconPlugin` | Base class for reconstruction plugins. |
| {obj}`~pulserver.recon.Gadget` | One per-acquisition step, run before a readout is placed. |
| {obj}`~pulserver.recon.load_plugin` | Import a plugin file and return its `PLUGIN` instance. |

## Readout gadgets

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.AsymmetricEcho` | Zero-fill a partial echo to the full echo that is symmetric about its `center_sample`. |
| {obj}`~pulserver.recon.Prewhiten` | Whiten the channels of each readout with the stream's noise measurement. |
| {obj}`~pulserver.recon.RemoveReadoutOversampling` | Crop a full echo to the readout field of view of the reconstruction. |

## Scan context

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.ReconContext` | Header, exam cache and configuration passed to every hook. |
| {obj}`~pulserver.recon.ExamCache` | Thread-safe store of artifacts shared by the series of one exam. |
| {obj}`~pulserver.recon.ExamImage` | A map one series measured, with its geometry, resampled onto a later series' grid. |
| {obj}`~pulserver.recon.B0_MAP` | Where a calibration scan leaves the off-resonance map, in Hz, for the series that follow. |
| {obj}`~pulserver.recon.B1_MAP` | Where a calibration scan leaves the transmit field map, as a fraction of what was asked for. |
| {obj}`~pulserver.recon.COIL_SENSITIVITIES` | Where a calibration scan leaves the receive coil sensitivities, a {class}`~pulserver.recon.CoilSensitivities`. |
| {obj}`~pulserver.recon.NOISE_COVARIANCE` | Where a noise scan leaves the prewhitening of the exam, for the series that follow. |
| {obj}`~pulserver.recon.EXAM_ARTIFACTS` | The maps a context carries as attributes of itself, each its own key in the exam cache. |

## Calibration

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.coil_maps` | Return the coil sensitivities of a unit from its calibration k-space, the stream or the exam. |
| {obj}`~pulserver.recon.CoilSensitivities` | Receive coil sensitivity maps with the coils, prewhitening, compression and geometry they were estimated in. |
| {obj}`~pulserver.recon.MissingCalibration` | No source holds coil sensitivities the unit can use. |
| {obj}`~pulserver.recon.CoilCompression` | Reduce the receive channels of a unit's k-space to virtual channels. |

## Image correction

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.gradient_unwarped` | Return an image resampled from where its voxels were acquired to where they belong. |

## Buffers and results

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.ReconData` | The readouts of one reconstruction unit: imaging and calibration k-space, counters and waveforms. |
| {obj}`~pulserver.recon.ReconBuffer` | K-space of one encoding space, filled one acquisition at a time, with the acquisition nearest the k-space centre. |
| {obj}`~pulserver.recon.ReconResult` | Image array the runtime packages as an MRD image or DICOM dataset, with the acquisition its geometry is taken from and the rescale of its DICOM pixels. |
