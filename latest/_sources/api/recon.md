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
is described in {doc}`../user-guide/reconstruction-plugins`.

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
| {obj}`~pulserver.recon.RemoveReadoutOversampling` | Crop a full echo to the readout field of view of the reconstruction. |

## Scan context

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.ReconContext` | Header, exam cache and configuration passed to every hook. |
| {obj}`~pulserver.recon.ExamCache` | Thread-safe store of artifacts shared by the series of one exam. |
| {obj}`~pulserver.recon.B0_MAP` | Where a calibration scan leaves the off-resonance map, in Hz, for the series that follow. |
| {obj}`~pulserver.recon.B1_MAP` | Where a calibration scan leaves the transmit field map, as a fraction of what was asked for. |
| {obj}`~pulserver.recon.COIL_SENSITIVITIES` | Where a calibration scan leaves each receive coil's sensitivity. |
| {obj}`~pulserver.recon.EXAM_ARTIFACTS` | The maps a context carries as attributes of itself, each its own key in the exam cache. |

## Buffers and results

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.ReconData` | The readouts of one reconstruction unit: imaging and calibration k-space, counters and waveforms. |
| {obj}`~pulserver.recon.ReconBuffer` | K-space of one encoding space, filled one acquisition at a time. |
| {obj}`~pulserver.recon.ReconResult` | Image array the runtime packages as an MRD image or DICOM dataset. |
