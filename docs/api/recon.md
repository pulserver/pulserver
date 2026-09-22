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
{class}`~pulserver.mrd.AcquisitionBucket` (calling the instance). Writing one is
described in {doc}`../user-guide/reconstruction-plugins`.

## Plugin

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.ReconPlugin` | Base class for reconstruction plugins. |
| {obj}`~pulserver.recon.Gadget` | One per-acquisition step of a plugin's chain. |
| {obj}`~pulserver.recon.load_plugin` | Import a plugin file and return its `PLUGIN` instance. |

## Scan context

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.ReconContext` | Header, exam cache and configuration passed to every hook. |
| {obj}`~pulserver.recon.ExamCache` | Thread-safe store of artifacts shared by the series of one exam. |

## Buffers and results

| Object | Description |
| --- | --- |
| {obj}`~pulserver.recon.ReconData` | Every encoding space of a scan, by space index. |
| {obj}`~pulserver.recon.ReconBuffer` | K-space of one encoding space, filled one acquisition at a time. |
| {obj}`~pulserver.recon.ReconResult` | Image array the runtime packages as an MRD image or DICOM dataset. |
