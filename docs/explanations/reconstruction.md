# Reconstruction plugins and their context

```{admonition} TL;DR
:class: tldr

- Your plugin receives k-space already sorted into units, one per image or group of images, closed by MRD flags.
- Its images carry the values it returned, with the geometry of the acquisition nearest the k-space centre.
- Calibration is shared across the series of an exam, and reused only where every recorded field matches.
```

A recon plugin is written as in Gadgetron: gadgets act on each readout as it
arrives, and a reconstruction runs when the flag that ends its data arrives.
What pulserver adds is the sorting, so that `recon` is handed k-space rather
than a stream, and a context that carries calibration from one series of an
exam to the next. Lessons 4 and 5 of the {doc}`Course <../examples/course>`
write such a plugin.

## What pulserver does

:::{container} capabilities

- **Runs your gadgets on each readout before it is buffered.**

  Code: {class}`~pulserver.recon.Gadget`. Tests: *gadgets run before a readout is buffered* (`test_units.py`); *the completed echo is symmetric about the sample at its echo*; *a scan of full echoes is cropped to the reconstruction matrix* (`test_gadgets.py`).
- **Groups readouts into units by branch, encoding space and image counters, except those you name as axes.**

  Code: {class}`~pulserver.recon.ReconPlugin`. Tests: *an image counter separates units unless it is an axis of the unit*; *interleaved slices each close at their own flag with their own readouts* (`test_units.py`).
- **Closes a unit when its branch's flag has arrived at every position along its axes.**

  Code: `triggers` of {class}`~pulserver.recon.ReconPlugin`. Test: *a unit waits for its flag at every position along its axes* (`test_units.py`).
- **Places each readout in k-space by its counters, and calibration readouts in `ref`.**

  Code: {class}`~pulserver.recon.ReconBuffer`. Tests: *each acquisition lands where its own counters say*; *a partial fourier scan keeps its lines where the centre puts them* (`test_buffers.py`); *a unit places each readout in data and reference by its flags* (`test_units.py`).
- **Holds only the units open at once, not the series.**

  Code: {class}`~pulserver.recon.ReconData`. Test: *the bytes held stay those of one frame over a hundred frames* (`test_units.py`).
- **Writes each image header from the acquisition nearest the k-space centre, and keeps the values unscaled.**

  Code: {class}`~pulserver.recon.ReconResult`. Tests: *the position and physiology are those of the acquisition nearest the k space centre*; *the ratios between the images of a series survive to the mrd images and the dicom pixels* (`test_images.py`).
- **Prewhitens on arrival, from the series' noise readouts or the exam's noise covariance.**

  Code: {class}`~pulserver.recon.Prewhiten`. Test: *noise readouts whiten the imaging readouts* (`test_prewhiten.py`).
- **Compresses coils with one basis per stream.**

  Code: {class}`~pulserver.recon.CoilCompression`. Tests: *virtual channels are uncorrelated and carry the leading eigenvalues*; *compressed data and maps reconstruct the uncompressed image* (`test_coil_compression.py`).
- **Finds coil maps in the unit, the stream or the exam, and refuses maps that differ in any recorded field.**

  Code: {func}`~pulserver.recon.coil_maps`, {class}`~pulserver.recon.CoilSensitivities`. Tests: *a later unit without calibration readouts is given the maps of its slice*; *maps that differ in a field are refused naming it with both values*; *exam maps at another matrix are refused not resampled* (`test_calibration.py`).
- **Shares an exam cache across the series of one exam.**

  Code: {class}`~pulserver.recon.ExamCache`. Test: *a value one cache stores is read by the next on its directory* (`test_app.py`).
- **Ships a `pics` plugin with prewhitening, partial Fourier and asymmetric echo.**

  Code: `src/pulserver/_zoo/recon/pics.py`. Tests: *every other phase encode and the centre reconstruct to the image of all lines*; *a partial fourier image is closer to the full one than the solve of its lines alone* (`test_pics.py`).
- **Runs a plugin offline on a kept series, unchanged.**

  Code: `python -m pulserver.recon`. Test: *plugin runs unchanged offline* (`test_app.py`).

:::

## What bartorch and your plugin do

- **bartorch** computes the whitening, the compression basis, the coil maps and the reconstruction.
- **Your plugin** chooses the gadgets, the axes, the triggers and the reconstruction.

## How it works

```{figure} ../_static/context.svg
:figclass: only-light

Calibration stored by one series of an exam and reused by the next only on an exact match.
```

```{figure} ../_static/context-dark.svg
:figclass: only-dark

Calibration stored by one series of an exam and reused by the next only on an exact match.
```

### Units

A unit is the readouts of one branch (such as `imaging` or `navigator`) and
encoding space that share their slice, contrast, phase, repetition, set and
average counters. The counters you name in `axes`, such as the echoes, become
axes of its k-space instead. Readouts flagged as calibration go to `ref`,
calibration-and-imaging to both, phase correction to neither, as in Gadgetron's
acquisition bucket. A unit still open at the end of the stream is
reconstructed then.

### Images

An image's field of view is that of its unit's reconstruction space, and its
position, orientation and time stamps those of the *reference acquisition*,
the one placed nearest the k-space centre. Values are written as returned
(`float32`, `complex64` or integer), so ratios between the images of a series
are those of the reconstruction.

### The calibration context

Maps estimated from whitened, compressed data describe only data with the same
channels, whitening, compression and voxel grid. So a stored set records all
of them, and {func}`~pulserver.recon.coil_maps` hands it to a unit only where
every field is equal; otherwise it raises
{class}`~pulserver.recon.MissingCalibration`, naming each source and the field
that differed. Nothing is resampled. The model and the comparison are in
{doc}`../developer-guide/internals/calibration`; units and image headers in
{doc}`../developer-guide/internals/reconstruction-proxy`.

## See it run

- {doc}`../generated/gallery/01-course/04_reconstruction_plugin`: a plugin over the units of `gre2d`.
- {doc}`../user-guide/reconstruction-plugins`: writing a plugin, with prewhitening, compression and maps.
- {doc}`reconstruction-session`: how a series reaches the plugin.
- {doc}`../api/recon`: the plugin interface.
