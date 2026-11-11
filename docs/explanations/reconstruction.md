# Reconstruction plugins and calibration context

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

## Capabilities

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
- **Writes each image header from the acquisition nearest the k-space centre, and keeps the values your plugin returns.**

  Code: {class}`~pulserver.recon.ReconResult`. Tests: *the position and physiology are those of the acquisition nearest the k space centre*; *the ratios between the images of a series survive to the mrd images and the dicom pixels* (`test_images.py`).
- **Prewhitens on arrival, from the series' noise readouts or the exam's noise covariance.**

  Code: {class}`~pulserver.recon.Prewhiten`. Test: *noise readouts whiten the imaging readouts* (`test_prewhiten.py`).
- **Compresses coils with one basis per stream, as the readouts arrive or as a unit closes.**

  Code: {class}`~pulserver.recon.CompressChannels`, {class}`~pulserver.recon.CoilCompression`. Tests: *virtual channels are uncorrelated and carry the leading eigenvalues*; *compressed data and maps reconstruct the uncompressed image*; *calibration readouts are held then given on projected ahead of the first imaging readout*; *the basis comes from the calibration readouts of the first contrast* (`test_coil_compression.py`); *contrasts open together are buffered with the virtual channels and lose nothing* (`test_pics.py`).
- **Finds coil maps in the unit, the stream or the exam, and refuses maps that differ in any recorded field.**

  Code: {func}`~pulserver.recon.coil_maps`, {class}`~pulserver.recon.CoilSensitivities`. Tests: *a later unit without calibration readouts is given the maps of its slice*; *maps that differ in a field are refused naming it with both values*; *exam maps at another matrix are refused not resampled* (`test_calibration.py`).
- **Shares an exam cache across the series of one exam.**

  Code: {class}`~pulserver.recon.ExamCache`. Test: *a value one cache stores is read by the next on its directory* (`test_app.py`).
- **Ships a `pics` plugin with prewhitening, coil compression, partial Fourier and asymmetric echo.**

  Code: `src/pulserver/_zoo/recon/pics.py`. Tests: *every other phase encode and the centre reconstruct to the image of all lines*; *a partial fourier image is closer to the full one than the solve of its lines alone*; *homodyne over two partial fourier axes together beats one axis after the other*; *a multi echo scan sends the real and imaginary parts of each echo as series of their own*; *channels beyond the virtual coils are compressed before the maps and the solve* (`test_pics.py`).
- **Corrects the images of every shipped reconstruction in one order, each step on unless the scan or the site turns it off: a Fermi filter of the solved image's k-space, gradient nonlinearity, and the N4 receive field of the magnitude, one per slice and set.**

  The receive field is fitted to each slice of a 2D multi-slice series on its own and normalised to mean 1 inside that slice's object, so the shipped reconstructions do not keep intensity ratios between slices.

  Code: {class}`~pulserver.recon.Corrections`, {func}`~pulserver.recon.fermi_filtered`, {func}`~pulserver.recon.gradient_unwarped`, {class}`~pulserver.recon.IntensityCorrection`. Tests: *each correction is made unless the header turns it off*; *the corrections run in order fermi then unwarp then n4 and each says so*; *a header stating no coil table is not unwarped and the log says why* (`test_pics.py`); *every shipped reconstruction makes the three corrections by default*; *a streams corrections build the unwarp of each geometry once* (`test_correct.py`); *the filtered image has the k space of the image times the fermi window* (`test_correct.py`); *a stacks partitions are fermi filtered before they are transformed* (`test_nufft.py`); *a shaded uniform object is flattened and keeps its mean*; *separate compartments keep their intensity ratios*; *later contrasts and repetitions of a slice share its first field* (`test_intensity_correction.py`); *the magnitude series is intensity corrected and the echoes the maps take are not* (`test_qsm.py`).
- **Keeps a unit's k-space, maps and solver vectors on the host, the encoding computing on the series' GPU.**

  Code: `pics_image` in `src/pulserver/recon/handlers/pics.py`. Test: *the solve on host k space is bartorchs pics with the image on the host* (`test_pics.py`).
- **Runs a plugin offline on a kept series, unchanged.**

  Code: `python -m pulserver.recon`. Test: *plugin runs unchanged offline* (`test_app.py`).

:::

## Division of responsibility

- **bartorch** computes the whitening, the compression basis, the coil maps, the reconstruction, the Fermi window, the gradient nonlinearity correction and the N4 field.
- **Your plugin** chooses the gadgets, the axes, the triggers and the reconstruction.

## Method

```{figure} ../_static/context.svg
:figclass: only-light

Calibration stored by one series of an exam and reused by the next with the same coils, whitening and frame of reference, resliced onto its grid.
```

```{figure} ../_static/context-dark.svg
:figclass: only-dark

Calibration stored by one series of an exam and reused by the next with the same coils, whitening and frame of reference, resliced onto its grid.
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

### Calibration context

Maps estimated from whitened, compressed data describe only data with the same
channels, whitening, compression and voxel grid. So a stored set records all
of them. {func}`~pulserver.recon.coil_maps` hands maps a series stored for
itself to a unit only where every field is equal. Maps of the exam, measured
by a calibration series, serve a unit with the same coils, whitening, frame of
reference and table position: on another position, orientation or grid they
are resliced onto the unit's grid, and maps of uncompressed channels are
projected onto the unit's compression basis. A unit's own calibration readouts
always win over both. Where no source serves, it raises
{class}`~pulserver.recon.MissingCalibration`, naming each source and why it
was not used, a unit reaching outside the calibration volume among them. The model and the comparison are in
{doc}`../developer-guide/internals/calibration`; units and image headers in
{doc}`../developer-guide/internals/reconstruction-proxy`.

## Examples

- {doc}`../generated/gallery/01-course/04_reconstruction_plugin`: a plugin over the units of `gre2d`.
- {doc}`../user-guide/reconstruction-plugins`: writing a plugin, with prewhitening, compression and maps.
- {doc}`reconstruction-session`: how a series reaches the plugin.
- {doc}`../api/recon`: the plugin interface.
