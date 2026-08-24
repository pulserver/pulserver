# pulserver.mrd

The data model a scan is described by, and the array toolbox both sides of it
speak. A sequence writes a scan into this vocabulary; a reconstruction reads
it back out. Everything is reached one way: `pulserver.mrd.<name>`.

```python
import pulserver.mrd as mrd

clean = mrd.noise_prewhiten(data, noise)
image = mrd.center_crop(mrd.coil_combine(coil_images), shape)
```

The sections below group the names by what they are for. That grouping is a
way of reading the library, not a constraint on importing from it — every name
on this page is reachable directly from `pulserver.mrd`.

Names resolve on first use, so importing the module needs neither an MRD
server environment nor any optional numerical backend.

The array operations are backend-polymorphic: NumPy in, NumPy out; Torch in,
Torch out, on the device the tensors arrived on. A plugin composes them
without converting by hand.

```{eval-rst}
.. currentmodule:: pulserver.mrd
```

## Acquisitions

What a scanner sends, one readout at a time: a line of k-space, the counters
saying where in the encoding space it belongs, and the flags saying what it is
for. A bucket is what they accumulate into while a boundary is still open.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd
   :template: autosummary/class.rst

   AcquisitionFlag
   AcquisitionBucket
   AcquisitionBucketStats
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd

   has_acquisition_flag
   acquisition_label
   acquisition_labels
```

## The header

Almost nothing is here, and that is the point: the header's encoding spaces are
what `ReconBuffer` is laid out from, so it answers for them — `.extents` for
how far each axis runs, `.image_shape` for the matrix to crop to. A second
reading of the same fields could only disagree with the buffer a plugin
actually fills.

What is left is what the encoding spaces do not describe: the free-form
parameters a sequence attached to the scan. MRD splits those across four typed
collections of `name`/`value` pairs, so `user_parameter` searches all four and
answers with the value, whichever it was written as.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd

   user_parameter
   diffusion_table
```


## Arrays

What happens to the measurement before it is inverted: noise, oversampling,
coil count, and the corrections a particular readout demands. The first two are
per-acquisition work, which is why a plugin does them in `receive` rather than
waiting for a trigger.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd

   noise_prewhiten
   coil_compress
   remove_readout_oversampling
   fftc
   ifftc
   pipe_menon_dcf
```

### Partial Fourier

Recovering the k-space edge a partial acquisition never took, from the
conjugate symmetry an image with slowly varying phase carries. `POCS` iterates
towards an image that reproduces every acquired sample and `Homodyne` reaches
an answer in one pass; `fill_partial_echo` takes either by name, and that name
is what a plugin exposes as its `partial_fourier` setting.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd

   fill_partial_echo
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd
   :template: autosummary/class.rst

   POCS
   Homodyne
```

### EPI

A reversed EPI line carries a phase its forward neighbours do not, and leaving
it there puts a ghost at half the field of view. `estimate_epi_phase` reads that
phase off a blip-nulled navigator triplet and `correct_lines` flips and
demodulates by it — which is what a plugin does in `receive`, before the readout
is placed, because a corrected line is what belongs on the grid.

One estimator, with the order as its parameter: first order is the
gradient-delay ramp every product reconstruction corrects, and raising it picks
up what an eddy current leaves beyond a ramp.

A train worth playing samples across its read ramps rather than waiting for the
plateau, so k does not advance at a constant rate along a readout and the
samples are not on the grid. Where they fell is the trajectory the acquisition
carries — a client attaches one exactly when the gradient was still moving
under the ADC — and `epi_ramp_operator` is the change of basis onto the grid,
which is exact while the samples outnumber the pixels they determine.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd

   estimate_epi_phase
   correct_lines
   epi_ramp_operator
```


## Sequence description

The description of itself a scan carries, decoded from its MRD waveforms --
the same object the design side writes.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd
   :template: autosummary/class.rst

   SequenceDescription
   SequenceDescriptionCollection
   SequenceEvent
   SequenceParameters
   RfDefinition
   RfShape
   RfUse
   ShimDefinition
   AdcRole
   EventType
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd

   decode_sequence_description
   decompress_shape
```


## Images

What comes out: the coil dimension combined away, the oversampled
reconstruction cropped back to what was prescribed, and the result brought off
whichever device it was computed on.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd

   coil_combine
   center_crop
   as_numpy
```

## Files and DICOM

A stored MRD file replays through the same contract a live scanner drives, so
a reconstruction written for the scanner runs unchanged over a recording.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/mrd

   reconstruct_file
   images_to_dicom
   dicom_folder_to_mrd
```
