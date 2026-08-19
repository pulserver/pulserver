# pulserver.recon

Everything a reconstruction needs, reached one way: `pulserver.recon.<name>`.
There are no subpackages to remember and no name that lives in two places, so
an import says what a plugin uses rather than where the library happens to
keep it.

```python
import pulserver.recon as recon

physics = recon.NonCartesian2D(trajectory, (256, 256))
image = recon.pics(kspace, physics)
```

The sections below group the names by what they are for. That grouping is a
way of reading the library, not a constraint on importing from it — every
name on this page is reachable directly from `pulserver.recon`.

Names resolve on first use, so importing the module needs neither an MRD
server environment nor any optional numerical backend; only what a plugin
touches pulls its dependencies in.

The primitives are backend-polymorphic: NumPy in, NumPy out; Torch in, Torch
out, on the device the tensors arrived on. A plugin composes them without
converting by hand.

```{eval-rst}
.. currentmodule:: pulserver.recon
```

## The plugin contract

What the runtime drives, and what it hands a plugin. A reconstruction plugin
is one `ReconPlugin` subclass and a module-level `PLUGIN` instance; see
{doc}`app_recon` for the zoo built on it.

Three hooks, and the division between them is the same in every plugin:
`startup` lays out the buffers the header's encoding spaces describe, `receive`
places each acquisition and — reading its flags — routes the boundaries it
closes to a named branch, and `recon` holds the reconstruction of each branch
over buffers that are already filled.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   ReconPlugin
   ReconContext
   ReconResult
   AcquisitionBucket
   AcquisitionBucketStats
   AcquisitionFlag
   ExamCache
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   has_acquisition_flag
```

## Buffers

Where the acquisitions go. The header describes every encoding space the scan
will produce and each acquisition names its own, so the layout is known before
the first line arrives and `ReconPlugin.receive` places each one as it lands —
a plugin reconstructing sorted k-space overrides no hook at all.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   ReconData
   ReconBuffer
   EncodingSpace
```

## Reading the header

What a scan declares about itself. The encoded grid is not here: `ReconBuffer`
is laid out from it and answers with `.axes` and `.kspace.shape`, so a second
reading of the same fields could only disagree with the buffer a plugin
actually fills. What is left is what the header says and the buffer does not —
the matrix to crop to, and how many echoes to loop over.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   recon_shape
   recon_volume
   echo_count
   diffusion_table
```

## Preprocessing

What happens to the measurement before it is inverted: noise, oversampling,
coil count, and the corrections a particular readout demands.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   noise_prewhiten
   coil_compress
   remove_readout_oversampling
   fftc
   ifftc
   pipe_menon_dcf
```

### Partial Fourier

Recovering the k-space edge a partial acquisition never took, from the
conjugate symmetry an image with slowly varying phase carries.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   fill_partial_echo
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   POCS
   Homodyne
```

### EPI

A reversed EPI line needs the odd/even ramp fitted from a navigator before it
belongs anywhere: `odd_even_fit` reads the ramp off a blip-nulled triplet and
`correct_lines` flips and demodulates by it, which is what a plugin does in
`receive` before placing the readout.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   odd_even_fit
   correct_lines
   epi_ramp_interpolate
   estimate_epi_eddy_phase
   correct_epi_eddy_currents
```

## Calibration

Estimating what the reconstruction needs but the scan does not measure
directly: coil sensitivities, and the point-spread function of a wave
encoding.

`NLINV` solves for the maps and the object together and is the better estimate
for one image; `coil_maps_from_reference` reads them straight off a prescan and
is the one to use when several images are solved against each other, because
maps read off one reference share a scale a per-image solve does not guarantee.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   calibration_extent
   coil_maps_from_reference
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   NLINV
   NLINVPhysics
   NLINVResult
   PhasePoleCorrection
   WavePSF
   WavePSFCalibration
   WavePSFResult
```

## Physics

The forward model: what the scanner did to the image to produce the
measurement. Every solver takes one of these.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   MRIPhysics
   Cartesian2D
   Cartesian3D
   NonCartesian2D
   NonCartesian3D
   SMS
   Subspace
   WaveEncoding
   WaveShuffling
   OffResonance
   Toeplitz
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   available_nufft_backends
```

## Solvers

`pics` is the solve: a measurement, a physics, and a prior. `cartesian_recon`
is the Cartesian composition of it — the adjoint, the partial-Fourier
constraint and the CG-SENSE solve, chosen by what the buffer's mask says the
scan sampled, so a plugin reconstructs a filled buffer in one call.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   pics
   cartesian_recon
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   ConjugateGradient
   FISTA
   ADMM
   PDHG
   IRGNM
   PolynomialPreconditioner
   StackedPrior
   OptimResult
   OptimState
   CGInfo
```

## Regularizers

Priors the iterative solvers minimize against.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   TV
   TGV
   LLR
   Wavelet
   Positive
   AverageDenoiser
```

## Learned reconstruction

Unrolled networks and the stores their weights come from. A real-valued
network sees complex data through adapters applied internally; nothing has to
be packed or unpacked by the caller.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   UnrolledReconstructor
   StatefulReconstructor
   UnrollResult
   UnrollState
   GradientDataConsistency
   ScaledAdjoint
   ContextAgnosticDenoiser
```

### Weights

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   ModelBundle
   ModelStore
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   load_model
   save_bundle
   default_model_paths
   MODEL_PATH_ENV
```

### Datasets

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   TorchIODataset
   IXI
   IXITiny
```

## Sequence description

The description of itself a scan carries, decoded from its MRD waveforms --
the same object the design side writes.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
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
   :toctree: ../generated/recon

   decode_sequence_description
   decompress_shape
```

## Motion

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   RigidRegistration
   RigidMotionEKF
   RigidMotionEstimate
```

## Postprocessing

What happens to the image on its way out: coil combination, cropping,
gradient-nonlinearity unwarping, distortion correction.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   coil_combine
   center_crop
   as_numpy
   run_pyhysco
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   Gradunwarp
   GradientCoefficients
   CoefficientAccessor
   ImageGeometry
```

## Execution

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   CudaStreaming
```
