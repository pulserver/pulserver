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
{doc}`apps` for the zoo built on it.

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

## The chain

What happens to a readout on the way in. Gadgetron puts these steps ahead of
the buffer, and for the same reason: each is per-acquisition work that has
nothing to do with the reconstruction after it, and a readout is only worth
placing once it has had them. A plugin lists the ones it wants as its `chain`,
and a step that returns `None` consumes the acquisition — which is how a noise
scan and a navigator line reach no buffer at all.

| Pulserver | Gadgetron |
|---|---|
| `chain` | the gadgets ahead of the buffer |
| `NoiseAdjust` | `NoiseAdjustGadget` |
| `CoilCompression` | `PCACoilGadget` |
| `EpiPhaseCorrection` | `EPICorrGadget` |
| `RampSampling` | `EPIReconXGadget` |
| `branches` | `AcquisitionAccumulateTriggerGadget` |
| `ReconData` / `ReconBuffer` | `BucketToBufferGadget` |

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   Gadget
   NoiseAdjust
   CoilCompression
   EpiPhaseCorrection
   RampSampling
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
   :toctree: ../generated/recon

   user_parameter
   diffusion_table
```

## Preprocessing

What happens to the measurement before it is inverted: noise, oversampling,
coil count, and the corrections a particular readout demands. The first two are
per-acquisition work, which is why a plugin does them in `receive` rather than
waiting for a trigger.

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
conjugate symmetry an image with slowly varying phase carries. `POCS` iterates
towards an image that reproduces every acquired sample and `Homodyne` reaches
an answer in one pass; `fill_partial_echo` takes either by name, and that name
is what a plugin exposes as its `partial_fourier` setting.

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
   :toctree: ../generated/recon

   estimate_epi_phase
   correct_lines
   epi_ramp_operator
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

A solve spends nearly all of its time in the normal operator, so the
non-Cartesian operators build a Toeplitz transfer kernel for it on first use
and run on the host's GPU when there is one. Both are defaults rather than
options — `toeplitz=False` and `device=None` opt out — and neither changes the
answer, only what it costs. `Toeplitz` is the spelling for building the kernel
with different options.

The transfer is the trajectory **gridded** onto a grid twice the image in
every dimension: the adjoint of the sample weights — ones for a plain normal,
the density for a compensated one, a basis product for a subspace frame or an
off-resonance segment — transformed. That is the Gram of the transform
actually being inverted, and it is what puts the weight where the scan is:
a projection scan leaves a ball, so the cube's corners hold little.

Two things then make the kernel small. `compress=True`, the default, keeps
only the locations the gridding actually reached, the way BART's
`--compress-psf` and `MRISubspaceRecon.jl`'s `calculate_kmask_indcs` do; for a
subspace kernel the mask is the union over every frame. How much it saves is
the scan's business — a 3D projection keeps under half the grid and less as
the matrix grows, a 2D radial at Nyquist keeps nearly all of it, because in
two dimensions the transfer really is dense. `compress=False` keeps the whole
grid, which is what a calibration wants: `NLINV` solves over a small centred
window where dropping what the trajectory never reached would cost the solve
its positive definiteness.

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

`pics` is the solve: a measurement, a physics, and a prior. The other two are
the compositions of it a filled buffer is reconstructed by, so a plugin does it
in one call: `cartesian_recon` chooses between the adjoint, the partial-Fourier
constraint and CG-SENSE by what the buffer's mask says the scan sampled, and
`noncartesian_recon` between the density-compensated adjoint and CG-SENSE by
whether the views reach Nyquist.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   pics
   cartesian_recon
   noncartesian_recon
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
   image_result
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
