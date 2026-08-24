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
   ExamCache
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

What happens to the image on its way out: packaging it as a result the
runtime can send, unwarping gradient nonlinearity, correcting susceptibility
distortion. The array operations underneath -- combining coils, cropping,
bringing a result off its device -- are {mod}`pulserver.mrd`'s.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   image_result
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
