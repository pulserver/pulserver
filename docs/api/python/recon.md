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

```{eval-rst}
.. currentmodule:: pulserver.recon
```

## The plugin contract

What the runtime drives, and what it hands a plugin. A reconstruction plugin
is one `ReconPlugin` subclass and a module-level `PLUGIN` instance; see
{doc}`app_recon` for the zoo built on it.

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
   CartesianGridder
```

## Reading the header

What a scan declares about itself, for the shapes a reconstruction allocates
and crops to.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   encoded_shape
   encoded_volume
   recon_shape
   recon_volume
   receiver_channels
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
   grid_cartesian
   cartesian_3d_to_2d
   fftc
   ifftc
   pipe_menon_dcf
```

### EPI

An EPI stream is several roles in one, and its reversed lines need the
odd/even ramp fitted from a navigator before they belong anywhere.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   partition_epi_acquisitions
   odd_even_fit
   correct_lines
   epi_ramp_interpolate
   estimate_epi_eddy_phase
   correct_epi_eddy_currents
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   EpiAcquisitionGroups
   EPIPhaseCorrection
   SmsEpiInputs
```

## Calibration

Estimating what the reconstruction needs but the scan does not measure
directly: coil sensitivities, and the point-spread function of a wave
encoding.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   sensitivities
   smooth_sensitivities
   calibration_extent
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
   measurement_to_channels
   measurement_to_trailing
```

## Reconstruction

The recipes that compose a physics operator with a solver — the part every
Cartesian reconstruction repeats. Two and three dimensions are the same
recipe, read off the sampling mask.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   coil_images
   sense
   fill_partial_echo
   reconstruct_plane
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   POCS
   Homodyne
```

## Solvers

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   pics
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

Unrolled networks, the adapters that let a real-valued network see complex
data, and the stores their weights come from.

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
   ComplexAdapter
   MoDL
   VarNet
   RAM
   ContextAgnosticDenoiser
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   as_complex_channels
   as_real_channels
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

   ImageDataset
   ImageFolder
   TensorDataset
   PatchDataset
   RandomPatchSampler
   HDF5Dataset
   TorchIODataset
   MRISliceTransform
   FastMRISliceDataset
   SimpleFastMRISliceDataset
   CMRxReconSliceDataset
   SKMTEASliceDataset
   LidcIdriSliceDataset
   IXI
   IXITiny
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon

   check_dataset
   download_archive
   generate_dataset
```

## Simulation

Bloch and EPG signal models, and the sequence description a scan carries so a
simulation can be driven from the sequence itself.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/recon
   :template: autosummary/class.rst

   SPGR
   BSSFP
   SSFPFID
   SSFPEcho
   FSE
   EpgInterpreter
   TissueProperties
   SimulationResult
   SubspaceBasis
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

   make_interpreter
   simulate_subspace
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
