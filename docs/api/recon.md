# `pulserver.recon`

`pulserver.recon` is the Torch-first scientific reconstruction API. It owns
the small integration layer between MRI-NUFFT operators, DeepInverse inverse
problem algorithms, and Pulserver-specific dynamic MRI models.

The API is module-oriented, in the style of DeepInverse, MRI-NUFFT, and SciPy.
The root namespace exposes the modules below and one convenience entry point,
`pics` (documented as {func}`pulserver.recon.algorithms.pics`). MRD
connections, message serialization, server lifecycle, and Gadgetron handlers
are implementation details and are not part of the public API.

```python
import pulserver.recon as recon

physics = recon.physics.NonCartesian3D(
    trajectory,
    image_shape=(192, 192, 128),
    coil_maps=coil_maps,
)
physics = recon.physics.Subspace(physics, basis)

coefficients = recon.pics(
    dynamic_kspace,
    physics,
    denoiser=recon.denoisers.TV(n_it_max=20),
    regularization=0.02,
    iterations=50,
)
```

Install `pulserver[recon-cpu]` for the portable numerical stack or
`pulserver[recon-cuda]` for Torch-native CUFINUFFT support without CuPy. The
default non-Cartesian `backend="auto"` chooses FINUFFT when CUDA is unavailable
and CUFINUFFT on a CUDA host. The opt-in extras
`recon-nlinv`, `recon-distortion`, and `recon-sim` add NLINV calibration,
reverse-polarity distortion correction, and TorchSim EPG simulation,
respectively.

For operator composition, Toeplitz behavior, and bounded-memory CUDA
execution, see {doc}`../explanations/reconstruction/model_based`.

## API organization

```{eval-rst}
.. autosummary::
   :toctree: generated/recon

   pulserver.recon.algorithms
   pulserver.recon.physics
   pulserver.recon.denoisers
   pulserver.recon.density
   pulserver.recon.calibration
   pulserver.recon.preprocessing
   pulserver.recon.corrections
   pulserver.recon.execution
   pulserver.recon.simulation
```

## Reconstruction algorithms

`pics` selects conjugate gradients for an unregularized or quadratic problem
and plug-and-play FISTA when a denoiser is supplied. A sequence of denoisers
is combined with an equal-weight proximal average.

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.algorithms.pics
   pulserver.recon.algorithms.PolynomialPreconditioner
```

## Forward operators

Classes return a common `MRIPhysics` facade. Start with a Cartesian or
non-Cartesian acquisition operator, then compose dynamic subspace,
off-resonance, and Toeplitz behavior explicitly.

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.physics.MRIPhysics
   pulserver.recon.physics.Cartesian2D
   pulserver.recon.physics.NonCartesian2D
   pulserver.recon.physics.NonCartesian3D
   pulserver.recon.physics.Subspace
   pulserver.recon.physics.OffResonance
   pulserver.recon.physics.Toeplitz
   pulserver.recon.physics.available_nufft_backends
```

## Denoisers and priors

The CamelCase constructors follow DeepInverse's public style and return
modules compatible with its plug-and-play optimizers.

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.denoisers.AverageDenoiser
   pulserver.recon.denoisers.LLR
   pulserver.recon.denoisers.Wavelet
   pulserver.recon.denoisers.TV
   pulserver.recon.denoisers.TGV
```

## Density compensation and calibration

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.density.pipe_menon_dcf
   pulserver.recon.calibration.estimate_sensitivities
   pulserver.recon.calibration.nlinv_sensitivities
```

## Preprocessing

These functions operate on arrays or tensors and stay independent of an MRD
connection or streaming server.

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.preprocessing.cartesian_3d_to_2d
   pulserver.recon.preprocessing.remove_readout_oversampling
   pulserver.recon.preprocessing.coil_compress
   pulserver.recon.preprocessing.noise_prewhiten
   pulserver.recon.preprocessing.epi_ramp_interpolate
   pulserver.recon.preprocessing.estimate_epi_eddy_phase
   pulserver.recon.preprocessing.correct_epi_eddy_currents
```

## Image corrections

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.corrections.CoefficientAccessor
   pulserver.recon.corrections.GradientCoefficients
   pulserver.recon.corrections.ImageGeometry
   pulserver.recon.corrections.Gradunwarp
   pulserver.recon.corrections.run_pyhysco
```

## Execution policies

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.execution.CudaStreaming
```

## Sequence simulation

The simulation module decodes LiveSDK sequence-description waveforms into a
vendor-neutral event model and estimates TorchSim EPG signal dictionaries and
temporal subspaces. See {doc}`../explanations/reconstruction/simulator` for
the event contract.

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.simulation.decode_sequence_description
   pulserver.recon.simulation.SequenceDescription
   pulserver.recon.simulation.TissueProperties
   pulserver.recon.simulation.EpgInterpreter
   pulserver.recon.simulation.FSE
   pulserver.recon.simulation.SPGR
   pulserver.recon.simulation.SSFPEcho
   pulserver.recon.simulation.SSFPFID
   pulserver.recon.simulation.BSSFP
   pulserver.recon.simulation.simulate_subspace
```
