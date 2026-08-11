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
and CUFINUFFT on a CUDA host. NLINV is part of the CPU/CUDA numerical stack;
the opt-in `recon-distortion` and `recon-sim` extras add reverse-polarity
distortion correction and TorchSim EPG simulation, respectively.

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
   pulserver.recon.optim
   pulserver.recon.simulation
```

## Reconstruction algorithms

`pics` selects implicitly differentiated conjugate gradients for an
unregularized or quadratic problem and plug-and-play FISTA when one denoiser
is supplied. The optimizer classes follow DeepInverse's constructor and
`model(y, physics)` conventions.

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.algorithms.pics
   pulserver.recon.algorithms.PolynomialPreconditioner
```

### Multiple regularizers

`StackedPrior` distinguishes simultaneous regularizers from DeepInverse's
iteration-wise prior lists. FISTA accepts one proximal prior. PDHG and ADMM
accept multiple transformed priors without silently replacing their proximal
sum by an average.

```python
import deepinv as dinv
import pulserver.recon as recon

prior = recon.optim.StackedPrior(
    priors=[dinv.optim.L1Prior(), dinv.optim.L1Prior()],
    transforms=[wavelet_transform, finite_difference_transform],
    weights=[1e-3, 2e-4],
)
model = recon.optim.PDHG(
    data_fidelity=dinv.optim.L2(),
    prior=prior,
    stepsize=0.2,
    stepsize_dual=0.2,
    max_iter=50,
)
image = model(kspace, physics)
```

ADMM uses the physics' fast `A_adjoint_A` method for its image update, so a
Toeplitz MRI physics remains active underneath the optimizer. Its CG solve has
a manual implicit backward and does not retain the Krylov history.

`IRGNM(inner=...)` is the nonlinear composition layer. An analytic
`linearize(x)` is used when available. An ordinary DeepInverse `Physics`
works without extra glue: Pulserver builds matrix-free JVP/VJP products with
`torch.func`. The selected FISTA, PDHG, ADMM, or CG instance remains
responsible for each linearized subproblem.

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.optim.FISTA
   pulserver.recon.optim.IRGNM
   pulserver.recon.optim.PDHG
   pulserver.recon.optim.ADMM
   pulserver.recon.optim.StackedPrior
   pulserver.recon.optim.ConjugateGradient
   pulserver.recon.optim.OptimState
   pulserver.recon.optim.OptimResult
```

### Advanced training loops

The regular call remains `model(y, physics)`. Training strategies that need
intermediate losses, truncated backpropagation, alternating data splits, or
progressive freezing can use the same model one iteration at a time:

```python
import torch

state = model.init_state(kspace, physics)
for iteration in range(model.iterations):
    state = model.step(state, kspace, physics, iteration)
    if iteration in supervised_iterations:
        loss = loss + criterion(model.get_output(state), target)
    if (iteration + 1) % truncate_every == 0:
        state = state.detach()

model.set_trainable("prior", enabled=False)
optimizer = torch.optim.AdamW(model.parameter_groups())
```

`return_info=True` with `record_iterations=(...)` is the concise alternative
when only selected intermediate estimates are needed. This protocol is plain
PyTorch, so Lightning can own the training schedule and TorchIO can own patch
sampling without either becoming a Pulserver runtime dependency.

## Forward operators

Every class is a DeepInverse `LinearPhysics`, with MRI metadata and execution
policy supplied by the common `MRIPhysics` base. Start with a Cartesian,
non-Cartesian, or Wave acquisition operator, then compose dynamic subspace,
off-resonance, and Toeplitz behavior explicitly.

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.physics.MRIPhysics
   pulserver.recon.physics.Cartesian2D
   pulserver.recon.physics.Cartesian3D
   pulserver.recon.physics.NonCartesian2D
   pulserver.recon.physics.NonCartesian3D
   pulserver.recon.physics.WaveEncoding
   pulserver.recon.physics.WaveShuffling
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
   pulserver.recon.calibration.NLINV
   pulserver.recon.calibration.NLINVPhysics
   pulserver.recon.calibration.NLINVResult
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
