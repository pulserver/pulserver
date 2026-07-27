# `pulserver.recon`

`pulserver.recon` combines the Gadgetron-compatible MRD server with a compact
Torch-first reconstruction API. Pulserver owns the integration boundary while
mri-nufft provides non-Cartesian operators and DeepInverse provides linear
physics, denoisers, CG, and FISTA. Torch owns CPU/CUDA dispatch, so tensors
remain on the caller-selected device.

Install `pulserver[recon]` for the server, DICOM, and MRD layers.
Install `pulserver[recon-cpu]` for FINUFFT, MRPro, and DeepInverse; add
`pulserver[recon-nlinv]` for PyGROG NLINV calibration or install
`pulserver[recon-cuda]` for CUFINUFFT on Linux CUDA hosts.
Install `pulserver[recon-distortion]` only when PyHySCO reverse-polarity
distortion correction is required. PyHySCO is GPL-3.0-only, so Pulserver does
not vendor or import it.

Gradient nonlinearity correction is included in `pulserver[recon]`. It accepts
a coefficient-file path, serialized coefficient text, or a keyed string
parameter from the vendor-neutral MRD XML header. It uses the reconstructed
matrix rather than the acquired matrix when constructing its physical grid:

```python
from pulserver.recon import (
    Gradunwarp,
    ImageGeometry,
    MrdCoefficientAccessor,
)

geometry = ImageGeometry.from_mrd(
    image_header,
    shape=reconstructed_volume.shape[-3:],
)
correct = Gradunwarp.from_file(
    MrdCoefficientAccessor(
        mrd_xml_header,
        key=coefficient_parameter_key,
    ),
    geometry,
)
corrected_volume = correct(reconstructed_volume)
```

The parameter key belongs to the acquisition-client/MRD transport contract and
is therefore supplied by the caller. Coefficient contents are excluded from
object representations. Resampling always uses SimpleITK's native cubic
B-spline implementation in both 2D and 3D.

The main application surface is one physics factory, optional decorators, one
denoiser, and `pics`:

```python
from pulserver.recon import Cartesian2D, NonCartesian3D, Subspace, pics, tv

cartesian = Cartesian2D(mask, coil_maps)
image = pics(kspace, cartesian, iterations=30)

noncartesian = NonCartesian3D(
    trajectory,
    image_shape=(192, 192, 128),
    coil_maps=coil_maps,
    backend="cufinufft",
)
dynamic = Subspace(noncartesian, basis)
coefficients = pics(
    dynamic_kspace,
    physics=dynamic,
    denoiser=tv(n_it_max=20),
    regularization=0.02,
    iterations=50,
)
```

With no denoiser, `pics` uses CG and solves
`(AᴴA + λI)x = Aᴴy`. With a denoiser it uses plug-and-play FISTA, where
`regularization` is the denoiser threshold/noise level.

LLR treats image channels as contrasts or subspace coefficients and applies
nuclear soft-thresholding to local 2D or 3D blocks. The implementation stays
entirely in Torch, uses the small channel-channel Gram matrix when channels
are fewer than block voxels, and bounds its workspace with
`block_batch_size`:

```python
from pulserver.recon import llr, pics, wavelet

regularizers = [
    llr(dimension=3, block_size=8, block_batch_size=1024),
    wavelet(dimension=3, level=3, complex_data=True),
]
coefficients = pics(
    dynamic_kspace,
    physics=Toeplitz(dynamic),
    denoiser=regularizers,  # equal-weight proximal average
    regularization=0.02,
    polynomial_degree=3,
    iterations=30,
)
```

Passing a denoiser sequence is equivalent to wrapping it with
`AverageDenoiser`. A positive `polynomial_degree` uses the L2-optimal
polynomial preconditioner for the FISTA gradient. Degree `d` adds `d`
applications of the normal operator per iteration, making this most useful
when `AᴴA` is much cheaper than denoising, as with a Toeplitz normal.

Pipe--Menon density compensation delegates to the installed MRI-NUFFT
backend:

```python
from pulserver.recon import pipe_menon_dcf

dcf = pipe_menon_dcf(
    trajectory,
    image_shape=(256, 256, 192),
    backend="cufinufft",
    max_iter=30,
)
```

`Subspace`, `OffResonance`, and `Toeplitz` are composable decorators:

```python
from pulserver.recon import NonCartesian2D, OffResonance, Subspace, Toeplitz

physics = NonCartesian2D(trajectory, (256, 256), coil_maps=coil_maps)
physics = OffResonance(physics, field_map, readout_time)
physics = Subspace(physics, basis)  # off-resonance must precede subspace
physics = Toeplitz(physics)
```

Base mri-nufft operators use native Toeplitz kernels. Cartesian FFTs are
already exact. Stacked, subspace, and off-resonance compositions use a correct
exact normal operation when mri-nufft has no combined Toeplitz kernel.

## Reconstruction primitives

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.physics.Cartesian2D
   pulserver.recon.physics.NonCartesian2D
   pulserver.recon.physics.NonCartesian3D
   pulserver.recon.physics.Subspace
   pulserver.recon.physics.OffResonance
   pulserver.recon.physics.Toeplitz
   pulserver.recon.algorithms.pics
   pulserver.recon.denoisers.average
   pulserver.recon.denoisers.llr
   pulserver.recon.denoisers.wavelet
   pulserver.recon.denoisers.tv
   pulserver.recon.denoisers.tgv
   pulserver.recon.density.pipe_menon_dcf
   pulserver.recon.optimizers.PolynomialPreconditioner
   pulserver.recon.preprocessing.cartesian_3d_to_2d
   pulserver.recon.preprocessing.remove_readout_oversampling
   pulserver.recon.preprocessing.coil_compress
   pulserver.recon.preprocessing.noise_prewhiten
   pulserver.recon.linops.available_nufft_backends
   pulserver.recon.calibration.nlinv_sensitivities
   pulserver.recon.calibration.estimate_sensitivities
```

## Gadgetron-style MRD helpers

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.metadata.MrdMetadata
   pulserver.recon.metadata.user_parameter
   pulserver.recon.metadata.acquisition_label
   pulserver.recon.metadata.acquisition_labels
   pulserver.recon.metadata.has_acquisition_flag
   pulserver.recon.grouping.group_by_labels
   pulserver.recon.grouping.split_on_flag
   pulserver.recon.grouping.filter_acquisitions
   pulserver.recon.serialization.images_to_dicom
   pulserver.recon.serialization.write_dicom_series
   pulserver.recon.mrd2dicom.MrdDicomBuilder
   pulserver.recon.dicom2mrd.dicom_folder_to_mrd
```

## EPI and SMS preprocessing

The EPI zoo can optionally emit a blip-nulled `NAV`/`REF` navigator for
odd/even phase calibration, a `SET=1` reverse-phase-encode b=0 reference, and
a single-band `REF` volume for SMS. `partition_epi_acquisitions` turns those
standard MRD flags and labels into ordered groups.

`epi_ramp_interpolate` moves ramp-sampled readouts to a uniform grid.
`estimate_epi_eddy_phase` and `correct_epi_eddy_currents` fit and remove the
smooth odd/even navigator phase. Reconstructed reverse-polarity image pairs
can be passed to the opt-in `run_pyhysco` wrapper for distortion correction.
SMS does not need a separate container-level physics type: the known CAIPI
encoding belongs in the forward model, while the single-band reference is
preprocessing/calibration input.

FSE uses its CPMG RF/receiver phase relation and does **not** need EPI's
alternating-readout navigator or an opposite-PE distortion pair. A
phase-sensitive FSE application may still acquire its own phase reference.

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.epi.EpiAcquisitionGroups
   pulserver.recon.epi.partition_epi_acquisitions
   pulserver.recon.preprocessing.epi_ramp_interpolate
   pulserver.recon.preprocessing.estimate_epi_eddy_phase
   pulserver.recon.preprocessing.correct_epi_eddy_currents
   pulserver.recon.sms.SmsEpiInputs
   pulserver.recon.distortion.run_pyhysco
   pulserver.recon.gradunwarp.Gradunwarp
   pulserver.recon.gradunwarp.ImageGeometry
   pulserver.recon.gradunwarp.GradientCoefficients
   pulserver.recon.gradunwarp.MrdCoefficientAccessor
```

## Gadgetron integration

Numerical operators and preprocessing functions remain array-level callables,
not subclasses of a particular Gadgetron ABI. A Gadgetron/Pulserver handler
first uses `filter_acquisitions`, `group_by_labels`, `split_on_flag`, or
`partition_epi_acquisitions`, converts one complete group to tensors, then
calls the same preprocessing and `pics` functions. This keeps stateful stream
grouping at the MRD boundary and makes stateless numerical components directly
testable and reusable offline.

Importable runnable examples mirror the zoo layout:

```python
from pulserver.examples.recon import prepare_epi, prepare_sms_epi

groups = prepare_epi(acquisitions)
sms_inputs = prepare_sms_epi(
    groups.imaging,
    multiband_factor=2,
    caipi_encoding=caipi_encoding,
    coil_maps=coil_maps,
    single_band_reference=groups.single_band_reference,
)
```

## Server submodules

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.connection.Connection
   pulserver.recon.server.Server
   pulserver.recon.rtp_connection.RtpServer
   pulserver.recon.main.main
   pulserver.recon.replay
   pulserver.recon.readers
   pulserver.recon.writers
   pulserver.recon.handlers
   pulserver.recon.mrdhelper
   pulserver.recon.constants
   pulserver.recon.concurrency
```

## Submodule API reference

```{eval-rst}
.. automodule:: pulserver.recon.linops
   :members:
   :no-index:

.. automodule:: pulserver.recon.physics
   :members:
   :no-index:

.. automodule:: pulserver.recon.denoisers
   :members:
   :no-index:

.. automodule:: pulserver.recon.density
   :members:
   :no-index:

.. automodule:: pulserver.recon.algorithms
   :members:
   :no-index:

.. automodule:: pulserver.recon.preprocessing
   :members:
   :no-index:

.. automodule:: pulserver.recon.prox
   :members:
   :no-index:

.. automodule:: pulserver.recon.optimizers
   :members:
   :no-index:

.. automodule:: pulserver.recon.calibration
   :members:
   :no-index:

.. automodule:: pulserver.recon.epi
   :members:
   :no-index:

.. automodule:: pulserver.recon.sms
   :members:
   :no-index:

.. automodule:: pulserver.recon.distortion
   :members:
   :no-index:

.. automodule:: pulserver.recon.gradunwarp
   :members:
   :no-index:

.. automodule:: pulserver.recon.metadata
   :members:
   :no-index:

.. automodule:: pulserver.recon.grouping
   :members:
   :no-index:

.. automodule:: pulserver.recon.serialization
   :members:
   :no-index:

.. automodule:: pulserver.recon.connection
   :members:
   :no-index:

.. automodule:: pulserver.recon.server
   :members:
   :no-index:

.. automodule:: pulserver.recon.rtp_connection
   :members:
   :no-index:

.. automodule:: pulserver.recon.main
   :members:
   :no-index:

.. automodule:: pulserver.recon.readers
   :members:
   :no-index:

.. automodule:: pulserver.recon.writers
   :members:
   :no-index:

.. automodule:: pulserver.recon.mrdhelper
   :members:
   :no-index:

.. automodule:: pulserver.recon.mrd2dicom
   :members:
   :no-index:

.. automodule:: pulserver.recon.dicom2mrd
   :members:
   :no-index:

.. automodule:: pulserver.recon.replay
   :members:
   :no-index:

.. automodule:: pulserver.recon.concurrency
   :members:
   :no-index:

.. automodule:: pulserver.recon.constants
   :members:
   :no-index:

.. automodule:: pulserver.recon.handlers
   :members:
   :no-index:

.. automodule:: pulserver.recon.handlers.simplefft
   :members:
   :no-index:

.. automodule:: pulserver.recon.handlers.fftrecon
   :members:
   :no-index:

.. automodule:: pulserver.recon.handlers.savedataonly
   :members:
   :no-index:
```
