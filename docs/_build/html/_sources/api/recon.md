# `pulserver.recon`

`pulserver.recon` combines the Gadgetron-compatible MRD server with a
Torch-first integration layer. It does not implement imaging operators,
proximal maps, or solvers: Cartesian/non-Cartesian MRPro operators, DeepInverse
physics/priors/optimisers, and PyGROG NLINV are selected and composed directly.
Torch owns CPU/CUDA dispatch, so tensors remain on the caller-selected device.
mri-nufft remains available with FINUFFT or CUFINUFFT for its established
backend-specific operators.

Install `pulserver[recon]` for the server, DICOM, and MRD layers.
Install `pulserver[recon-cpu]` for FINUFFT, MRPro, and DeepInverse; add
`pulserver[recon-nlinv]` for PyGROG NLINV calibration or install
`pulserver[recon-cuda]` for CUFINUFFT on Linux CUDA hosts.
Install `pulserver[recon-distortion]` only when PyHySCO reverse-polarity
distortion correction is required. PyHySCO is GPL-3.0-only, so Pulserver does
not vendor or import it.

```python
from pulserver.recon import deepinverse_multicoil_mri, deepinverse_prior

physics = deepinverse_multicoil_mri(mask=mask, coil_maps=coil_maps)
image = physics.A_dagger(kspace)  # tensors remain on coil_maps.device
prior = deepinverse_prior("tv")
```

For MRD-native Cartesian and non-Cartesian reconstructions, obtain the exact
MRPro operator class with `mrpro_operator("fourier")` and use MRPro's
reconstruction algorithms on `KData`. For non-Cartesian mri-nufft workflows,
`mrinufft_operator("finufft", ...)` and `mrinufft_operator("cufinufft", ...)`
return the maintained backend operators.

## Reconstruction primitives

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.linops.deepinverse_multicoil_mri
   pulserver.recon.linops.mrpro_operator
   pulserver.recon.linops.mrinufft_operator
   pulserver.recon.linops.available_nufft_backends
   pulserver.recon.prox.deepinverse_prior
   pulserver.recon.optimizers.mrpro_optimizer
   pulserver.recon.optimizers.deepinverse_optimizer
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
standard MRD flags and labels into ordered groups; it deliberately does not
reimplement odd/even phase fitting, Slice-GRAPPA, or distortion estimation.
Pass the grouped tensors to an EPI-capable upstream backend and use the known
CAIPI encoding for SMS.

FSE uses its CPMG RF/receiver phase relation and does **not** need EPI's
alternating-readout navigator or an opposite-PE distortion pair. A
phase-sensitive FSE application may still acquire its own phase reference.

```{eval-rst}
.. autosummary::
   :toctree: generated/recon
   :nosignatures:

   pulserver.recon.epi.EpiAcquisitionGroups
   pulserver.recon.epi.partition_epi_acquisitions
   pulserver.recon.sms.SmsEpiInputs
   pulserver.recon.distortion.run_pyhysco
```

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
