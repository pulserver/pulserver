# `pulserver.app`

The reconstruction each {doc}`app.sequence <app_sequence>` sequence is designed for, under
the same module name. A module holds one `pulserver.ReconPlugin` subclass and the
module-level `PLUGIN` instance the inline runtime discovers, so the same code
runs offline on arrays and online on an MRD stream.

```python
from pulserver import AcquisitionBucket, ReconContext
from pulserver.app import gre2D_recon

bucket = AcquisitionBucket.from_arrays(
    kspace, labels={"kspace_encode_step_1": lines}
)
image = gre2D_recon.PLUGIN(bucket, ReconContext.offline(header)).data
```

Calling the plugin replays the bucket through the lifecycle the inline runtime
drives, so the offline result is the streamed one. The MRD header comes along
because that is what sizes the buffers a plugin allocates in `startup`.

A paired reconstruction reads what its sequence encoded — its counters and the
flags marking where its calibration block and its slices end — rather than
being told separately. The remaining modules are single-purpose examples of one
reconstruction primitive.

Install `pulserver[recon-cpu]` for the numerical stack the model-based
branches need.

## Reconstructions

```{eval-rst}
.. autosummary::
   :toctree: generated/app_recon
   :nosignatures:

   pulserver.app.recon.bssfp2D_recon
   pulserver.app.recon.bssfp3D_recon
   pulserver.app.recon.epi2D_recon
   pulserver.app.recon.epi3D_recon
   pulserver.app.recon.fse2D_recon
   pulserver.app.recon.fse3D_recon
   pulserver.app.recon.gre2D_recon
   pulserver.app.recon.gre3D_recon
   pulserver.app.recon.gre_multiecho2D_recon
   pulserver.app.recon.gre_multiecho3D_recon
   pulserver.app.recon.gre_radial2D_recon
   pulserver.app.recon.gre_spiral2D_recon
   pulserver.app.recon.gre_stack_of_spirals3D_recon
   pulserver.app.recon.gre_stack_of_stars3D_recon
   pulserver.app.inline_fft
   pulserver.app.recon.mprage3D_recon
   pulserver.app.recon.mprage_stack_of_spirals3D_recon
   pulserver.app.recon.se2D_recon
   pulserver.app.recon.se_propeller2D_recon
   pulserver.app.recon.se3D_recon
   pulserver.app.sms_epi
   pulserver.app.recon.subspace_basis_recon
```
