# `pulserver.reczoo`

The reconstruction each {doc}`seqzoo <seqzoo>` sequence is designed for, under
the same module name. A module holds one `pulserver.ReconPlugin` subclass and the
module-level `PLUGIN` instance the inline runtime discovers, so the same code
runs offline on arrays and online on an MRD stream.

```python
from pulserver import AcquisitionBucket, ReconContext
from pulserver.reczoo import gre_2d

bucket = AcquisitionBucket.from_arrays(
    kspace, labels={"kspace_encode_step_1": lines}
)
image = gre_2d.PLUGIN(bucket, ReconContext.offline(header)).data
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
   :toctree: generated/reczoo
   :nosignatures:

   pulserver.reczoo.bssfp_2d
   pulserver.reczoo.bssfp_3d
   pulserver.reczoo.fse_2d
   pulserver.reczoo.fse_3d
   pulserver.reczoo.gre_2d
   pulserver.reczoo.gre_3d
   pulserver.reczoo.gre_multiecho_2d
   pulserver.reczoo.gre_multiecho_3d
   pulserver.reczoo.inline_fft
   pulserver.reczoo.mprage_3d
   pulserver.reczoo.se_2d
   pulserver.reczoo.se_3d
   pulserver.reczoo.sms_epi
   pulserver.reczoo.subspace_basis
```
