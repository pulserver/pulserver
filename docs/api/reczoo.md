# `pulserver.reczoo`

The reconstruction each {doc}`seqzoo <seqzoo>` sequence is designed for, under
the same module name. A module holds one `pulserver.ReconApp` subclass and the
module-level `PLUGIN` instance the inline runtime discovers, so the same code
runs offline on arrays and online on an MRD stream.

```python
import numpy as np
from pulserver import AcquisitionBucket, ReconContext
from pulserver.reczoo import gre_2d

bucket = AcquisitionBucket.from_arrays(
    kspace, labels={"kspace_encode_step_1": lines, "center_sample": centers}
)
image = gre_2d.PLUGIN(bucket, ReconContext.offline()).data
```

A paired reconstruction reads what its sequence encoded — its counters, its
calibration flags, the echo position it recorded — rather than being told
separately. The remaining modules are single-purpose examples of one
reconstruction primitive.

Install `pulserver[recon-cpu]` for the numerical stack the model-based
branches need.

## Reconstructions

```{eval-rst}
.. autosummary::
   :toctree: generated/reczoo
   :nosignatures:

   pulserver.reczoo.gre_2d
   pulserver.reczoo.inline_fft
   pulserver.reczoo.sms_epi
   pulserver.reczoo.subspace_basis
```
