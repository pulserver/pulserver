# `pulserver.seqzoo`

One complete, self-contained sequence per module. Each is a worked example of
the whole authoring stack — the {doc}`design <design>` modules it composes, the
encoding plan it builds from {doc}`pypulseq <pypulseq>`, and the loop that
writes the blocks — and each carries two entry points over one implementation.

`main` takes explicit keyword controls and returns a
`pulserver.pypulseq.Sequence`. This is the whole sequence, written in the style
of a PyPulseq example script, and it is what to read, copy and edit.

```python
from pulserver.seqzoo import gre_2d

seq = gre_2d.main(n_x=128, n_y=128, n_slices=5, acceleration=2, n_acs=24)
seq.write("gre_2d.seq")
```

`PLUGIN` is the same sequence behind the {doc}`plugin contract <pulserver>`,
so the bridge can offer it in the scanner UI. Running the module as a script
builds a `.seq` offline from the same controls:

```bash
python -m pulserver.seqzoo.gre_2d --nx 128 --ny 128 --ry 2 -o gre_2d.seq
```

{doc}`pulserver.reczoo <reczoo>` holds the reconstruction that matches each
sequence, under the same module name.

## Sequences

```{eval-rst}
.. autosummary::
   :toctree: generated/seqzoo
   :nosignatures:

   pulserver.seqzoo.bssfp_2d
   pulserver.seqzoo.bssfp_3d
   pulserver.seqzoo.fse_2d
   pulserver.seqzoo.fse_3d
   pulserver.seqzoo.gre_2d
   pulserver.seqzoo.gre_3d
   pulserver.seqzoo.gre_multiecho_2d
   pulserver.seqzoo.gre_multiecho_3d
   pulserver.seqzoo.mprage_3d
   pulserver.seqzoo.se_2d
   pulserver.seqzoo.se_3d
```
