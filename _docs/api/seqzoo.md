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

## Coverage

Every slot earns its place by the interpreter feature it stresses; a
sequence that stressed nothing new would be bloat.

| module | what it stresses |
| --- | --- |
| `gre_2d` | the exemplar: passes, ACS-first calibration, partial echo/Fourier, `auto_label` |
| `gre_3d` | two-axis phase encoding, the ACS rectangle leading the traversal |
| `se_2d`, `se_3d` | the refocusing k-flip in its simplest setting, TE split around the 180 |
| `gre_multiecho_2d`, `gre_multiecho_3d` | echo trains, monopolar/bipolar polarity, per-echo `ECO` |
| `bssfp_2d`, `bssfp_3d` | balance and TE = TR/2 solved, the half-flip catalyst, phase alternation |
| `fse_2d` | CPMG with per-echo trapezoid encodes, rolled-linear effective TE |
| `fse_3d` | Busse-style view orderings (linear/radial/radial-adaptive/shuffling), TRAPS variable flips |
| `mprage_3d` | inversion segments, TI at the centre view, the same orderings |
| `gre_radial_2d`, `gre_spiral_2d` | `ROTATIONS` extensions: one waveform, any number of shots |
| `gre_stack_of_stars_3d`, `gre_stack_of_spirals_3d` | rotation in-plane plus a scaled partition encode |
| `se_propeller_2d` | a spin echo per rotated blade, every blade through the centre |
| `mprage_stack_of_spirals_3d` | **explicit** golden-angle arms: one waveform per shot, the untemplatable path |
| `epi_2d`, `epi_3d` | PNS and mechanical resonance; blips on the ramps; the `NextSequence` collection export |
| `zte_3d` | continuous gradients with no dead time for SSP placement; the declared missing centre |

FOV positioning runs through `TransformFOV` in every slot: the Cartesian
modules bake scalar offsets, and the rotated ones defer their ADC share to
the consumer of the base trajectory -- which is the mechanism the
non-Cartesian half of the table exists to demonstrate.

## Sequences

```{eval-rst}
.. autosummary::
   :toctree: generated/seqzoo
   :nosignatures:

   pulserver.seqzoo.bssfp_2d
   pulserver.seqzoo.bssfp_3d
   pulserver.seqzoo.epi_2d
   pulserver.seqzoo.epi_3d
   pulserver.seqzoo.fse_2d
   pulserver.seqzoo.fse_3d
   pulserver.seqzoo.gre_2d
   pulserver.seqzoo.gre_3d
   pulserver.seqzoo.gre_multiecho_2d
   pulserver.seqzoo.gre_multiecho_3d
   pulserver.seqzoo.gre_radial_2d
   pulserver.seqzoo.gre_spiral_2d
   pulserver.seqzoo.gre_stack_of_spirals_3d
   pulserver.seqzoo.gre_stack_of_stars_3d
   pulserver.seqzoo.mprage_3d
   pulserver.seqzoo.mprage_stack_of_spirals_3d
   pulserver.seqzoo.se_2d
   pulserver.seqzoo.se_propeller_2d
   pulserver.seqzoo.se_3d
```
