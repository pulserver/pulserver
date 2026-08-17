# `pulserver.app`

One complete, self-contained sequence per module. Each is a worked example of
the whole authoring stack — the {doc}`design <design>` modules it composes, the
encoding plan it builds from {doc}`pypulseq <pypulseq>`, and the loop that
writes the blocks — and each carries two entry points over one implementation.

`main` takes explicit keyword controls and returns a
`pulserver.pypulseq.Sequence`. This is the whole sequence, written in the style
of a PyPulseq example script, and it is what to read, copy and edit.

```python
from pulserver.app import gre2D_sequence

seq = gre2D_sequence.main(n_x=128, n_y=128, n_slices=5, acceleration=2, n_acs=24)
seq.write("gre_2d.seq")
```

`PLUGIN` is the same sequence behind the {doc}`plugin contract <pulserver>`,
so the bridge can offer it in the scanner UI. Running the module as a script
builds a `.seq` offline from the same controls:

```bash
python -m pulserver.app.sequence.gre2D_sequence --nx 128 --ny 128 --ry 2 -o gre2D_sequence.seq
```

{doc}`pulserver.app <app_recon>` holds the reconstruction that matches each
sequence, under the same module name.

## Coverage

Every slot earns its place by the interpreter feature it stresses; a
sequence that stressed nothing new would be bloat.

| module | what it stresses |
| --- | --- |
| `gre2D_sequence` | the exemplar: passes, ACS-first calibration, partial echo/Fourier, `auto_label` |
| `gre3D_sequence` | two-axis phase encoding, the ACS rectangle leading the traversal |
| `se2D_sequence`, `se3D_sequence` | the refocusing k-flip in its simplest setting, TE split around the 180 |
| `gre_multiecho2D_sequence`, `gre_multiecho3D_sequence` | echo trains, monopolar/bipolar polarity, per-echo `ECO` |
| `bssfp2D_sequence`, `bssfp3D_sequence` | balance and TE = TR/2 solved, the half-flip catalyst, phase alternation |
| `fse2D_sequence` | CPMG with per-echo trapezoid encodes, rolled-linear effective TE |
| `fse3D_sequence` | Busse-style view orderings (linear/radial/radial-adaptive/shuffling), TRAPS variable flips |
| `mprage3D_sequence` | inversion segments, TI at the centre view, the same orderings |
| `gre_radial2D_sequence`, `gre_spiral2D_sequence` | `ROTATIONS` extensions: one waveform, any number of shots |
| `gre_stack_of_stars3D_sequence`, `gre_stack_of_spirals3D_sequence` | rotation in-plane plus a scaled partition encode |
| `se_propeller2D_sequence` | a spin echo per rotated blade, every blade through the centre |
| `mprage_stack_of_spirals3D_sequence` | **explicit** golden-angle arms: one waveform per shot, the untemplatable path |
| `epi2D_sequence`, `epi3D_sequence` | PNS and mechanical resonance; blips on the ramps; the `NextSequence` collection export |
| `zte3D_sequence` | continuous gradients with no dead time for SSP placement; the declared missing centre |

FOV positioning runs through `TransformFOV` in every slot: the Cartesian
modules bake scalar offsets, and the rotated ones defer their ADC share to
the consumer of the base trajectory -- which is the mechanism the
non-Cartesian half of the table exists to demonstrate.

## Sequences

```{eval-rst}
.. autosummary::
   :toctree: generated/app_sequence
   :nosignatures:

   pulserver.app.sequence.bssfp2D_sequence
   pulserver.app.sequence.bssfp3D_sequence
   pulserver.app.sequence.epi2D_sequence
   pulserver.app.sequence.epi3D_sequence
   pulserver.app.sequence.fse2D_sequence
   pulserver.app.sequence.fse3D_sequence
   pulserver.app.sequence.gre2D_sequence
   pulserver.app.sequence.gre3D_sequence
   pulserver.app.sequence.gre_multiecho2D_sequence
   pulserver.app.sequence.gre_multiecho3D_sequence
   pulserver.app.sequence.gre_radial2D_sequence
   pulserver.app.sequence.gre_spiral2D_sequence
   pulserver.app.sequence.gre_stack_of_spirals3D_sequence
   pulserver.app.sequence.gre_stack_of_stars3D_sequence
   pulserver.app.sequence.mprage3D_sequence
   pulserver.app.sequence.mprage_stack_of_spirals3D_sequence
   pulserver.app.sequence.se2D_sequence
   pulserver.app.sequence.se_propeller2D_sequence
   pulserver.app.sequence.se3D_sequence
```
