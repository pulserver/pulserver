# pulserver.app

The shipped plugins: one complete, self-contained module each, and the
contracts they are written against. A sequence module composes
{doc}`design` modules into an encoding plan and loops over it writing blocks; a
reconstruction module takes what the scanner sends back and turns it into
images.

Every module is callable, and calling it does the module's job:

```python
from pulserver.app import gre3D_sequence, cartesian3D_recon

seq = gre3D_sequence(n_x=128, n_y=128, n_z=64, slab_thickness=0.128)
seq.write("gre3D.seq")

images = cartesian3D_recon("scan.h5", virtual_coils=8)
```

That call is the module's `main`, and it is what is documented below: a sequence
writes one, in the style of a PyPulseq example script — what to read, copy and
edit — and a reconstruction has one built from the settings its plugin takes, so
the signature answers for the call either way.

Beside it, `PLUGIN` is the same thing behind the scanner contract: a
`SequencePlugin`, so the bridge can offer the sequence in the UI and running the
module as a script writes a `.seq` offline, or a `ReconPlugin`, so the
reconstruction can be driven over a live stream. A file holds what the scanner
sent, so calling a reconstruction streams it to that same plugin in this
process — there is no second code path.

One flat namespace: every plugin is `pulserver.app.<name>`. The grouping below —
gradient echo, spin echo, Cartesian, EPI — is a way of reading the zoo rather
than a way of importing from it.

```{eval-rst}
.. currentmodule:: pulserver.app
```

## Sequences

One module per sequence family. These are the validation corpus as well as the
worked examples: the zoo tests hold their timing, their structure and their
safety verdicts, so a change that moves any of them has to say so.

### Gradient echo

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_sequence
   :template: autosummary/plugin.rst

   gre2D_sequence
   gre3D_sequence
   gre_multiecho2D_sequence
   gre_multiecho3D_sequence
   bssfp2D_sequence
   bssfp3D_sequence
```

### Spin echo

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_sequence
   :template: autosummary/plugin.rst

   se2D_sequence
   se3D_sequence
   fse2D_sequence
   fse3D_sequence
   se_propeller2D_sequence
```

### Prepared

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_sequence
   :template: autosummary/plugin.rst

   mprage3D_sequence
   mprage_stack_of_spirals3D_sequence
```

### Echo planar

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_sequence
   :template: autosummary/plugin.rst

   epi2D_sequence
   epi3D_sequence
```

### Non-Cartesian

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_sequence
   :template: autosummary/plugin.rst

   gre_radial2D_sequence
   gre_spiral2D_sequence
   gre_stack_of_stars3D_sequence
   gre_stack_of_spirals3D_sequence
   zte3D_sequence
```

## Reconstructions

One per **sampling**, deliberately not one per sequence: what a reconstruction
has to know is how k-space was covered, not which contrast the sequence was
after. A spin echo, a gradient echo, an MPRAGE and a balanced SSFP all leave one
Cartesian grid, so one plugin serves them all, and a radial, spiral or PROPELLER
scan differs only in the trajectory its acquisitions carry.

Every plugin here is a declaration and one hook. The per-acquisition steps go
in its `chain` — `NoiseAdjust`, `CoilCompression`, `EpiPhaseCorrection`,
`RampSampling`, all of them names in {doc}`recon` — and the boundaries worth
reconstructing at go in its `branches`. What is left to write is `recon`, over
buffers that are already filled, so a plugin reads as the composition it is.

A noise scan measures the receiver, not the object, so it never reaches a
buffer: the chain keeps it and whitens every readout that follows against it.
Coil compression is the same idea one step later, and where it can happen
depends on what the sequence acquired. A scan whose calibration is a separate
prescan — both EPI plugins — reads the array's principal channels off it when
that prescan closes and leaves the basis in `context.exam`; every imaging
readout is then compressed as it arrives. A scan whose calibration is imaging
data on the imaging grid has no such moment, so those plugins buffer the full
array and compress once, on the way into the solve.

### Cartesian

Which of three reconstructions runs is read off the sampling mask rather than
declared, by {func}`pulserver.recon.cartesian_recon`: the coil-wise adjoint when
everything is there, POCS when the readout is truncated, CG-SENSE against NLINV
maps when phase encodes are missing.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_recon
   :template: autosummary/plugin.rst

   cartesian2D_recon
   cartesian3D_recon
```

### Non-Cartesian

Density compensation, NLINV sensitivities calibrated from the samples inside the
calibration radius, and a CG-SENSE solve against the trajectory the acquisitions
carry. `noncartesian_stack_recon` is the factorised case: the partition axis is
Cartesian, so an inverse FFT along z turns the volume into independent planes.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_recon
   :template: autosummary/plugin.rst

   noncartesian2D_recon
   noncartesian3D_recon
   noncartesian_stack_recon
```

### EPI

The one family that corrects before it places: a reversed line has to be flipped
and phase corrected against the fit its navigator triplet produced, and a
ramp-sampled one resampled onto the grid, before either belongs on it — which is
what those two gadgets do as each line arrives. The calibration prescan is a
subsequence, so it is an encoding space of its own and never touches the imaging
grid; it is also the one case whose routing needs two flags at once, so these
two plugins write a `receive` where the others declare one.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_recon
   :template: autosummary/plugin.rst

   epi2D_recon
   epi3D_recon
```

### Prospective motion correction

The one reconstruction that answers while the scan is still running. It takes
the navigator readouts the sequence interleaves with its encoding, measures
where the head has moved to, and sends the correction back over the real-time
port for the readouts not yet played. It emits no image: what it produces is a
pose, and the images it makes of the navigator planes exist only to measure
one.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_recon
   :template: autosummary/plugin.rst

   pmc_recon
```

## The contract behind them

A sequence plugin declares what the scanner UI shows and builds the sequence
the console asks for; a reconstruction plugin receives the acquisitions that
come back. Both contracts, and the typed protocol parameters a console builds
its controls from, are documented where they live: {doc}`design` for the
sequence side, {doc}`recon` for the reconstruction side.

## See also

{doc}`design` is the toolbox the sequences are assembled from, {doc}`pypulseq`
the event layer under it, and {doc}`recon` the toolbox the reconstructions are
written against.
