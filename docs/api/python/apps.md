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

images = cartesian3D_recon("scan.h5")
```

Each carries two entry points over one implementation. `main(...)` is the whole
design under explicit keyword controls, written in the style of a PyPulseq
example script — what to read, copy and edit. `PLUGIN` is the same thing behind
the scanner protocol contract, so the bridge can offer it in the UI and running
the module as a script does the job offline.

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

   mprage3D_sequence
   mprage_stack_of_spirals3D_sequence
```

### Echo planar

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_sequence

   epi2D_sequence
   epi3D_sequence
```

### Non-Cartesian

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_sequence

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

Every plugin here is three hooks and nothing else. `startup` lays out the
buffers the header's encoding spaces describe, `receive` places each acquisition
and routes the boundaries it closes to a named branch, and `recon` holds the
reconstruction of each branch over buffers that are already filled. There is no
local helper between them: what a step needs is a name in {doc}`recon`, so a
plugin reads as the composition it is.

A noise scan measures the receiver, not the object, so it never reaches a
buffer: `receive` keeps it and whitens every readout that follows against it.
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

   noncartesian2D_recon
   noncartesian3D_recon
   noncartesian_stack_recon
```

### EPI

The one family that corrects before it places: a reversed line has to be flipped
and phase corrected against the fit its navigator triplet produced before it
belongs on the grid, so `receive` does that as each line arrives. The
calibration prescan is a subsequence, so it is an encoding space of its own and
never touches the imaging grid.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_recon

   epi2D_recon
   epi3D_recon
```

## The plugin contracts

What a plugin subclasses, and what the runtime hands it.

```{eval-rst}
.. currentmodule:: pulserver

.. autosummary::
   :toctree: ../generated/app
   :template: autosummary/class.rst

   SequencePlugin
   SequenceModule
```

```{eval-rst}
.. currentmodule:: pulserver.app

.. autosummary::
   :toctree: ../generated/app
   :template: autosummary/class.rst

   PluginModule
```

{class}`pulserver.ReconPlugin`, the reconstruction half of the contract, is
documented with the toolbox it drives, in {doc}`recon`.

```{eval-rst}
.. currentmodule:: pulserver
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app

   run_cli
   main_kwargs
   write_sequence
```

## Protocol

What the scanner UI shows, declared as typed parameters rather than as a form.
A plugin publishes a `Protocol` — the alias for a mapping from a `UIParam` key
to a `ProtocolValue`, one of the parameter kinds below — and the bridge builds
the UI from it, validates what comes back against it, and hands the plugin the
values.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app
   :template: autosummary/class.rst

   UIParam
   ParamKind
   Description
   Validate
```

### Parameter kinds

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app
   :template: autosummary/class.rst

   TypeinFloatParam
   TypeinIntParam
   DropdownFloatParam
   DropdownIntParam
   OffFloatParam
   OffIntParam
   BoolParam
   StringListParam
```

### Enumerations

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app
   :template: autosummary/class.rst

   ImagingMode
   InputMode
   PreparationType
   SequenceType
   TriggerType
   TEPreset
   TRPreset
   BoolKey
   EnumKey
   FloatKey
   IntKey
```

### Reading and writing one

`params` is the accessor module: index by canonical key, unwrap the parameter
object and coerce its value, once, with the type the key declares.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app

   params
   make_enum_param
   dict_to_protocol
   protocol_to_dict
   validate_protocol
```

## See also

{doc}`design` is the toolbox the sequences are assembled from, {doc}`pypulseq`
the event layer under it, and {doc}`recon` the toolbox the reconstructions are
written against.
