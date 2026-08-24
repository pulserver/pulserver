# pulserver.design

The building blocks a sequence is assembled from. Every entry is a
`SequenceModule`: it is given a system and its physical parameters, solves
its own timing against the gradient and slew limits, and publishes the events
the acquisition loop plays. Modules do not know about each other — a readout
is handed the excitation's pulse, not the excitation — so any excitation
pairs with any readout.

```{eval-rst}
.. currentmodule:: pulserver.design
```

## Excitation

Pulses that create transverse magnetization, and the slice or slab selection
that goes with them. All of them publish `rf`, and the selective ones also
`gz` and its rephaser.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/design
   :template: autosummary/class.rst

   NonSelectiveExcitation
   SpatialSelectiveExcitation
   SpatialSelective2DExcitation
   FrequencySelectiveExcitation
   SpspExcitation
   MultibandExcitation
   SmsExcitation
```

## Refocusing

Pulses that reverse dephasing, for spin-echo families.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/design
   :template: autosummary/class.rst

   NonSelectiveRefocusing
   SpatialSelectiveRefocusing
```

## Preparation

Modules played before the readout to give the acquisition its contrast. Each
leaves the magnetization in a stated condition and reports the time it took
to get there, which is what a loop needs to place its TI or TE.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/design
   :template: autosummary/class.rst

   Inversion
   InversionPreparation
   T2Preparation
   T1T2Preparation
   DiffusionPreparation
   BlochSiegertPreparation
   MtPreparation
   IhMtPreparation
   FatSaturation
   OffResonanceSaturation
```

## Cartesian readouts

Readouts that sample on a rectilinear grid: one line per repetition, or a
train of lines per excitation.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/design
   :template: autosummary/class.rst

   LineReadout2D
   LineReadout3D
   BssfpReadout2D
   BssfpReadout3D
   FseReadout2D
   FseReadout3D
   EpiReadout2D
   EpiReadout3D
```

## Non-Cartesian readouts

Readouts whose trajectory is a solved waveform, oriented by rotating the
blocks. Each family comes in three geometries: a plane (`2D`), a stack of
planes with partition encoding (`Stack`), and a set of orientations covering
a sphere (`Projection`).

```{eval-rst}
.. autosummary::
   :toctree: ../generated/design
   :template: autosummary/class.rst

   RadialReadout2D
   RadialStackReadout
   RadialProjectionReadout
   SpiralReadout2D
   SpiralStackReadout
   SpiralProjectionReadout
   RosetteReadout2D
   RosetteStackReadout
   RosetteProjectionReadout
   PropellerReadout2D
   PropellerStackReadout
   ZteReadout
   NonCartesianReadout
```

`NonCartesianReadout` is the base a new family subclasses: design a
trajectory, hand it over, and the bracket alignment, TE and TR budget,
spoiler and explicit-rotation path are inherited. See
{doc}`../../examples/python/new_readout`.

## Base

```{eval-rst}
.. autosummary::
   :toctree: ../generated/design
   :template: autosummary/class.rst

   RfModule
```

## The scanner protocol

A sequence reaches a scanner as a *plugin*: a class the console can ask for
its default protocol, hand a filled-in one back, and run. The protocol itself
is a mapping from {class}`UIParam` to typed parameters — each carrying the
value, its unit, and what the user interface may do with it — so the console
can build its own controls without knowing the sequence.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/design
   :template: autosummary/class.rst

   SequencePlugin
   SequenceModule
```

The parameter kinds. A *typein* accepts any value in range; a *dropdown*
offers a fixed set; an *off* variant adds a disabled state; the presets carry
the echo and repetition times a protocol may offer.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/design
   :template: autosummary/class.rst

   UIParam
   TypeinFloatParam
   DropdownFloatParam
   TypeinIntParam
   DropdownIntParam
   OffFloatParam
   OffIntParam
   BoolParam
   StringListParam
   Description
   TEPreset
   TRPreset
```

The option enumerations a dropdown draws on.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/design
   :template: autosummary/class.rst

   SequenceType
   ImagingMode
   PreparationType
   TriggerType
```

Reading a protocol, writing one, and running a plugin from a shell.
{mod}`~pulserver.design.params` holds the readers a sequence uses to pull
typed values out of the protocol it was handed.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/design

   params
   make_enum_param
   protocol_to_dict
   dict_to_protocol
   main_kwargs
   run_cli
   write_sequence
```
