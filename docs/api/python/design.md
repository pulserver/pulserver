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
