# `pulserver`

The root namespace is the **plugin contract**: the base class a sequence
plugin subclasses, the abstract types it exchanges with the authoring
namespace, the typed parameters its scanner UI is built from, and the offline
CLI.

It deliberately exports *no* waveform-authoring helpers. RF pulses, gradients,
readout factories, sampling factories and phase schedules live in
{doc}`pulserver.pypulseq <pypulseq>` and are importable only from there, so a
plugin's two halves stay visibly separate:

```python
import pulserver.pypulseq as pp           # waveforms and events
from pulserver import Sequence, UIParam   # plugin contract
```

## Plugin contracts

The base class a plugin builds on, plus the entry point that turns one into a
command-line tool. `Sequence` (alias `PulseqSequence`) declares the default
protocol and synthesises the `.seq` file.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.PulseqSequence
   pulserver.Sequence
   pulserver.run_cli
```

## Abstract types

The things every `pulserver.pypulseq` factory returns, named here because they
are part of the contract rather than of any one waveform family. A
`SequenceModule` is a reusable, stateful fragment — an RF pulse, a
preparation, a readout shot — designed once and re-parameterised per shot;
`AcquisitionPlan` is the high-level frame/slice/shot loop and yields one
`Acquisition` at a time; `SamplingPattern` is its lower-level k-space support
and ordering; `SliceSampling` is the independent physical slice/SMS schedule;
`SliceGroup` is one excitation's worth of slices.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.Acquisition
   pulserver.AcquisitionPlan
   pulserver.SamplingPattern
   pulserver.SequenceModule
   pulserver.SliceGroup
   pulserver.SliceSampling
```

## Protocol parameter types

One class per scanner UI control. Each carries the value, its display unit,
and the bounds the interpreter enforces, so a single object drives both the UI
and the offline default protocol.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.BoolParam
   pulserver.Description
   pulserver.DropdownFloatParam
   pulserver.DropdownIntParam
   pulserver.StringListParam
   pulserver.TypeinFloatParam
   pulserver.TypeinIntParam
```

## Protocol keys and enumerations

`UIParam` is the canonical key namespace every protocol is written against.
The `*Key` groups record which value type each key expects; the remaining
enumerations are the fixed option sets behind dropdown controls.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.BoolKey
   pulserver.EnumKey
   pulserver.FloatKey
   pulserver.ImagingMode
   pulserver.InputMode
   pulserver.IntKey
   pulserver.ParamKind
   pulserver.PreparationType
   pulserver.SequenceType
   pulserver.TriggerType
   pulserver.UIParam
   pulserver.Validate
```

`Protocol` and `ProtocolValue` are the mapping and value type aliases used by
the helpers below.

## Protocol handling

The four calls a plugin actually makes: validate a protocol, round-trip it
between typed parameter objects and the plain-dictionary form exchanged with
the interpreter, and build a dropdown from an enumeration. The per-entry and
per-parameter primitives underneath them are implementation detail and stay in
`pulserver._core`.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.dict_to_protocol
   pulserver.make_enum_param
   pulserver.protocol_to_dict
   pulserver.validate_protocol
```

## Protocol accessors

`pulserver.params` reads and writes protocol values by canonical key and
resolves derived quantities — phase FOV, ACS size, readout/phase axes — that
plugins would otherwise each recompute.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.params
```
