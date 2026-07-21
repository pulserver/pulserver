# `pulserver`

The root namespace is the **plugin contract**: the base class a sequence
plugin subclasses, the typed parameters its scanner UI is built from, the
helpers that serialise and validate a protocol, and the offline CLI.

It deliberately exports *no* waveform-authoring helpers. RF pulses, gradients,
readout modules, sampling patterns and phase schedules live in
{doc}`pulserver.pypulseq <pypulseq>` and are importable only from there, so a
plugin's two halves stay visibly separate:

```python
import pulserver.pypulseq as pp           # waveforms and events
from pulserver import Sequence, UIParam   # plugin contract
```

## Plugin contracts

The two base classes a plugin builds on, plus the entry point that turns one
into a command-line tool. `Sequence` (alias `PulseqSequence`) declares the
default protocol and synthesises the `.seq` file; `Module` is a reusable,
stateful fragment — an RF pulse or a readout train — that is re-parameterised
and appended shot after shot.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.Sequence
   pulserver.PulseqSequence
   pulserver.Module
   pulserver.run_cli
```

## Protocol parameter types

One class per scanner UI control. Each carries the value, its display unit,
and the bounds the interpreter enforces, so a single object drives both the UI
and the offline default protocol.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.TypeinFloatParam
   pulserver.DropdownFloatParam
   pulserver.TypeinIntParam
   pulserver.DropdownIntParam
   pulserver.BoolParam
   pulserver.StringListParam
   pulserver.Description
```

## Protocol keys and enumerations

`UIParam` is the canonical key namespace every protocol is written against.
The `*Key` groups record which value type each key expects; the remaining
enumerations are the fixed option sets behind dropdown controls.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.UIParam
   pulserver.Validate
   pulserver.ParamKind
   pulserver.InputMode
   pulserver.FloatKey
   pulserver.IntKey
   pulserver.BoolKey
   pulserver.EnumKey
   pulserver.SequenceType
   pulserver.ImagingMode
   pulserver.PreparationType
   pulserver.TriggerType
```

`Protocol` and `ProtocolValue` are the mapping and value type aliases used by
the helpers below.

## Protocol handling

Validation, and round-tripping between the typed parameter objects above and
the plain-dictionary form exchanged with the interpreter and the offline CLI.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.expected_param_kind
   pulserver.enum_options
   pulserver.make_enum_param
   pulserver.validate_protocol_entry
   pulserver.validate_protocol
   pulserver.param_to_dict
   pulserver.dict_to_param
   pulserver.protocol_to_dict
   pulserver.dict_to_protocol
   pulserver.set_protocol_value
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
