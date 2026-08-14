# `pulserver`

The root namespace is the **plugin contract**: the base class a sequence
plugin subclasses, the abstract types it exchanges with the authoring
namespace, the typed parameters its scanner UI is built from, and the offline
CLI.

It deliberately exports *no* authoring helpers. Those are split across two
namespaces by role, so a plugin's parts stay visibly separate:

```python
import pulserver.pypulseq as pp             # events, Sequence, Opts
import pulserver.design as design           # modules and scan loops
from pulserver import SequencePlugin        # sequence plugin contract
from pulserver import ReconPlugin           # reconstruction plugin contract
```

{doc}`pulserver.pypulseq <pypulseq>` is the event layer: upstream PyPulseq
re-exported whole, plus Pulserver's replacements for a few of its objects.
{doc}`pulserver.design <design>` is the toolbox above it — every
`SequenceModule` Pulserver ships.

Complete worked sequences and the reconstructions that match them are two
further namespaces: {doc}`pulserver.seqzoo <seqzoo>` and
{doc}`pulserver.reczoo <reczoo>`.  This root namespace holds only the contracts
they are written against.

## Plugin contracts

The base classes a plugin builds on, plus the entry point that turns one into a
command-line tool. `SequencePlugin` declares the default protocol and
synthesises the `.seq` file; `ReconPlugin` is its counterpart on the
reconstruction side.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.SequencePlugin
   pulserver.ReconPlugin
   pulserver.run_cli
```

## Abstract types

The type every `pulserver.design` class is, named here because it is part of
the contract rather than of any one waveform family.

`SequenceModule` answers one question — *what do I play?* — and its blocks
feed `seq.add_block(*block)`. It owns a design: the gradients solved, the TE
and TR budgeted, the ADC landed on both time rasters, and the resulting events
published under the names its constructor gave them.

It owns nothing else. A module does not iterate, holds no per-shot state, and
never sees a sampling pattern. How frames, slices, contrasts and shots nest is
plain `for` statements in the plugin, which is what keeps a preparation, a
trigger or a dummy TR insertable at any level — and per-shot variation is
ordinary PyPulseq: a phase is an attribute write, an encode is `scale_grad`,
an orientation is a rotation event.

The tables a loop indexes with — masks, view orderings, projection angles, RF
phase schedules — are plain arrays and live in {doc}`pulserver.pypulseq
<pypulseq>`.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.SequenceModule
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
