# `pulserver`

The root namespace is the **plugin contract**: the base class a sequence
plugin subclasses, the abstract types it exchanges with the authoring
namespace, the typed parameters its scanner UI is built from, and the offline
CLI.

It deliberately exports *no* authoring helpers. Those are split across two
namespaces by role, so a plugin's parts stay visibly separate:

```python
import pulserver.pypulseq as pp           # events, Sequence, Opts
import pulserver.design as design         # modules and scan loops
from pulserver import Sequence, UIParam   # plugin contract
```

{doc}`pulserver.pypulseq <pypulseq>` is the event layer: upstream PyPulseq
re-exported whole, plus Pulserver's replacements for a few of its objects.
{doc}`pulserver.design <design>` is the toolbox above it — every factory
returning a `SequenceModule` or a `ScanLoop`.

Ready-to-run sequence callbacks are intentionally a separate public namespace:
{doc}`pulserver.sequences <sequences>`.  They accept explicit sequence
controls and return a `pulserver.pypulseq.Sequence`, while this root namespace
continues to hold the bridge/plugin contract.

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

The things every `pulserver.design` factory returns, named here because they
are part of the contract rather than of any one waveform family. There are
two, plus the descriptor that annotates one of them:

| Type | Question | Feeds |
| --- | --- | --- |
| `SequenceModule` | *what do I play?* | `seq.add_block(*block)` |
| `ScanLoop` | *what varies, in what order?* | `lin_idx`/`par_idx`, `rotation`, `freq_offset_hz` |
| `EncodingAxis` | *what does one column of that loop mean?* | the counter it emits, the converter that applies |

There is one loop type, not one per axis. A `ScanLoop` is a table of
**positions**, a grouping of them into **shots** — one shot being one
excitation's worth of the loop — and one `EncodingAxis` per position column.
An encoding position is a set of frequency offsets, gradient scalings and
rotations, and nothing about that restricts it to k-space: a Cartesian echo
train, a non-Cartesian view list, an SMS slice group, a dynamic frame and an
inversion-time series are the same table under different axis declarations.
The axis fixes which converter applies (`to_scales`, `to_rotations`,
`to_frequencies`) and which counter `label_state()` reports.

A module has one setter, `set_state`: the numbers that re-render its waveforms
(lowercase keywords) plus the per-shot counters and sticky flags (uppercase
ones). Triggers and digital outputs are declared with the module, because which
block they belong on is a property of its design.

Neither type owns the loop itself. How frames, slices, contrasts and shots
nest is plain `for` statements in the plugin, which is what keeps a
preparation, a trigger or a dummy TR insertable at any level.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.ScanLoop
   pulserver.SequenceModule
   pulserver.EncodingAxis
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
