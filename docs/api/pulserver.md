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
   pulserver.write_sequence
```

A finished sequence is written differently depending on where it is going, and
`write_sequence` is the one place that knows which is which. `make_sequence`
takes an `offline` flag and hands it straight on:

| | `offline=False` — the scanner | `offline=True` — anywhere else |
|---|---|---|
| format | binary | `.seq` text |
| timing / gradient / limit checks | no | yes |
| signature | no | yes |

Both are deduplicated: it costs a millisecond or two and takes several times
the size off the file, which the interpreter then does not have to parse.

The scanner form skips the checks because the interpreter runs its own at
predownload, against its real rasters and its real limits — the authoritative
pass — and it parses the binary format far faster than the text one. The
offline form runs everything, because a bench file, a foreign toolbox or a
reader has nothing downstream to catch anything. `run_cli` passes
`offline=True`; the bridge takes the default.

The two sides meet at the labels: what a sequence writes with
{func}`~pulserver.pypulseq.make_label` is what arrives on an acquisition, and
the vocabulary maps one-to-one onto ISMRMRD. A `ReconPlugin` names the flag it
splits on in either spelling — `split_on="LASTSLC"` and
`split_on="ACQ_LAST_IN_SLICE"` mean the same bit — so a plugin can wait for a
boundary in the words the sequence used to mark it. See
{doc}`pulserver.pypulseq <pypulseq>` for the map and for how
`Sequence.auto_label` derives the `FIRST`/`LAST` flags rather than making a
sequence compute them.

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
and the offline default protocol. The `Off*` pair is for what the protocol
carries without showing: the prescribed volume offset, which the scanner fills
in from where the operator put the slab rather than from a widget.

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.BoolParam
   pulserver.Description
   pulserver.DropdownFloatParam
   pulserver.DropdownIntParam
   pulserver.OffFloatParam
   pulserver.OffIntParam
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
   pulserver.TEPreset
   pulserver.TRPreset
   pulserver.TriggerType
   pulserver.UIParam
   pulserver.Validate
```

`TEPreset` and `TRPreset` are the TE and TR dropdown entries a scanner UI shows
as words rather than numbers — "Minimum", "Min Full". They are negative, which
is what marks them as requests rather than times, and they go in a dropdown's
`options` beside the real values:

```python
UIParam.TE: DropdownFloatParam(value=8.0, min=1.0, max=80.0, unit="ms",
                               options=[TEPreset.MINIMUM, 5.0, 8.0, 15.0])
```

`main_kwargs` turns any of them into `te=None` or `tr=None`, which is what every
readout module already reads as "as short as possible" — so offering the preset
costs a plugin nothing beyond listing it.

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

`main_kwargs` sits above them and does the whole translation at once. It reads
the signature of the sequence's `main` and fills in every keyword-only
parameter that names a quantity the scanner prescribes — matrix, FOV, slice
geometry, flip angle, TE, TR, bandwidth, acceleration, volume offset — each in
the unit `main` states it in rather than the millimetres and milliseconds the
protocol carries. Parameters it does not recognise keep their own defaults, and
a plugin's own controls, the ones living in its user slots, are passed through
as overrides:

```python
def make_sequence(self, system, protocol, output_path):
    prot = dict_to_protocol(protocol)
    seq = main(**main_kwargs(main, system, protocol,
                             n_acs=params.acs_lines_from_protocol(prot, n_y, 0),
                             partial_echo=params.user_float(prot, 1, 1.0)))
    seq.write(output_path)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/pulserver
   :nosignatures:

   pulserver.main_kwargs
```
