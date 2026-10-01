# Scanner sequences

A scanner sequence is a plugin file, `<name>.py` in a `--plugins` directory,
that binds a pypulseqpp {class}`~pypulseqpp.sequences.SequenceApp` to the
scanner protocol with one {class}`~pulserver.design.ScannerSequence` subclass:

```python
# sequences/gre.py
from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp
from pulserver.design import ScannerSequence

class Gre(ScannerSequence):
    app = Gre2DApp
```

The interpreter names it `--plugin gre`. Without `ui`, the protocol is the
application's defaults. A PyPulseq script becomes a `SequenceApp` by moving
what precedes its loop into `init_sequence` and the loop body into `kernel`
([from a PyPulseq script](https://pulserver.github.io/pypulseqpp/latest/user-guide/from-pypulseq.html)).
`MAX_GRAD` (mT/m) and `MAX_SLEW` (T/m/s) are required on every application:
they cap the scanner's limits, which the design call passes in.

## Binding the protocol

`ui` maps {class}`~pulserver.protocol.UIParam` members, the parameters of the
interpreter's table, to entries; a name outside that table is refused when the
class is defined. An entry names the `init_sequence` argument it sets. `recon`
names the reconstruction plugin of the sequence's data
({doc}`reconstruction-plugins`).

```pycon
>>> from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp
>>> from pulserver.design import (FloatParam, IntParam, ScannerSequence, TEPreset,
...                               TimeParam, TRPreset, UIParam)
>>> class Gre2D(ScannerSequence):
...     app = Gre2DApp
...     recon = "gre2d"
...     ui = {
...         UIParam.TE: TimeParam("te", range_min=1000, range_max=80000, range_incr=10,
...                         presets={TEPreset.MINIMUM: None}),
...         UIParam.TR: TimeParam("tr", range_min=1000, range_max=5_000_000,
...                         presets={TRPreset.MINIMUM: None}),
...         UIParam.FOV: FloatParam("fov_x", unit="mm", scale=1e-3,
...                           range_min=50.0, range_max=500.0),
...         UIParam.NX: IntParam("n_x", range_min=32, range_max=512, range_incr=2),
...     }

```

| Entry | UI | Argument |
| --- | --- | --- |
| {class}`~pulserver.design.TimeParam` | integer microseconds, with presets | seconds |
| {class}`~pulserver.design.FloatParam` | the argument divided by `scale`, in `unit` | as the application takes it |
| {class}`~pulserver.design.IntParam` | integer | integer |
| {class}`~pulserver.design.BoolParam` | checkbox | boolean |
| {class}`~pulserver.design.StringListParam` | dropdown | the chosen string |
| {class}`~pulserver.design.ConfigParam` | not shown | none; a value declared to the interpreter |
| {class}`~pulserver.design.Description` | read-only text | none |

The options of the four string-list parameters are
{class}`~pulserver.protocol.SequenceType`,
{class}`~pulserver.protocol.ImagingMode`,
{class}`~pulserver.protocol.PreparationType` and
{class}`~pulserver.protocol.TriggerType`, which
{class}`~pulserver.design.StringListParam` takes as they are.

A preset is a negative value the interpreter sends in place of a time;
`{TEPreset.MINIMUM: None}` passes `None`, for which the application designs its
shortest echo time. An entry's `default` replaces the application's default as
the initial value, in UI units: `default=TEPreset.MINIMUM` offers a protocol a
scanner with weaker gradients than the application assumes can play.

## Shipped sequences

pulserver ships a plugin for each of these pypulseqpp sequences, searched after
every `--plugins` directory, so a file of the same name there replaces it:

| Plugin | Reconstruction |
| --- | --- |
| `gre2d`, `se2d`, `bssfp2d`, `gre_multiecho2d`, `gre3d`, `se3d` | `pics` |
| `gre_radial2d`, `gre_spiral2d`, `se_radial2d`, `se_spiral2d` | `nufft` |
| `gre_stack_of_stars3d`, `gre_stack_of_spirals3d`, `se_stack_of_stars3d`, `se_stack_of_spirals3d` | `nufft` |

A 3D sequence takes its number of partitions from the number of slices. The
Cartesian ones take `Ry`, and the 3D ones `Rz`, as their undersampling.

## Resolving a protocol

{meth}`~pulserver.design.ScannerSequence.listing` is the protocol with its
schema, as the `list` design call replies it:

```pycon
>>> from pulserver.protocol import format_listing
>>> print(format_listing(Gre2D().listing()), end="")
[Protocol]
TE: int|dropdown|8000|1000|80000|10|us|-2
TR: int|dropdown|250000|1000|5000000|1|us|-1
fov: float|typein|220.0|50.0|500.0|1.0|mm
nx: int|typein|128|32|512|2|
fov_offset_x: float|off|0.0|-1000.0|1000.0|0.1|mm
fov_offset_y: float|off|0.0|-1000.0|1000.0|0.1|mm
fov_offset_z: float|off|0.0|-1000.0|1000.0|0.1|mm
fov_rotation_11: float|off|1.0|-1.0|1.0|1e-06|
fov_rotation_12: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_13: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_21: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_22: float|off|1.0|-1.0|1.0|1e-06|
fov_rotation_23: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_31: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_32: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_33: float|off|1.0|-1.0|1.0|1e-06|
[Protocol End]

```

The last twelve entries are the prescription, which the interpreter fills from
the scanner's: the field-of-view offset, which the host applies when it builds
the IR, and the rotation from the logical to the physical axes, in whose frame
the host checks the design ({doc}`../explanations/ir-cache`).

{meth}`~pulserver.design.ScannerSequence.validate` constructs the application
under the scanner limits and returns the protocol it will play. Entries the
request omits keep their initial values:

```pycon
>>> import pypulseqpp as pp
>>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
>>> reply = Gre2D().validate(system, {"TE": TEPreset.MINIMUM, "nx": 96})
>>> reply.valid, reply.duration, reply.info
(True, 36.0, '')
>>> {name: reply.values[name] for name in ("TE", "TR", "fov", "nx")}
{'TE': 3080, 'TR': 250000, 'fov': 220.0, 'nx': 96}

```

The resolved `TE` is the echo time the design achieved, as `init_sequence`
records it with {meth}`~pypulseqpp.sequences.SequenceApp.resolve`; an argument
the application does not record keeps the requested value. Values are rounded
to the precision a scanner parameter stores, so a resolved protocol sent back
resolves to itself.

A protocol the design refuses is invalid, and the error the application raised
is the reply's `info`:

```pycon
>>> reply = Gre2D().validate(system, {"TE": 1000})
>>> reply.valid, reply.info
(False, 'the requested TE of 1.000 ms is shorter than the 3.400 ms this readout can achieve')

```

`duration` is the scan time in seconds,
{meth}`~pypulseqpp.sequences.SequenceApp.scan_time`: the `duration`
`init_sequence` states, otherwise the summed duration of the designed prescans
and main sequence.

{meth}`~pulserver.design.ScannerSequence.generate` writes the design as signed
binary Pulseq, prescans first; the `generate` design call converts it to the IR
cache and stores it ({doc}`../explanations/designs`).

## See also

* {doc}`../explanations/protocol` — resolution, presets, units and precision.
* {doc}`../api/design` — the scanner-sequence interface and its UI entries.
