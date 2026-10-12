# Scanner sequences

A scanner sequence is a plugin file, `<name>.py` in a `--plugins` directory,
defining one {class}`~pulserver.design.SequencePlugin` subclass that binds a
sequence function to the scanner protocol:

```python
# sequences/gre.py
from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d
from pulserver.design import SequencePlugin


class Gre(SequencePlugin):
    app = gre2d
```

The interpreter names it `--plugin gre`. Without `protocol`, the protocol is
the sequence function's defaults. A PyPulseq script becomes a sequence function
by taking the scanner limits and its parameters as arguments and returning its
sequence
([from a PyPulseq script](https://pulserver.github.io/pypulseqpp/latest/user-guide/from-pypulseq.html)).

## Protocol entries

`protocol` maps the parameters of the interpreter's table, the members of
{data}`~pulserver.protocol.ProtocolKey`, to entries;
{class}`~pulserver.protocol.UIParam` collects those of the UI controls. A plain
string naming a parameter is stored as its member, and a name outside the
table is refused when the class is defined. Each entry names the keyword
argument of the sequence function it sets.

```pycon
>>> from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d
>>> from pulserver.design import (FloatParam, IntParam, SequencePlugin, TEPreset,
...                               TimeParam, TRPreset, UIParam)
>>> class Gre2D(SequencePlugin):
...     app = gre2d
...     protocol = {
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
| {class}`~pulserver.design.TimeParam` | integer µs, with presets | s |
| {class}`~pulserver.design.FloatParam` | the argument divided by `scale`, in `unit` | as the sequence function takes it |
| {class}`~pulserver.design.IntParam` | integer | integer |
| {class}`~pulserver.design.BoolParam` | checkbox | boolean |
| {class}`~pulserver.design.ChoiceParam` | dropdown | the chosen member of a `StrEnum` |
| {class}`~pulserver.design.ConfigParam` | not shown | none; a value declared to the interpreter |
| {class}`~pulserver.design.Description` | read-only text | none |

The options of the four string-list parameters are
{class}`~pulserver.protocol.SequenceType`,
{class}`~pulserver.protocol.ImagingMode`,
{class}`~pulserver.protocol.PreparationType` and
{class}`~pulserver.protocol.TriggerType`, which
{class}`~pulserver.design.ChoiceParam` takes as its `choices`:
`ChoiceParam("mode", ImagingMode)`. The argument receives the member, a `str`
equal to the option.

A preset is a negative value the interpreter sends in place of a time.
`{TEPreset.MINIMUM: None}` passes `None`, for which the sequence function
designs its shortest echo time. An entry's `default` replaces the sequence
function's default as the initial value, in UI units: `default=TEPreset.MINIMUM`
offers an initial protocol that a scanner with weaker gradients than the
function's default assumes can play.

The class names no reconstruction: the reconstruction client, or the scan on a
console, names it ({doc}`reconstruction-plugins`). A `recon` or `follows`
attribute has no effect and raises a `DeprecationWarning`, as do `ui` and
`ScannerSequence`, the deprecated names of `protocol` and `SequencePlugin`, and
{func}`~pulserver.design.StringListParam`.

(sequence-functions)=
## Sequence functions

A sequence function takes the scanner limits, a {class}`pypulseqpp.Opts`, and
one keyword argument for each entry of `protocol`, and returns the designed
{class}`pypulseqpp.Sequence`. A list of sequences is a chain, the prescans
first and the main sequence last; each file names the next as its
`NextSequence` definition. The initial value of an entry is the default of its
argument in the signature, so a {func}`functools.partial` is a sequence function
too. The default evaluation does not call the function;
{meth}`~pulserver.design.SequencePlugin.generate` calls it when the design is
generated ({ref}`evaluating-a-protocol`).

```pycon
>>> import numpy as np
>>> import pypulseqpp as pp
>>> def pulse_acquire(system, flip=90.0, tr=100e-3, averages=8):
...     seq = pp.Sequence(system)
...     rf = pp.make_block_pulse(np.deg2rad(flip), duration=1e-3, system=system)
...     adc = pp.make_adc(256, duration=10e-3, system=system)
...     for _ in range(averages):
...         seq.add_block(rf)
...         seq.add_block(adc)
...         seq.add_block(pp.make_delay(tr - 11e-3))
...     return seq
>>> class PulseAcquire(SequencePlugin):
...     app = pulse_acquire
...     protocol = {
...         UIParam.FLIP: FloatParam("flip", unit="deg", range_min=1.0, range_max=180.0),
...         UIParam.TR: TimeParam("tr", range_min=20000, range_max=5_000_000),
...         UIParam.NEX: IntParam("averages", range_min=1, range_max=64),
...     }
>>> listing = PulseAcquire().listing()
>>> {key.value: listing[key].value for key in PulseAcquire.protocol}
{'flip': 90.0, 'TR': 100000, 'nex': 8}

```

## Protocol resolution

{meth}`~pulserver.design.SequencePlugin.listing` is the protocol with its
schema, as the `list` design call replies it:

```pycon
>>> from pulserver.protocol import format_listing
>>> print(format_listing(Gre2D().listing()), end="")
[Protocol]
TE: int|dropdown|8000|1000|80000|10|us|-2
TR: int|dropdown|250000|1000|5000000|1|us|-1
fov: float|typein|220.0|50.0|500.0|1.0|mm
nx: int|typein|128|32|512|2|
nex: float|typein|1.0|1.0|16.0|1.0|
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
the scanner's: the field-of-view offset, applied when the IR is built, and the
rotation from the logical to the physical axes, in whose frame the design is
checked ({doc}`../explanations/protocol`). A plugin cannot bind them.

`nex` is the number of signal averages, listed by every plugin that does not
declare its own ({class}`~pulserver.design.AveragesParam`). The design plays
its main sequence that many times, written into the block table once the
sequence is designed, each repetition past the first numbered by `AVG`, and the
scan time counts each. A plugin that binds `nex` to an argument of its sequence
function, as `PulseAcquire` above does, plays its averages itself.

{meth}`~pulserver.design.SequencePlugin.validate` evaluates a request under the
scanner limits and returns the protocol the design plays. Entries the request
omits keep their initial values. The default evaluation accepts the protocol
unchanged, so the reply repeats the request, a preset included, and states no
scan time:

```pycon
>>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
>>> reply = Gre2D().validate(system, {"TE": TEPreset.MINIMUM, "nx": 96})
>>> reply.valid, reply.duration, reply.info
(True, None, '')
>>> reply.values["TE"] is TEPreset.MINIMUM, reply.values["nx"]
(True, 96)

```

An evaluation that designs the sequence states the values the design achieved.
The `TE` and `TR` definitions of a pypulseqpp sequence record its echo time and
repetition time, and {func}`pypulseqpp.sequences.duration` sums the durations
of the sequences of a chain:

```pycon
>>> from pypulseqpp import sequences
>>> from pulserver.design import Evaluation
>>> class Resolved(Gre2D):
...     def evaluate(self, system, protocol):
...         seq = self.app(system, **protocol.arguments)
...         achieved = {UIParam.TE: seq.definitions["TE"][0],
...                     UIParam.TR: seq.definitions["TR"][0]}
...         return Evaluation(protocol.replace(achieved), sequences.duration(seq))
>>> reply = Resolved().validate(system, {"TE": TEPreset.MINIMUM, "nx": 96})
>>> reply.valid, round(reply.duration, 3), reply.info
(True, 38.25, '')
>>> {name: reply.values[name] for name in ("TE", "TR", "fov", "nx")}
{'TE': 3080, 'TR': 250000, 'fov': 220.0, 'nx': 96}

```

The resolved `TE` is the shortest echo time of the design in place of the
preset. Values are rounded to the precision a scanner parameter stores, so a
resolved protocol sent back resolves to itself. A protocol the evaluation
rejects is invalid, and the message of the error it raised is the reply's
`info`:

```pycon
>>> reply = Resolved().validate(system, {"TE": 1000})
>>> reply.valid, reply.info
(False, 'the requested TE of 1.000 ms is shorter than the 3.400 ms this readout can achieve')

```

`duration` is the scan time in seconds. An evaluation that states a duration of
`0.0` states no estimate: the protocol is valid, `duration` is `None`, and the
reply reports the scan time as unknown.

{meth}`~pulserver.design.SequencePlugin.design` generates the design and writes
it as signed binary Pulseq, prescans first, and returns the written paths. The
first file is `sequence.seq`, and each file names the next as its
`NextSequence` definition. The `generate` design call then converts the design
to the IR cache and stores it ({doc}`../explanations/architecture`).

(evaluating-a-protocol)=
## Protocol evaluation

A plugin overrides {meth}`~pulserver.design.SequencePlugin.evaluate` to check a
protocol and to state what the console shows with it. The hook takes the
scanner limits and the requested {class}`~pulserver.design.Protocol`, in the
units of the sequence function's arguments, and returns an
{class}`~pulserver.design.Evaluation`: the protocol holding the values the
design achieves, the scan time in seconds, a note shown with the valid protocol
and, optionally, the RF layout ({ref}`stating-the-rf-layout`). Returning `None`
accepts the protocol unchanged. Raising an exception makes the protocol
invalid:

```pycon
>>> class Timed(PulseAcquire):
...     def evaluate(self, system, protocol):
...         if protocol[UIParam.TR] < 12e-3:
...             raise ValueError("the TR is shorter than the pulse and the readout")
...         scan_time = protocol[UIParam.NEX] * protocol[UIParam.TR]
...         return Evaluation(protocol, scan_time, "256 samples per average")
>>> reply = Timed().validate(system, {"TR": 50000, "nex": 4})
>>> reply.valid, reply.duration, reply.info
(True, 0.2, '256 samples per average')
>>> reply = Timed().validate(system, {"TR": 10000})
>>> reply.valid, reply.info
(False, 'the TR is shorter than the pulse and the readout')

```

The default evaluation does not call the sequence function, so a protocol the
function cannot realize is found only when the sequence is generated, as an
error of that call, and not while the operator edits the protocol. An
`evaluate` that builds or checks the design rejects such a protocol when it is
evaluated.

`ValueError` and `AssertionError`, which pypulseqpp and PyPulseq raise for an
event or a timing they cannot realize, reject the protocol with the message of
the error as `info`, and are logged as a warning. Any other exception also
makes the protocol invalid, but the reply names only its type, as in
`TypeError in Timed.evaluate`, and the exception is logged as an error with its
traceback. A request that names an entry the protocol does not declare, or an
option a choice does not offer, is invalid in the same way.

{meth}`~pulserver.design.SequencePlugin.generate` builds the sequence of a
requested protocol. The default calls the sequence function with the arguments
of the protocol; a plugin overrides it to return a sequence, or a list of them,
built another way. `validate` and `design` are not overridden: they are the
boundary between a request and the code of a plugin, and `design` writes
nothing for an invalid request.

(exam-cache)=
## Exam cache

An `evaluate` or `generate` that declares an `exam` argument receives the
directory of the current exam's cache, which the design calls take as
`--exam`, and reads what the exam's reconstructions stored there, such as
`b0_map`, `b1_map` or `coil_sensitivities`, with
{func}`~pulserver.design.load_exam`. Without an exam the argument is not
passed:

```python
from pulserver.design import Evaluation, SequencePlugin, load_exam


class Shimmed(SequencePlugin):
    app = pulse_acquire

    def evaluate(self, system, protocol, exam=None):
        if exam is None:
            return Evaluation(protocol, 0.0, "no B0 map")
        b0_map = load_exam(exam)["b0_map"]
        return Evaluation(protocol, 0.0, f"B0 map {tuple(b0_map.shape)}")
```

A key no reconstruction stored raises `KeyError`. The values are those of the
series that measured them, on their own grid. For such a plugin the files of
the exam's cache are part of the design's identity, so a new map yields a new
design.

(stating-the-rf-layout)=
## RF layout

An evaluation may state the RF its protocol plays, from which a scanner
estimates the RF of a prescription before the design is generated. The layout
is one TR: the RF instances that TR plays, with the TR as its `period`.
{meth}`RfLayout.of <pulserver.design.RfLayout.of>` takes the RF instances of a
sequence that is one TR, as {meth}`pypulseqpp.Sequence.rf_instances` returns
them, and the control each instance's amplitude follows: the flip angle,
`UIParam.FLIP`, or a float user entry, `UIParam.user_value(n)`, either declared
in `protocol`. One control applies to every instance; a list gives one per
instance in play order, `None` for an instance no entry scales. The scanner
multiplies the amplitude of an instance by the ratio of the value of its
control in the protocol it plays to the value in the evaluated protocol. The
`period` is the TR in seconds, and the duration of the sequence where omitted:

```pycon
>>> from pulserver.design import RfLayout
>>> from pulserver.protocol import format_rf_layout
>>> class Costed(PulseAcquire):
...     def evaluate(self, system, protocol):
...         one_tr = pulse_acquire(system, **(protocol.arguments | {"averages": 1}))
...         layout = RfLayout.of(one_tr, UIParam.FLIP)
...         scan_time = protocol[UIParam.NEX] * protocol[UIParam.TR]
...         return Evaluation(protocol, scan_time, rf_layout=layout)
>>> reply = Costed().validate(system, {"nex": 4})
>>> print(format_rf_layout(reply.rf_layout), end="")
[RfLayout]
period 0.1
run 0 1 flip 1
[RfLayout End]

```

A sequence shorter than the TR, such as one TR without its closing delay, is
stated with the TR: `RfLayout.of(seq, UIParam.FLIP, period=tr)`.

The layout is optional. An evaluation that states none is valid and states no
estimate; one whose control is not an entry of `protocol`, or has no positive
value in the evaluated protocol, is invalid. What the scanner checks before the
scan is the stored design, not the layout. The blocks that carry the layout are
described in {doc}`../explanations/protocol`.

(shipped-sequences)=
## Shipped sequences

pulserver ships a plugin for each of these pypulseqpp sequences, searched after
every `--plugins` directory, so a file of the same name there replaces it:

| Plugin | Paired reconstruction |
| --- | --- |
| `gre2d`, `se2d`, `bssfp2d`, `gre_multiecho2d`, `gre3d`, `se3d`, `bssfp3d`, `gre_multiecho3d` | `pics` |
| `fse3d`, `mprage3d` | `pmc` |
| `gre_radial2d`, `gre_spiral2d`, `se_radial2d`, `se_spiral2d`, `gre_propeller2d`, `se_propeller2d`, `se_epi_propeller2d`, `zte3d` | `nufft` |
| `gre_stack_of_stars3d`, `gre_stack_of_spirals3d`, `gre_stack_of_blades3d`, `se_stack_of_stars3d`, `se_stack_of_spirals3d`, `se_stack_of_blades3d` | `nufft` |
| `mprage_stack_of_stars3d`, `mprage_stack_of_spirals3d` | `nufft_train` |
| `epi2d`, `epi3d` | `epi` |

The paired reconstruction is the one a console uses for a shipped sequence when
the scan names none ({doc}`virtual-scanner`); a scan can name another, and a
scanner's reconstruction client names one in its config
({doc}`reconstruction-client`).

### Shared user entries

Each shipped plugin offers every one of these controls that its sequence
function takes, at the same user entry in every plugin, each as a
`user<n>_name` label and a `user<n>_value`; a plugin's own user entries start at
`user12`:

| User entry | Control |
| --- | --- |
| `user0` | Dummy scans; −1, the default, plays until the steady state, where the sequence chooses it (bSSFP, MPRAGE, ZTE and FSE count their own) |
| `user1` | Partial echo, the fraction of the echo acquired |
| `user2`, `user3` | Partial Fourier along the phase and the partition encodes |
| `user4`, `user5` | Calibration (ACS) lines along the phase and the partition encodes |
| `user6` | Readout oversampling |
| `user7` | Echo spacing, in ms; 0 the shortest |
| `user8` | Refocusing flip angle, in degrees |
| `user9` | CAIPIRINHA shift, in partitions per acquired line |

`bssfp3d` takes its number of phase cycles, each a train with its own RF phase
increment, as `user12`.

### Module constants

Choices a console has no entry for are module constants of the plugin, set to
the sequence function's default unless stated. To change one, copy the plugin
file into a `--plugins` directory and set the constant there; the copy replaces
the shipped plugin.

| Plugin | Constant | Choice |
| --- | --- | --- |
| `gre2d`, `se2d` | `BANDS` | Spatial saturation bands played before every excitation, 2 by default; 0 plays none |
| 3D Cartesian, stack and MPRAGE plugins | `EXCITATION` | `"slab"`, `"spsp"` (water only) or `"nonselective"`; `epi3d` ships `"spsp"` |
| `epi2d` | `FAT_SATURATION` | Fat saturation before every shot; `True` by default |
| `gre3d`, `se3d`, `gre_multiecho3d` | `ELLIPTICAL` | Only the phase encodes inside the ky-kz ellipse |
| `gre_multiecho2d`, `gre_multiecho3d` | `FLYBACK` | Monopolar echoes; `False` reads even echoes reversed |
| `fse3d` | `ORDERING` | `"radial"`, or `"shuffling"`: Poisson-disc views in random order |
| `zte3d` | `SCHEME` | Spokes ordered `"spiral"` or `"meridian"` |
| `bssfp2d` | `RETROSPECTIVE` | ECG gating: `False`, prospective, each heartbeat triggered and acquired once per cardiac phase; `True`, retrospective, each segment cycled over a heartbeat and binned into cardiac phases by the reconstruction |

The saturation bands of `gre2d` and `se2d` are placed graphically, as a
scanner's console places an explicit band, and are always on. `exsat_mask` is a
configuration the plugin declares, with bit `n - 1` set for each band `n` it
plays; `exsat<n>_normal_x`, `_y` and `_z`, `exsat<n>_loc` and
`exsat<n>_thickness` give each band's normal along the physical axes, the
distance of its centre from the isocentre along that normal, and its thickness,
in mm. The field-of-view offset and rotation leave a band where it was placed,
and a band not yet placed sits 150 mm from the isocentre, outside the head.

Optional features are switched by a module constant, off in the shipped files.
A switch adds the entries the feature needs, a scanner entry where one exists
and a user entry otherwise:

| Plugin | Constant | Feature | Entries added |
| --- | --- | --- | --- |
| `epi2d` | `MULTIBAND` | Simultaneous multislice | `multiband` |
| `fse3d` | `OPTIMIZED` | Refocusing train designed with blochsim, shared by every shot | Flip angle: the refocusing angle at the TE echo |
| `fse3d` | `DUAL_REGION` | Trains designed with blochsim for the centre and the periphery of k-space, each shot's train a cubic step between them; `TR` and `ETL` are the centre's | Flip angle, as `OPTIMIZED`; `user12` and `user13`: TR and ETL at the periphery |
| `fse3d`, `mprage3d` | `NAVIGATOR` | Three-plane navigators; the design sets `EnablePmc`, and `pmc` publishes each pose to the scan | None |
| `gre3d`, `gre_multiecho3d`, `mprage3d`, `fse3d` | `WAVE` | Wave-CAIPI at `WAVE_AMPLITUDE` (T/m) and `WAVE_CYCLES`; the calibration region is acquired first without the wave, and `pics` and `pmc` reconstruct the wave-encoded volume | None |
| `gre_spiral2d`, `se_spiral2d`, `gre_stack_of_spirals3d`, `se_stack_of_spirals3d`, `mprage_stack_of_spirals3d` | `VARIABLE_DENSITY` | Variable-density spirals | `user12`: periphery undersampling |

### Diffusion tables

A diffusion sequence plays a table of b-values and directions read from a file,
as a scanner's own diffusion sequences do: the console names the table by
number in the `UIParam.DIFFUSION_TABLE` entry, with the largest b-value in
`UIParam.DIFFUSION_BVALUES` and the number of directions in
`UIParam.DIFFUSION_DIRECTIONS`; the sequence function turns the number into the
file's path with module constants a site sets, reads the file and records its
path with {func}`pypulseqpp.io.set_diffusion_table`. The proxy reads the same file
again, so the table reaches the reconstruction without being copied into the
sequence ({doc}`reconstruction-plugins`).

```python
import os

import pypulseqpp as pp
from pulserver.design import FloatParam, IntParam, SequencePlugin, UIParam

TENSOR_DIRECTORY = "/data/diffusion"
TENSOR_PREFIX = "tensor"
TENSOR_SUFFIX = ".dat"


def dwi(system, table=1, b_value=1000.0, directions=30):
    seq = pp.Sequence(system)
    name = f"{TENSOR_PREFIX}{table}{TENSOR_SUFFIX}"
    path = os.path.join(TENSOR_DIRECTORY, name)
    b, g = pp.io.set_diffusion_table(seq, path, b_value, directions)
    for volume, (b_volume, direction) in enumerate(zip(b, g)):
        ...  # one volume, its REP label set to `volume`
    return seq


class Dwi(SequencePlugin):
    app = dwi
    protocol = {
        UIParam.DIFFUSION_TABLE: IntParam("table", range_min=1),
        UIParam.DIFFUSION_BVALUES: FloatParam("b_value", unit="s/mm²"),
        UIParam.DIFFUSION_DIRECTIONS: IntParam("directions", range_min=1),
    }
```

The table is in the frame of the gradient channels, so its directions turn with
the prescription.

### Protocol conventions

- A 3D sequence takes its number of partitions from the number of slices.
- `Ry` undersamples the phase encode of the Cartesian sequences, the blades of
  `gre_propeller2d`, `se_propeller2d` and the stacks of blades, and the shells
  of `zte3d`; `Rz` undersamples the partition encode of the 3D Cartesian ones.
- The `ETL` of `se_epi_propeller2d` is its blade width.
- The MPRAGE sequences take their inversion time as `prep_time`.

### Shipped evaluation

Each shipped plugin evaluates a protocol from one repetition of the scan: one
line, partition, spoke, interleaf, blade line, echo train or inversion shot of
one slice, without dummy repetitions. Most call the sequence function with the
arguments that reduce it to that repetition; `epi3d` and `zte3d` build the
repetition from the modules their function is built from.

- **Scan time.** The duration of the repetition times the number of
  repetitions the prescription plays, with the slices that share a TR counted
  in the packets the sequence plays them in, plus the dummy repetitions: those
  of `user0`, or, at −1, the count {func}`pypulseqpp.sequences.steady_state_dummies`
  gives for that TR and flip angle.
- **Resolved values.** Those the main sequence states for the prescription,
  where the plugin has the entry: the echo time and repetition time in its `TE`
  and `TR` definitions, the receiver bandwidth as the inverse of the dwell time
  of its first ADC event, and the slice thickness in its `SliceThickness`
  definition. A multi-echo sequence lists every echo time, and the `TE` entry
  holds the first.
- **Invalid protocols.** A prescription the design refuses is invalid, with
  the message of its error.
- **RF layout.** The first TR of the repetition, its shot repeated once per
  slice of the largest packet, with the TR as its period; for the balanced
  steady state, the TR after the half-angle pulse. The TR of a fast spin echo
  holds an echo train, that of an MPRAGE an inversion shot, and that of `epi3d`
  a volume. The flip angle is the control of every excitation of the plugins
  with a `flip` entry, which an MPRAGE's inversion is not. The spin-echo
  plugins, `fse3d` among them, have none, and the amplitude of their
  excitation and refocusing pulses is that of the design.

## Logging

A plugin logs with the standard `logging` module, under a logger of its own
(`logging.getLogger(__name__)`). Under a warm design server, what a call logs,
prints or warns, and why it refuses a protocol, is appended to the file the
request names under `log`, or else to the server's standard error
({doc}`running`).

## See also

* {doc}`../explanations/protocol`: resolution, presets, units and precision.
* {doc}`../api/design`: the scanner-sequence interface and its entries.
