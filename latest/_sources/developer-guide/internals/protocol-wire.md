# Protocol text blocks

The text blocks in which the design calls and the interpreter exchange
protocols and RF descriptions ({doc}`../../explanations/protocol`). The
grammar is implemented in {mod}`pulserver.protocol` for the host and in
`pulseg_protocol.h` for the interpreter.

(rf-layout)=
## RF layout

The RF a protocol plays depends on the protocol. An evaluation may state it as
an {class}`~pulserver.design.RfLayout`, from which a scanner estimates the RF of
a prescription without designing it: the RF definitions of a sequence, the RF
instances of one TR in play order, and for each instance the protocol entry its
amplitude follows.

A definition is one RF pulse: a complex waveform at unit peak magnitude, its
timing and its use. Instances of a definition differ in amplitude and in
frequency and phase offset. In the sequence an evaluation builds, the amplitude
of an instance is relative to the first instance of its definition with a
nonzero amplitude, so a refocusing train with a flip angle schedule is one
definition with one instance per pulse, and RF spoiling, which steps the phase
offset, adds no definition ({meth}`pypulseqpp.Sequence.rf_instances`).

A control of an instance is the flip angle, or a float user entry, of the
plugin's protocol. It states that the amplitude of the instance is proportional
to the entry: a scanner that plays a protocol with another value of the entry
multiplies the amplitude by the ratio of that value to the value in the
evaluated protocol, which is positive. An instance without a control is played
at the amplitude it was evaluated at.

A validation states the amplitude of an instance over the `peak_hz` the listing
states for its definition, not over the first instance of the definition in the
evaluation of the validation. A protocol that changes a flip angle changes the
peak of the definition in its own evaluation, and the amplitude stated includes
that change, so that the product of the amplitude and the listed `peak_hz` is the
peak RF amplitude the instance plays.

A layout is one TR: the instances one TR plays, and the TR as its `period`,
over which a scanner averages their RF. Its size does not depend on the matrix.
The shipped plugins state the first TR of the repetition they design, its shot
repeated once per slice of the largest packet, and a balanced steady state the
TR after its half-angle pulse. A definition is named by its number in the
sequence the evaluation builds, in the order of first play. The listing carries
the definitions of the evaluation at the default protocol, and a layout names
the pulses of the listing only where the protocol does not change which pulses
are played first: the length of a train, a flip angle schedule and the matrix do
not, and a protocol that adds a preparation pulse does.

The layout is an estimate for the prescription. An evaluation that states none
states no estimate and is valid, and what the scanner checks before the scan is
the stored design ({doc}`../../explanations/architecture`).

## Keys and values

The keys of a protocol are members of the key enums of
{mod}`pulserver.protocol`, of which {data}`~pulserver.protocol.ProtocolKey` is
the union. A key is a `str` that equals and hashes as its wire name, so a plain
string indexes a mapping of keys, and a scanner sequence stores a plain string
in `protocol` as the member it names. A {class}`~pulserver.design.Protocol`
maps each key to its value in the units of the sequence function's argument: seconds for a
time, the argument's unit for a float, a member of the enum for a
{class}`~pulserver.design.ChoiceParam`. The prescription entries bind no
argument and keep the units of the wire.

The wire carries the same protocol as integer microseconds, values in the
entry's UI unit and, for a choice, the index of the chosen option.
{meth}`~pulserver.design.Protocol.from_wire` and
{meth}`~pulserver.design.Protocol.to_wire` are the only conversions between
the two, and the formatting and parsing functions of
{mod}`pulserver.protocol` are the only conversions between keys and values and
the text of a block. A value block that names no option of a choice is refused
when it is parsed, and a request that names none is invalid when it is given
to {meth}`~pulserver.design.SequencePlugin.validate` as a mapping.
{func}`~pulserver.design.StringListParam`, which builds the enum from option
strings, is deprecated: declare the enum and use `ChoiceParam`.

## Listing and value blocks

Protocols are exchanged as text blocks delimited by `[Protocol]` and
`[Protocol End]`. A listing block carries each entry with its schema, one line
per entry:

```text
TE: int|dropdown|8000|1000|80000|10|us|-2
```

that is, kind, input mode, current value, minimum, maximum, increment, unit and
the dropdown options. The line of a string list is its kind, the index of the
chosen option and the options:

```text
imaging_mode: stringlist|1|2d|3d
```

A value block carries `name: value` lines only, a string list as the index of
its chosen option. The grammar is implemented in {mod}`pulserver.protocol` for
the host and in `pulseg_protocol.h` for the interpreter.

The RF blocks follow a reply when the call asks for them, and no block follows
where the evaluation is invalid or states no layout. They are lists of numbers:
each number is ASCII decimal with nine significant digits, the numbers of a list
are separated by single spaces, and each list is one line, so that a reader
takes tokens. `[RfDefinitions]` follows the listing of a `list` call asked with
`rf_definitions`, which evaluates the plugin at its default protocol under the
scanner limits. Each definition is a `definition` line, then the sample times,
then the real and the imaginary parts of each channel:

```text
[RfDefinitions]
definition <index> <use> <flip_deg> <peak_hz> <bandwidth_hz> <delay_s> <center_s> <duration_s> <channels> <samples>
<time_s> ...
<real> ...
<imaginary> ...
[RfDefinitions End]
```

`index` is the number the runs of a layout name the definition by, from 0 in the
order of first play, and `use` is the RF use of pypulseqpp. `flip_deg` and
`peak_hz`, in degrees and Hz, are those of the first instance with a nonzero
amplitude, and `bandwidth_hz` is the bandwidth that
{func}`pypulseqpp.calc_rf_bandwidth` measures on the sum of the channels, in Hz.
`delay_s` is the delay of the RF event in its block; `center_s`, `duration_s` and
the sample times are in seconds from the start of the event, its delay excluded.
The channels share the sample times. The waveform is scaled to unit peak
magnitude over all channels and samples, with the RF shim applied; the frequency
and phase offsets of the event are playout parameters and are not applied.

`[RfLayout]` follows the value block of a valid `validate` reply asked with
`rf_layout`. The instances are run-length encoded in play order, one `run`
line for consecutive instances of one definition, control and printed
amplitude, and carry no samples:

```text
[RfLayout]
period <s>
run <index> <amplitude> <control|-> <count>
[RfLayout End]
```

`period` is the TR in seconds, over which the instances repeat. The `amplitude`
of a run is unitless: the peak RF amplitude of its instances over the `peak_hz`
the listing states for the definition, so that `amplitude × peak_hz × waveform`
is what an instance plays. The host reads those peaks by evaluating the plugin
at its default protocol as well. A run whose definition the listing does not
state, or states with a peak of zero, carries its amplitude relative to the first
instance of its definition with a nonzero amplitude in the evaluation of the
request, as does every run where the evaluation at the default protocol is
invalid. `control` is the wire name of the entry the amplitude is proportional
to, or `-`. A reader skips the blocks it does not know.

