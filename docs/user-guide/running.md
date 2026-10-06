# Running the services

Pulserver runs two services. The design calls answer the scanner's interpreter
host processes: they resolve protocols and write the designs the scanner plays into a
design store, one command per call or through a warm server. The
reconstruction proxy receives the raw data of each series from the scanner's
reconstruction client, reads the design it was played from in a design store,
and returns the images. When the two run on different computers, either the
store is a directory both can reach, or the proxy keeps a store of its own and
its design intake receives each design the design calls push to it. The store
and the identity of a design are described in
{doc}`../explanations/designs`. Where the reconstructions run on a computer of
their own, a reconstruction server runs there and the proxy forwards each
series to it.

## Design calls

```bash
pulserver design list     --plugins DIR --plugin NAME [--limits FILE --rf-definitions]
pulserver design validate --plugins DIR --plugin NAME --limits FILE [--rf-layout] < VALUES
pulserver design generate --plugins DIR --plugin NAME --limits FILE --store DIR [--push URL] < VALUES
pulserver design import   --limits FILE --store DIR [--push URL] < IMPORT
pulserver design push     --store DIR --to URL ID...
pulserver design prune    --store DIR [--max-age-days D] [--max-bytes B]
```

Each call is answered from its arguments and standard input alone and writes
its reply to standard output; `python -m pulserver.host` takes the same
arguments. `--plugins` is a directory of scanner-sequence plugin files,
`<plugin>.py`, and may be repeated: a plugin is the file of the first directory
that holds it. `--store` is the design store, created when missing.

| Call | Reply |
| --- | --- |
| `list` | `PROTOCOL` and the plugin's listing block; with `--rf-definitions`, the `[RfDefinitions]` block |
| `validate` | `VALID <seconds>` or `INVALID`, an `INFO` line and the value block of the resolved protocol; with `--rf-layout`, a valid reply ends with the `[RfLayout]` block |
| `generate` | `GENERATED <id>`: the identifier of the design in the store |
| `import` | `IMPORTED <id>` |
| `push` | `PUSHED <count>`: the designs sent, not counting those the intake holds already |
| `prune` | `PRUNED <count>` |

The RF blocks are sent only when asked, and only where the plugin's evaluation
states an RF layout ({ref}`stating-the-rf-layout`). `--rf-definitions` evaluates
the plugin at its default protocol under `--limits`, which it requires, and
follows the listing with the RF definitions of that evaluation. `--rf-layout`
follows a valid reply with the RF layout of the evaluation of the request, its
amplitudes over the peaks the listing states, which it reads by evaluating the
plugin at its default protocol as well. Both blocks are described in
{doc}`../explanations/protocol`. Without a flag, `list` reads no limits and
`validate` replies no RF block.

A call that fails replies `ERROR <message>` and exits with status 1; a plugin
that ends its process ends the call without a reply and with a nonzero exit
status. The value blocks are those of {mod}`pulserver.protocol`, and the import
block names the first file of a chain and, optionally, its prescription:

```text
[Import]
file: /data/sequence.seq
fov_offset_x: 0.0
fov_offset_y: 20.0
fov_offset_z: 0.0
[Import End]
```

with the offset in mm along the logical axes and the rotation in the
`fov_rotation_ij` lines of {doc}`../explanations/protocol`. `generate` designs a
request once, writes it in the logical frame, checks it and converts it to the
IR cache at the prescribed field-of-view offset; `import` copies the chain, then
checks and converts it the same way. A request that resolves to a stored
design returns its identifier without designing again.

### Limits

The limits file holds a `[Limits]` block, one `name: value` per line between
`[Limits]` and `[Limits End]`. The lines are keyword arguments of
`pypulseqpp.Opts`, of which `B0`, the field in T the scan runs at, is
required: ppm offsets are resolved at it, and a call without it is refused.
Further options are for the IR conversion:

| Option | Meaning |
| --- | --- |
| `ir_vendor` | `PULSEG_VENDOR_*` code the cache is tagged with; 0 is vendor-neutral |
| `ir_label_column_map` | Three Pulseq label state indices, separated by spaces, filling the ADC label columns |
| `ir_cache_ext` | Extension of the cache file, `.pseg` by default |
| `ir_vendor_file` | Vendor file the cache is grouped and stored under, read by {func}`~pulserver.ir.read_vendor`; a file it cannot read refuses the call |
| `ir_wave_max_samples` | Samples each gradient axis of the playout's waveform memory holds for waves |
| `ir_wave_raster_us` | The playout's gradient raster, in µs per sample |
| `ir_wave_load_us_per_sample` | Time the playout takes to load one sample on one axis, in µs; 0, the default, leaves the loading unchecked |
| `ir_wave_headroom` | Share of the playout's time its loading may take, 0.5 by default |
| `ir_wave_slots` | Slots per segment position that a playout streaming the waves rings through, 2 by default |

The `ir_wave_` options are the waveform memory of the scanner's playout, which
the cache lays the waves out for ({class}`~pulserver.ir.WaveBudget`);
`ir_wave_max_samples` and `ir_wave_raster_us` come together. Without them every
wave is held at once on the gradient raster of the chain's first file, and a
playout that holds them otherwise refuses the cache.

A generated design and an imported chain are checked against the limits before
their IR cache is written ({doc}`../explanations/designs`), so the limits
include the RF and ADC dead times, the RF ringdown time and the ADC sample
divisor of the scanner; `pypulseqpp.Opts` sets the dead times to zero and the
divisor to four when they are left out.

`max_grad` and `max_slew` are the limits of the gradient coils, which the checks
hold every physical axis to. Two design limits cap the limits a sequence is
designed under:

| Limit | Meaning |
| --- | --- |
| `design_max_grad` | Gradient amplitude each logical axis is designed under, in `grad_unit` as `max_grad` is |
| `design_max_slew` | Slew rate each logical axis is designed under, in `slew_unit` as `max_slew` is |

The scanner derates them for the prescription's rotation: logical axes played
together add on one physical axis, so under an oblique prescription a design
held under `max_grad` and `max_slew` alone can exceed them on a physical axis,
and is then refused. A design limit above the scanner's leaves the scanner's,
one left out is the scanner's, and `validate` resolves a request under the same
limits as `generate`. An imported chain is not designed, and is checked against
`max_grad` and `max_slew` alone.

The limits also carry what the host checks besides the gradient limits and
rasters, the scanner's nerve model and its forbidden gradient bands, and, where
SAR is computed from virtual observation points, the VOPs. A check whose limits
are left out is not run. No SAR limit is checked on the host: it writes into
the cache each subsequence's SAR at the VOPs relative to a reference pulse
({doc}`../explanations/designs`), and the interpreter computes the SAR and
the gradient heating.

| Limit | Meaning |
| --- | --- |
| `pns_chronaxie`, `pns_rheobase`, `pns_alpha` | Chronaxie nerve model: chronaxie in s, rheobase in T/m/s, `alpha` 1 when left out. The rheobase and the alpha are one value or three separated by spaces, one per physical axis |
| `pns_<axis>_<field>` | SAFE nerve model, for the axes `x`, `y` and `z` and the fields `a1` to `a3`, `tau1` to `tau3` in ms, `stim_limit` in T/m/s and `g_scale` |
| `pns_limit` | Largest PNS response allowed, as a fraction of the model's threshold; 1 when left out |
| `forbidden_band_<n>` | One forbidden band: its physical axis (`x`, `y`, `z` or `all`), its lowest and highest frequency in Hz and, optionally, the largest amplitude allowed in it in mT/m, separated by spaces |
| `vop_file` | `.mat` or `.npz` file of VOPs and the head SAR matrix of each body model, at a path the design calls can read; its `safety_factor` multiplies local SAR |
| `vop_head_limit`, `vop_local_limit` | The scanner's head and local SAR limits in W/kg in its current operating mode; both needed with a `vop_file` |
| `vop_drive_per_hz` | Channel drive per Hz of RF amplitude, one per channel separated by spaces; a scale common to every channel cancels, and equal drives when left out |
| `vop_default_shim` | Magnitude and phase in rad of each channel's weight for a pulse played without an RF shim, separated by spaces; equal weights when left out |
| `vop_coil` | Transmit configuration the scanner reports; the VOP file's `transmit` metadata must name the same one |

A band given no amplitude is held to the `min_threshold` of
`pypulseqpp.safety.check_mech_resonance`. The gradient, PNS and resonance
checks are made in the physical frame of the prescription rotation each
request carries. A call refuses limits it cannot read, and a design is
identified with the contents of the VOP file as well as its path.

A plugin can be exercised without a scanner through
{func}`~pulserver.host.call`, which answers a call in the calling process:

```python
from pulserver.host import DesignStore, call

limits = {"max_grad": 40.0, "grad_unit": "mT/m", "max_slew": 150.0, "slew_unit": "T/m/s"}
block = "[Protocol]\nTE: 5000\nnx: 96\n[Protocol End]\n"
status, reply = call("validate", plugins="sequences", plugin="gre2d",
                     limits=limits, block=block)
status, reply = call("generate", plugins="sequences", plugin="gre2d",
                     limits=limits, block=block, store=DesignStore("designs"))
```

A plugin file is imported again when its modification time changes.

### Design store

The store holds one directory per design, `<store>/<id>/`: the Pulseq files of
the chain, the IR cache, the resolved protocol and `manifest.json`, which
records the identity and identifier, the limits, the package versions, the
creation time, the plugin and the source it was designed from, the scan time,
the field-of-view offset (and the rotation, for an imported chain) and the
SHA-256 of every file. The identifier is three 24-bit integers, each
held exactly by a float32 scanner parameter. `prune` removes designs least recently used
first: those unused for longer than `--max-age-days`, then others until the
store holds at most `--max-bytes`. A design used within the last day is kept
whatever the limits, since a series may still be playing it or the proxy
reading it by identifier. A store that
receives pushed designs is pruned the same way, on the reconstruction computer.

### Pushing designs

With `--push`, or the `PULSERVER_DESIGN_PUSH` environment variable, naming the
URL of a proxy's design intake, `http://<host>:<intake-port>`, `generate` and
`import` send the design they reply to that intake before replying; `push`
sends the stored designs it names to the intake `--to` names. A design travels
as a bundle, a gzip-compressed tar of the files of its directory, and the
intake stores it only when every file has the SHA-256 its manifest records. A
design the intake holds already is not sent again.

A `generate` or `import` whose design cannot be pushed replies
`ERROR design <id> is stored but not pushed: <reason>` and exits with status 1,
so the interpreter receives no identifier of a design the intake lacks. The
design stays stored, and the same call made again pushes it without designing
again. `push` replies `ERROR design <id> is not pushed: <reason>` for the first
design it cannot send.

Without `--rf-definitions`, `list` depends on the plugin file and the installed
packages only, so its reply can be written when they are installed and read
without a call. With it, the reply depends on the limits as well.

### Warm server

```bash
pulserver design serve --plugins DIR --socket PATH
```

A call answered in its own process imports the design engine first, which
takes most of its time. A warm server imports it once, designs, checks and
converts a sequence of its own, and imports every plugin of its `--plugins`
directories. It then
answers each call forwarded to its socket in a child process forked from it, so
nothing a call does outlives the call, and a plugin that ends its process fails
its own call only, with `ERROR the design call ended with exit status N`.
Calls run concurrently, one child each. A call is forwarded when `--socket`, or
the `PULSERVER_DESIGN_SOCKET` environment variable, names the socket of a
running server, and is answered in its own process otherwise; the reply is the
same either way. The command imports neither NumPy nor pypulseqpp to forward a
call. The server stops on `SIGINT` or `SIGTERM` and ends the calls it is
running. The socket is neither authenticated nor encrypted: its file
permissions decide who may call.

The server logs to standard error one line per call, naming the call, its
plugin, its exit status and its duration. A request whose JSON carries `log`,
a file path, has its child append everything else it logs and prints to that
file, so a scanner's interpreter can keep the design calls of one scan beside
the rest of that scan's log.

## Reconstruction proxy

```bash
pulserver proxy --store DIR --port N --plugins DIR [--host ADDR] [--intake-port N] [--queue DIR] [--exams DIR] [--slots N] [--gpu-slots 1] [--spares 1] [--recon-timeout S] [--dicom] [--save-data DIR] [--idle-timeout S] [--logfile FILE] [--log-level LEVEL]
pulserver proxy --store DIR --port N --forward HOST:PORT [--forward-config NAME] [--host ADDR] [--intake-port N] [--recon-timeout S] [--dicom] [--save-data DIR] [--idle-timeout S] [--logfile FILE] [--log-level LEVEL]
```

| Option | Meaning |
| --- | --- |
| `--store` | The design store the proxy reads: the one the design calls write, or the one the intake writes |
| `--port` | TCP port the scanner's reconstruction client connects to |
| `--host` | Address to listen on; the loopback interface when unset, `0.0.0.0` for every interface |
| `--intake-port` | TCP port of the design intake, on the `--host` address; no intake when unset |
| `--plugins` | Directory of reconstruction plugin files, `<plugin>.py`, repeatable, searched in order; required unless `--forward` is given |
| `--queue` | Directory the series waiting for a slot are written to; a temporary directory, removed when the proxy stops, when unset |
| `--exams` | Directory the proxies of this host share the caches of an exam in; a location under the system temporary directory when unset |
| `--slots` | Series reconstructed at once; derived from available memory and the GPUs when unset |
| `--gpu-slots` | Series reconstructed at once on each GPU when `--slots` is unset, default 1 |
| `--spares` | Worker processes started ahead of a series, default 1 |
| `--recon-timeout` | Seconds a reconstruction may run after its series ends; the worker is then terminated, or the connection to the server closed, and the client told. Unlimited when unset |
| `--forward` | `HOST:PORT` of the MRD server that reconstructs every series, instead of local workers |
| `--forward-config` | Config name sent to the `--forward` server; the series' reconstruction plugin when unset |
| `--dicom` | Convert each image to DICOM before it is relayed, whatever reconstructed it |
| `--save-data` | Directory each series is kept in as the scanner sends it; kept nowhere when unset |
| `--idle-timeout` | Seconds without a client after which the proxy closes; it waits for whatever is running. Unlimited when unset |
| `--logfile` | File the log is appended to, with the standard output and error of the proxy and of its workers, so what a reconstruction plugin logs, prints or warns lands there too; the standard error when unset |
| `--log-level` | `DEBUG`, `INFO`, `WARNING` or `ERROR`; `INFO` when unset |

The MRD header of each series names the design it was played from in the
`pulserver_design` user parameter, and the proxy refuses a series whose header
names no stored design. The client's config text names the reconstruction
plugin, as a bare plugin name or the path of its file, or as either under
`parameters.config`; the design names none. A series whose config names none is
refused unless the proxy forwards with `--forward-config`. The messages and
fields the client sends are listed in {doc}`reconstruction-client`.

A series that finds every slot busy is written to the queue directory as it
arrives and reconstructed once a slot frees; the client stays connected
meanwhile.

With `--save-data`, each series is also written to
`<save-data>/mrd_<timestamp>.h5` as the scanner sends it, header first and each
acquisition as it arrives, before anything enriches it. A plugin runs on the
file against the store the series names, which reconstructs it again offline
exactly as the proxy did ({doc}`reconstruction-plugins`). The queue directory
holds a series only until it is reconstructed; this is what survives the scan.

A proxy started for one scan outlives whoever started it, so `--idle-timeout`
is how it ends: it closes once nothing has been connected for that long,
counted from the last client leaving, never while a series is still being
reconstructed. Without one it serves until it is signalled.

A reconstruction computer runs one proxy per acquisition, so the series of one
exam are reconstructed by different processes. What a calibration series
measures is kept under `--exams`, in a directory of the exam's own, and read
there by a later series whatever proxy takes it: a hook assigns
`context.b0_map`, `context.b1_map` or `context.coil_sensitivities` and a later
hook reads them ({class}`~pulserver.recon.ReconContext`). An exam's directory is removed
once the last proxy on that exam has moved to another, so a proxy prescribing
the next exam does not take the artifacts from one still reconstructing the
last.

With `--forward`, the proxy runs no workers: each series is enriched as it
arrives and sent on to the MRD server at `HOST:PORT`, whose own slots and queue
determine when it is reconstructed. The server receives a config file message naming the series'
reconstruction plugin, which a reconstruction server runs, or the
`--forward-config` name, such as the configuration a Gadgetron or a FIRE server
selects its pipeline by. The client's config text is not forwarded. What the
server returns is relayed to the client; a message the proxy has no reader for
ends the relay, and the client receives a `pulserver:` text naming its type.

Whether forwarded or reconstructed by a worker, the series carries, after its
header and before its first acquisition, one `TEXT` message (5) describing the
sequence chain to a simulation: a JSON object under the key
`pulserver_sequence_description`, holding per file of the chain one row per
block, RF at the pulse centre and ADC at the sample nearest the centre of
k-space, and the pulses the RF rows name. It is meant for a reconstruction
server that simulates the signal, which chooses the signal model the rows
drive, and is not relayed back to the client.

With `--intake-port`, the proxy runs a design intake beside it
({class}`~pulserver.proxy.DesignIntake`), an HTTP endpoint that writes the
designs pushed to it into `--store`: `HEAD /designs/<id>` answers 200 when the
store holds the design and 404 otherwise, and `PUT /designs/<id>` stores a
bundle, answering 201, 200 for a design already stored, or 400 with the reason
for a bundle that is not the design its path names. Asking for a design the
store holds marks it as used for `prune`.

The MRD stream is neither authenticated nor encrypted, and its header carries
patient data. The design intake is neither authenticated nor encrypted either,
and writes the designs it accepts into the store the proxy reconstructs from.
Both belong on the network between the scanner and the reconstruction
computer, with `--host` naming the address of the interface on it; a scanner's
reconstruction client cannot reach the loopback default.

The proxy and the warm design server stop on `SIGINT` or `SIGTERM`. The proxy
waits for the series it is running before it exits.

## Reconstruction server

```bash
pulserver recon --plugins DIR --port N [--host ADDR] [--queue DIR] [--slots N] [--gpu-slots 1] [--spares 1] [--recon-timeout S] [--save-data DIR] [--idle-timeout S] [--logfile FILE] [--log-level LEVEL]
```

The reconstruction server is what a forwarding proxy sends its series to, on
the computer that reconstructs them. It reconstructs each series it receives
with the plugin its config names, as a config file message or as a config text
naming it as a bare name or under `parameters.config`, and enriches nothing:
the series a proxy forwards arrive enriched. Its options are the proxy's, with
the same workers, slots, queue and exam directories, and `--save-data`,
`--idle-timeout`, `--logfile` and `--log-level` behave as they do there; what
it keeps is the series as the proxy forwarded it, enriched. Its stream is neither
authenticated nor encrypted, and `--host` names the interface the proxy
reaches it on. It stops on `SIGINT` or `SIGTERM` and waits for the series it is
running.

## Checking a sequence plays as it was written

```bash
pulserver validate SEQ [--vendor ge] [--played FILE] [--dead-time-us N] [--rf-wait-us N] [--tolerance MT] [--rf-tolerance PCT] [--limits FILE]
```

| Option | Meaning |
| --- | --- |
| `SEQ` | The sequence file to check |
| `--vendor` | Which machine the recording came from. Without one the check is against the sequence's own IR cache |
| `--played` | The recording to check against; without one, and with a vendor named, the recording is asked for from that vendor's tooling if it is installed |
| `--dead-time-us` | What the recording's times run ahead of the sequence's by, on every channel |
| `--rf-wait-us` | What the transmit channels run ahead by on top of that |
| `--tolerance` | The largest gradient difference that counts as agreement, in mT/m; taken from the slew rate and raster of `--limits` when unset |
| `--rf-tolerance` | The largest transmit difference, as a percentage of the peak; 1 by default |
| `--limits` | The `[Limits]` block, for a sequence that has no cache yet |

The command exits 0 where the two agree, 1 where they differ, and 2 where it
could not compare, and prints the largest difference on each gradient axis
against the peak that axis reaches.

Naming a vendor compares against a recording of a machine playing the sequence,
which establishes that the machine plays what was written. Naming none compares
against the waveforms the sequence's own IR cache holds, which establishes only
that the conversion kept the sequence — a conversion and a playout that are
wrong in the same way agree with each other. The output says which of the two
ran.

The gradients are compared in millitesla per metre, and the transmit channels
as the recording stores them: the phase over a converter that spans a turn, and
the magnitude by its shape, because what turns a recorded count into hertz is
the peak transmit field of the scan being played and the recording does not
carry it. The scale that the shape took is printed, which is the number a
machine's own calibration would have to supply.

A machine starts a unit before it plays anything of it, and drives its transmit
and gradient channels on separate timelines, so a recording runs ahead of the
sequence by one constant on every channel and by a second on the transmit
channels. `--dead-time-us` and `--rf-wait-us` state them. Nothing estimates
them, because a comparison that quietly aligns two waveforms can align away the
disagreement it exists to find.

The gradient tolerance is the hardware's: what the gradients can slew through
in the few raster steps the two renderings may be apart. A fixed number would
be either too tight for a fast machine or too loose for a careful one.

Checking a sequence leaves nothing beside it: where no cache is there already,
one is built beside a copy and discarded.

## See also

* {doc}`../explanations/architecture` — the services and what passes between them.
* {doc}`../explanations/designs` — the design store and the identity of a design.
* {doc}`reconstruction-client` — the MRD stream of a series.
* {doc}`../api/host` and {doc}`../api/proxy` — the design calls, the proxy and the reconstruction server.
