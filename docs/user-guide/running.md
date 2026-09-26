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
pulserver design list     --plugins DIR --plugin NAME
pulserver design validate --plugins DIR --plugin NAME --limits FILE < VALUES
pulserver design generate --plugins DIR --plugin NAME --limits FILE --store DIR [--push URL] < VALUES
pulserver design import   --limits FILE --store DIR [--push URL] < IMPORT
pulserver design push     --store DIR --to URL ID...
pulserver design prune    --store DIR [--max-age-days D] [--max-bytes B]
```

Each call is answered from its arguments and standard input alone and writes
its reply to standard output; `python -m pulserver.host` takes the same
arguments. `--plugins` is the directory of scanner-sequence plugin files,
`<plugin>.py`, and `--store` the design store, created when missing.

| Call | Reply |
| --- | --- |
| `list` | `PROTOCOL` and the plugin's listing block |
| `validate` | `VALID <seconds>` or `INVALID`, an `INFO` line and the value block of the resolved protocol |
| `generate` | `GENERATED <id>`: the identifier of the design in the store |
| `import` | `IMPORTED <id>` |
| `push` | `PUSHED <count>`: the designs sent, not counting those the intake holds already |
| `prune` | `PRUNED <count>` |

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
Three further options are for the IR conversion:

| Option | Meaning |
| --- | --- |
| `ir_vendor` | `PULSEG_VENDOR_*` code the cache is tagged with; 0 is vendor-neutral |
| `ir_label_column_map` | Three Pulseq label state indices, separated by spaces, filling the ADC label columns |
| `ir_cache_ext` | Extension of the cache file, `.pseg` by default |

A generated design and an imported chain are checked against the limits before
their IR cache is written ({doc}`../explanations/ir-cache`), so the limits
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
({doc}`../explanations/ir-cache`), and the interpreter computes the SAR and
the gradient heating.

| Limit | Meaning |
| --- | --- |
| `pns_chronaxie`, `pns_rheobase`, `pns_alpha` | Chronaxie nerve model: chronaxie in s, rheobase in T/m/s, `alpha` 1 when left out |
| `pns_<axis>_<field>` | SAFE nerve model, for the axes `x`, `y` and `z` and the fields `a1` to `a3`, `tau1` to `tau3` in ms, `stim_limit` in T/m/s and `g_scale` |
| `pns_limit` | Largest PNS response allowed, as a fraction of the model's threshold; 1 when left out |
| `forbidden_band_<n>` | One forbidden band: its physical axis (`x`, `y`, `z` or `all`), its lowest and highest frequency in Hz and, optionally, the largest amplitude allowed in it in mT/m, separated by spaces |
| `vop_file` | `.mat` or `.npz` file of VOPs and, optionally, a global SAR matrix, at a path the design calls can read |
| `vop_drive_per_hz` | Channel drive per Hz of RF amplitude, one per channel separated by spaces; a scale common to every channel cancels, and equal drives when left out |
| `vop_default_shim` | Magnitude and phase in rad of each channel's weight for a pulse played without an RF shim, separated by spaces; equal weights when left out |

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
block = "[NimPulseqGUI Protocol]\nTE: 5000\nnx: 96\n[NimPulseqGUI Protocol End]\n"
status, reply = call("validate", plugins="sequences", plugin="gre2d",
                     limits=limits, block=block)
status, reply = call("generate", plugins="sequences", plugin="gre2d",
                     limits=limits, block=block, store=DesignStore("designs"))
```

A plugin file is imported again when its modification time changes.

### Design store

The store holds one directory per design, `<store>/<id>/`: the Pulseq files of
the chain, the IR cache, the resolved protocol and `manifest.json`, which
records the plugin, the reconstruction plugin, the limits, the package versions
and the SHA-256 of every file. The identifier is three 24-bit integers, each
held exactly by a float32 scanner parameter. `prune` removes designs least recently used
first: those unused for longer than `--max-age-days`, then others until the
store holds at most `--max-bytes`. Nothing is removed otherwise, and a design a
series still needs must not be: the proxy reads it by identifier. A store that
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

`list` depends on the plugin file and the installed packages only, so its reply
can be written when they are installed and read without a call.

### Warm server

```bash
pulserver design serve --plugins DIR --socket PATH
```

A call answered in its own process imports the design engine first, which
takes most of its time. A warm server imports it once, designs, checks and
converts a sequence of its own, and imports every plugin in `--plugins`. It then
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

## Reconstruction proxy

```bash
python -m pulserver.proxy --store DIR --port N --plugins DIR [--host ADDR] [--intake-port N] [--queue DIR] [--slots N] [--gpu-slots 1] [--spares 1] [--recon-timeout S]
python -m pulserver.proxy --store DIR --port N --forward HOST:PORT [--forward-config NAME] [--forward-dicom] [--host ADDR] [--intake-port N] [--recon-timeout S]
```

| Option | Meaning |
| --- | --- |
| `--store` | The design store the proxy reads: the one the design calls write, or the one the intake writes |
| `--port` | TCP port the scanner's reconstruction client connects to |
| `--host` | Address to listen on; the loopback interface when unset, `0.0.0.0` for every interface |
| `--intake-port` | TCP port of the design intake, on the `--host` address; no intake when unset |
| `--plugins` | Directory of reconstruction plugin files, `<plugin>.py`; required unless `--forward` is given |
| `--queue` | Directory the series waiting for a slot are written to; a temporary directory, removed when the proxy stops, when unset |
| `--slots` | Series reconstructed at once; derived from available memory and the GPUs when unset |
| `--gpu-slots` | Series reconstructed at once on each GPU when `--slots` is unset, default 1 |
| `--spares` | Worker processes started ahead of a series, default 1 |
| `--recon-timeout` | Seconds a reconstruction may run after its series ends; the worker is then terminated, or the connection to the server closed, and the client told. Unlimited when unset |
| `--forward` | `HOST:PORT` of the MRD server that reconstructs every series, instead of local workers |
| `--forward-config` | Config name sent to the `--forward` server; the series' reconstruction plugin when unset |
| `--forward-dicom` | Convert each image the `--forward` server returns to DICOM before it is relayed |

The MRD header of each series names the design it was played from in the
`pulserver_design` user parameter, and the proxy refuses a series whose header
names no stored design. The reconstruction
plugin is the one the scanner sequence names in its `recon` attribute; when it
names none, the client's config text names it, either as a bare plugin name or
under `parameters.config`. The messages and fields the client sends are listed
in {doc}`reconstruction-client`.

A series that finds every slot busy is written to the queue directory as it
arrives and reconstructed once a slot frees; the client stays connected
meanwhile.

With `--forward`, the proxy runs no workers: each series is enriched as it
arrives and sent on to the MRD server at `HOST:PORT`, whose own slots and queue
determine when it is reconstructed. The server receives a config file message naming the series'
reconstruction plugin, which a reconstruction server runs, or the
`--forward-config` name, such as the configuration a Gadgetron or a FIRE server
selects its pipeline by. The client's config text is not forwarded. What the
server returns is relayed to the client; a message the proxy has no reader for
ends the relay, and the client receives a `pulserver:` text naming its type.

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
python -m pulserver.recon --plugins DIR --port N [--host ADDR] [--queue DIR] [--slots N] [--gpu-slots 1] [--spares 1] [--recon-timeout S]
```

The reconstruction server is what a forwarding proxy sends its series to, on
the computer that reconstructs them. It reconstructs each series it receives
with the plugin its config names, as a config file message or as a config text
naming it as a bare name or under `parameters.config`, and enriches nothing:
the series a proxy forwards arrive enriched. Its options are the proxy's, with
the same workers, slots, queue and exam directories. Its stream is neither
authenticated nor encrypted, and `--host` names the interface the proxy
reaches it on. It stops on `SIGINT` or `SIGTERM` and waits for the series it is
running.

## See also

* {doc}`../explanations/architecture` — the services and what passes between them.
* {doc}`../explanations/designs` — the design store and the identity of a design.
* {doc}`reconstruction-client` — the MRD stream of a series.
* {doc}`../api/host` and {doc}`../api/proxy` — the design calls, the proxy and the reconstruction server.
