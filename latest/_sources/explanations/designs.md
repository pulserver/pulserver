# Designs and the design store

```{admonition} TL;DR
:class: tldr

- A design is identified by the SHA-256 of the plugin, the code it designs
  with, the limits and the resolved protocol, and a request that resolves to a
  stored design returns it.
- The design store holds one directory per design, written whole and never
  modified, and pruned least recently used first.
- A design is checked under the scanner limits in the physical frame of the
  prescription before it is stored, and the cache records each subsequence's
  SAR against a reference pulse.
- The reconstruction side finds a design by the identifier the raw-data
  header carries.
```

A scanner sequence is validated on every protocol edit and generated when the
scan is prepared, often many times for one protocol, and a generated design
has to remain available, unchanged, for as long as data acquired with it may
be reconstructed. Pulserver stores each design by its content: a design is
identified by what it depends on, and a request that resolves to a stored
design returns it instead of designing again. No design call keeps state
between calls; the design store is all that persists.

## Design identity

A generated design depends on the scanner-sequence plugin, the code that
designs with it, the scanner limits and the resolved protocol. Its identity is
the SHA-256 hash of these ({func}`~pulserver.host.design_identity`):

- the plugin name;
- a digest of the plugin file, the source file of the sequence function it binds,
  and the installed versions of pypulseqpp and pulserver;
- the scanner limits, design limits, conversion options and check limits,
  with the contents of the VOP file they name;
- the resolved protocol, prescription included.

A changed plugin, sequence function or package gives a new design for an unchanged
protocol, and so does a VOP file rewritten at the same path. Because the hash
is computed over the resolved protocol, two requests that differ only in a
value the design replaces, such as a preset and the time it resolves to,
identify the same design ({doc}`protocol`). A design is made from its resolved
protocol: a request that resolves to itself, such as the value block of a
`validate` reply sent back, is designed as it stands, and one that resolves to
other values is designed from those, so the files of a design are a function of
its identity.

An imported chain is identified by the names and contents of its files, the
limits and the prescription of the import, offset and rotation.

## Design identifier

The identifier of a design is the first 18 hexadecimal digits of its identity,
72 bits ({func}`~pulserver.host.design_id`). The interpreter records the
numbers it passes to the reconstruction client in float32 fields of the
raw-data header, and a float32 represents every integer below $2^{24}$
exactly, so the identifier takes three of them, of six digits each. The store
refuses a design whose identifier already holds another identity.

## The store

The store is a directory with one subdirectory per design:

```text
designs/
  3f9c0a1b2c3d4e5f60/
    sequence.seq          first file of the NextSequence chain, binary Pulseq
    sequence_main.seq     the remaining files, when there are prescans
    sequence.pseg         IR cache
    resolved.protocol
    manifest.json         identity and identifier, limits, package
                          versions, creation time, plugin and source,
                          scan time, field-of-view offset (and rotation,
                          for an imported chain), SHA-256 of every file
```

A design is written into a staging directory in the store and renamed into
place once complete, so a reader sees it whole or not at all, and two processes
that write one design leave one directory. A design is not modified afterwards.
Designs are removed only by pruning ({doc}`../user-guide/running`), least
recently used first; a design is used when it is written or found again.

## Checks before a design is stored

The gradient coils are limited per physical axis, and the peak a gradient
reaches on one axis depends on the orientation it is played in: two logical
axes at 0.8 of the amplitude limit each put $0.8\sqrt{2}$ of it on one physical
axis at 45°. So the checks are made in the physical frame. The prescription's
rotation $R$, from logical to physical axes, reaches the host in the nine
`fov_rotation_ij` entries of the protocol, element $(i, j)$ of $R$, and each
file is rotated by it as the scanner plays it: composed after each block's own
rotation, with blocks labelled `NOROT` left unrotated. A prescription with a
reflection in it is checked as it plays, reflection included. A design is held
under limits per logical axis that the scanner derates for the rotation, the
`design_max_grad` and `design_max_slew` of the call's limits, and its physical
axes are checked against the gradient coils' own, `max_grad` and `max_slew`
({doc}`../user-guide/running`).

Before a chain is converted, {func}`~pulserver.ir.check` runs pypulseqpp's
timing check, gradient continuity included, and its gradient amplitude and
slew-rate checks on every file, against the gradient limits, dead times and
ringdown time of the scanner. Where the call's limits carry them
({class}`~pulserver.ir.CheckLimits`), it also runs pypulseqpp's PNS check
under the scanner's nerve model and its mechanical-resonance check against the
scanner's forbidden gradient bands. The waveforms are timed by the rasters the
file declares. The interpreter passes these limits with every design call
({doc}`../user-guide/running`); it computes the SAR and the gradient heating.

No design is stored for a generated design or an imported chain that fails a
check: `generate` and `import` reply with the problems, as they do with a
design error. The checks compute estimates; passing them does not
establish scanner or patient safety.

## SAR against a reference pulse

The interpreter computes SAR under its own calibration of the transmit chain,
which covers a pulse played in the coil's default channel weights. Where local
SAR is computed from virtual observation points (VOPs), the host relates every
pulse of a subsequence to such a pulse, the reference: hard, 180° and 1 ms, in
the default channel weights. The energy a pulse deposits at VOP $v$ is
$\int \mathbf{b}(t)^H Q_v\, \mathbf{b}(t)\,dt$, with $\mathbf{b}$ the drive of
each transmit channel and $Q_v$ the VOP's matrix, and in the head of body model
$b$ the same integral with that model's head SAR matrix $G_b$. For each
repetition $w$ of a subsequence, the blocks before the first repetition and
after the last included, as pypulseqpp's SAR check averages over them,
{func}`~pulserver.ir.sar_ratios` computes

$$
r_{\mathrm{local}} = \frac{L_{\mathrm{head}}}{L_{\mathrm{local}}} \max_w
\frac{M \max_v E_{v,w}}{N_w \min_b E^{\mathrm{ref}}_{G_b}},
\qquad
r_{\mathrm{head}} = \max_w \max_b \frac{E_{G_b,w}}{N_w\, E^{\mathrm{ref}}_{G_b}},
$$

with $E_{v,w}$ and $E_{G_b,w}$ the energy of the repetition at VOP $v$ and in
the head of body model $b$, $N_w$ the number of pulses it plays,
$E^{\mathrm{ref}}_{G_b}$ the head energy of one reference pulse, $M$ the VOP
file's safety factor, and $L_{\mathrm{head}}$ and $L_{\mathrm{local}}$ the
scanner's head and local SAR limits, `vop_head_limit` and `vop_local_limit`.

The interpreter gives the reference pulse the shortest time its calibration
allows, at which the reference's head SAR reaches $L_{\mathrm{head}}$. That
fixes the drive scale without a measurement of transmitted power: at that
scale, a pulse of the subsequence given $\max(r_{\mathrm{local}},
r_{\mathrm{head}})$ times the reference pulse's time deposits at most
$L_{\mathrm{local}}$ of peak local SAR and $L_{\mathrm{head}}$ of head SAR. The
smallest reference energy over the body models sets the largest drive scale, so
the local term holds in every model; the head term is taken body model by body
model, since a subject's head SAR is that of one body. The bound rests on the
interpreter's head SAR for the reference not falling below the true one, and on
the VOPs and the safety factor bounding peak local SAR; the reference's own
local SAR does not enter. A 1 ms hard pulse of 90° counts a quarter of the
reference in both terms, and a repetition of reference pulses has a head term
of 1. A scale common to every channel's drive and to the matrices cancels in
both terms; the relative channel gains do not.

The cache carries the two terms of each subsequence in its
`pulseg_subseq_info`, zero without VOPs or without RF, and the interpreter
charges each pulse of the subsequence the time of a reference pulse times the
larger of the two.

## Access from the reconstruction side

The raw-data header of a series names its design in the `pulserver_design`
user parameter ({doc}`../user-guide/reconstruction-client`). A design is
immutable, so the reconstruction proxy reads and tabulates it once and reuses
the result for later series acquired with it, keeping the designs it read most
recently ({class}`~pulserver.proxy.DesignCache`).

When the design calls and the proxy run on different computers, the proxy
reads either a store both can reach or a store of its own, which its design
intake fills with the designs the calls push ({class}`~pulserver.proxy.DesignIntake`).
A pushed design is carried as a bundle of the files of its directory, and the
intake stores it only when the manifest's identity is a SHA-256, its identifier
is that of the identity and the one the push names, and every file has the
SHA-256 the manifest records, so the two stores hold the same files under the
same identifier. A design is pushed before its identifier is replied, and a
design that cannot be pushed fails the call, so the interpreter plays no design
the proxy's store lacks.

## See also

* {doc}`../user-guide/running` — the design calls, the warm server, pushing and pruning.
* {doc}`../api/host` — the design calls and the store.
* {doc}`protocol` — the resolution of a request into the protocol a design plays.
