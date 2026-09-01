# Applying a prescription

```{admonition} TL;DR
:class: tip

- A prescription change is a rotation, a scaling, and a translation that is a
  phase. Measured on 2 103 300-block MPRAGE protocols it costs **1.2–1.7 s**,
  under a microsecond per block.
- **The running phase is one pass, not a walk.** `block_k_origins` gives $k$ at
  the start of every block for the whole sequence, so nothing is decoded into a
  struct and registered again.
- **The work is memoized** on what a block *plays*, not on where it plays it.
  The block's entry point $\Delta r\cdot k_0$ cancels out of every shape that
  gets stored, so one phase-encode table plans one readout once.
- **A non-Cartesian readout stores its base trajectory**, not a phase per
  sample: one row per distinct trajectory instead of one array per readout.
  Marked in `[DEFINITIONS] PhaseModulationMode`, opt-out via `compat=True`.
- **$k_0$ is never stored in the file.** Where the shift is baked it is folded
  into the ADC's `phase_offset` at design time; where it is deferred, the
  consumer recovers it with one `block_k_origins` pass when it parses the
  seqfile, and carries three floats per readout — because it needs absolute $k$
  for the trajectory it reports anyway, shift or no shift.
```

An operator angles the slab, slides it off isocentre and changes the field of
view, and the console expects the answer before it redraws. What the
prescription means for a written sequence is
{doc}`Pulseq's own transformation <../background/fov_transformation>`. This page
is what it costs here, and what it leaves in the file.

## Two costs the reference implementation cannot avoid

**The walk.** `applyToSeq` walks the sequence with `getBlock` and rebuilds it
with `addBlock`, and the running phase means the walk cannot be reordered or
split: block $n$ needs the accumulated $k$ that transforming every earlier block
produced. A million blocks is a million round trips through the event libraries,
in Python.

**The file.** Where the gradient is not flat across the ADC window, the shift
cannot be two scalars — it has to be given sample by sample, in
`adc.phaseModulation`, one array of `num_samples` per readout. That is fine for
one spiral. It is not fine for a stack of spirals: every arm carries its own
angle and its own entry point, so no two arrays are equal, and a scan with a few
hundred arms per slab writes a few hundred unequal arrays of a few thousand
numbers each. Deduplication cannot help, because there is nothing repeated. The
shift has turned a scan whose readouts were *one* stored waveform into a file
with one stored array per readout, and the growth lands on the `.seq` size, the
parse time and the resident representation alike.

Ramp-sampled EPI is the exception that shows the rule. Every readout of an EPI
train is the same waveform at the same amplitude, so every readout gets the same
phase array and they collapse to one row. It is precisely the families where the
readout *varies* — spirals, cones, radial with ramp sampling, florets — that the
phase-per-sample form penalises, and those had the least redundancy to spare.

## The running phase is one pass

`pulseq::block_k_origins` returns $k$ at the start of every block, reset by an
excitation and negated by a refocusing pulse, for the whole sequence in one
pass. That *is* the running phase, written as a k-space position rather than as
an accumulated scalar. With it in hand, applying a shift needs nothing from
Python but the block range — no `getBlock`, no `addBlock`, no event decoded into
a struct and registered again.

The labels work the same way. `NOSCL`, `NOPOS` and `NOROT` are read as **runs**
— `label_gate_runs` returns the block ranges each label leaves alone — so a scan
with no exemptions is one call per transformation and a scan with a
fat-saturation module is three or four. Nothing here is ever a Python loop over
the blocks of a scan.

`apply_to_sequence` calls `remove_duplicates` first, so everything below costs
the size of the *libraries* rather than the length of the scan. It is a no-op on
a sequence already deduplicated.

## A scaling and a rotation touch no samples

A gradient is stored as a normalised shape beside a scalar amplitude, so scaling
multiplies the amplitude and leaves the waveform alone. No shape is registered,
and two gradients that differed only in amplitude still share their shape
afterwards.

Rotation is attached as a `ROTATIONS` extension rather than baked into new
gradients, so one waveform serves every orientation. Only
`use_rotation_extension=True` is supported: baking rotations is what makes a
large non-Cartesian scan unaffordable, and the exemption labels exist precisely
so a module that must *not* be rotated again downstream can say so.

## One plan per distinct thing a block plays

The shift's phase at time $t$ inside a block splits in two:

$$
\varphi(t) \;=\; \underbrace{\Delta r \cdot k_0}_{\text{where the block sits}}
\;+\; \underbrace{\Delta r \cdot \!\int_0^t \! G(\tau)\,\mathrm{d}\tau}_{\text{what the block plays}}
$$

Everything the transformation *stores* — the phase shape an RF pulse carries,
the residual an ADC carries — is a **difference** of $\varphi$ against its value
at a reference time, so the first term cancels out of those shapes exactly. What
survives depends only on the gradient rows and the event timings.

```{figure} ../assets/transform_fov/phase_split.png
The phase a shift adds, before and after the block's entry point is taken out of
it.
```

So the work is memoized on a key of canonical event ids — the three gradient
rows the block plays, plus the RF or ADC row the plan is for. A phase-encode
table plays one readout ten thousand times at ten thousand different entry
points; without the memo each decompresses the same waveform, re-derives the
same residual sample by sample, and registers its own copy of the same shape,
which the writer then deduplicates back down to the one row it always was. With
it, the readout is planned once. Gradient waveforms are built lazily and only
when a plan has to be worked out at all, so a scan whose readouts repeat
decompresses a spiral arm a handful of times rather than two million.

Subtracting the two whole phases instead of splitting them is the same number in
exact arithmetic and a worse one in floating point: $\Delta r \cdot k_0$ reaches
hundreds of cycles at the edge of k-space while the residual is of order a
milliradian, so cancelling them numerically costs about as many digits as the
answer has.

Measured on the three MPRAGE protocols of {doc}`sequence_creation` — 2 103 300
blocks each — a prescription change is **1.2–1.7 s**.

## What a non-Cartesian readout stores

```{figure} ../assets/transform_fov/base_trajectory.png
Six readouts that differ only by rotation. Stored as a phase per sample they are
six unequal arrays; stored as the trajectory before the angle and the amplitude
are applied, they are one row that all six point at.
```

Under `compat=False` the readout stores its **base trajectory** instead of a
phase: the k-space curve the block's own gradient waveforms span *before* their
per-instance amplitudes and rotation are applied, normalised by the ADC window
duration. It starts at zero at the start of the block — it is the block's own
moment, and carries no history. A consumer turns it into a k-space position in
three steps:

1. **Undo the normalisation and scale by the instance amplitude.** Both factors
   are already in the ADC row, so

   $$k_a(t) \;=\; k_{0,a} \;+\; A_a \cdot n_\text{samples}\,\Delta t \cdot b_a(t)$$

   with $A_a$ the amplitude the block's gradient row carries on axis $a$ and
   $k_{0,a}$ the block's k origin. `read_base_trajectory` performs the
   $n_\text{samples}\Delta t$ half on the way out, so a C++ consumer holds a
   base already in seconds and multiplies only by the amplitude.
2. **Turn by the block's rotation.** The `ROTATIONS` quaternion is applied to
   the three-vector, which is what makes one row serve every angle.
3. **Dot with the prescribed offset.** $\Delta r \cdot k(t)$, sample by sample,
   is the phase — the same array a baked shift would have written into
   `adc.phaseModulation`, recovered rather than stored.

What collapses is the storage. Readouts differing only by rotation share one row
outright, and the ones that also scale share it too, because the amplitude is a
relative factor that survives the unit conversion. One row per *distinct
trajectory* replaces one array per *readout*.

Three details make that affordable:

**Three axes, always.** An axis flat across the window is stored flat rather
than omitted, so nothing has to be inferred from a length — a two-column shape
cannot distinguish $(x, y)$ from $(x, z)$.

**Axis-major, not interleaved.** The shape codec run-length encodes a
*derivative*. Laid out as all of $x$, then all of $y$, then all of $z$, a flat
axis costs three numbers; interleaved, three axes alternate every sample and
compress to nothing. That compression is the entire reason storing three axes is
affordable.

**Normalised by the window.** `compress_shape` quantises onto an absolute
$10^{-7}$ grid. Left in seconds — order $10^{-4}$, stepping by $10^{-5}$ — a
readout would have about a hundred levels across it. Dividing by
$n_\text{samples} \Delta t$ puts the samples at order one.

(k-origin)=
## Where $k_0$ lives

$k_0$ is the moment accumulated in *earlier* blocks since the last excitation or
refocusing pulse — the prewinder, the phase-encode lobe, the blips before this
one. It is not part of the readout's own block, so the stored base does not
contain it, and it is **not written into the file** at all. Where it appears
depends on whether the readout's shift was baked or deferred, and those two
answers are different.

**A baked readout carries it in `phase_offset`.** For a Cartesian, unrotated
readout the gradient is flat across the ADC window, so the whole shift is two
scalars: `apply_fov_shift` writes the slope into the ADC row's `freq_offset` and
the constant — which is exactly $\Delta r \cdot k_0$ plus the phase the shift has
reached at the window centre — into its `phase_offset`. That is a per-readout
number in the ADC library, computed at design time; the PSD plays it as the
receiver phase like any other ADC phase offset, and nothing downstream has to
know a shift happened. This is the whole of a Cartesian phase-encode-direction
shift: along the phase-encode axis $k$ is constant across the window, so its
shift phase *is* the constant term.

**A deferred readout carries nothing, and the consumer recovers it.** A readout
whose gradient moves across the ADC window has a phase no frequency describes.
A readout on a block carrying a `ROTATIONS` extension is a narrower case, and
worth stating exactly, because the obvious reason is the wrong one. Writing
$R$ for the extension and $R_\text{fov}$ for the prescribed rotation,

$$\varphi = \Delta r_\text{phys}\cdot R_\text{fov} R\, k_\text{log}
 = \underbrace{(R_\text{fov}^{\mathsf T}\Delta r_\text{phys})}_{\Delta r_\text{log}}
   \cdot\, (R\, k_\text{log}),$$

so the *prescribed* rotation cancels — that is what a logical-frame shift buys —
but $R$ does not. $R$ is in the file, so the phase is perfectly knowable at
design time. What defers the readout is that `apply_fov_shift` works in the
**unrotated** logical frame throughout, so the number it could compute is
$\Delta r_\text{log}\cdot k_\text{log}$, which is the wrong one. The guard keeps
that out of the file; the consumer, which composes $R$ when it builds the
trajectory, gets it right.

Either way those readouts get no `freq_offset`, no `phase_offset` and no
residual, and are marked as deferring. The consumer then runs the three steps
above against the centre that was finally prescribed, and $k_{0,a}$ has to come
from somewhere:

`pulseq::block_k_origins` is that somewhere, and the consumer runs it itself.
`read_sequence_files` walks the `NextSequence` chain — the same `.seq` text or
binary the scanner played, and nothing else — and the origins come out of that
walk, stored per readout as three floats in `TrajTableEntry::k_origin`.
`compose_entry_rows` then writes $k = k_0 + A\cdot b$ into the readout's
trajectory before `enrich_ismrmrd_acquisition` rotates it and attaches it to the
acquisition, so by the time `demodulate_fov_shift` multiplies by $\Delta r$ the
origin is already in the numbers it multiplies. Two triples of scalars — the
amplitudes and the origin — are what make a readout its own; everything else it
shares.

That is one pass over the parsed sequence, done once when the scan is prepared.
It reads no scanner-written sidecar: the `.pge` cache is the interpreter's,
written by the PSD for pulse generation and the scan loop, and the
reconstruction side does not open it.

### Why the origin is not in the stored row

The obvious simplification is to store $k_0 + A\,n_\text{samples}\Delta t\,b(t)$
and be done with it. Two properties of the row stop it, and the first is not a
trade-off but an arithmetic impossibility.

**There is no amplitude to fold it into.** The row is stored amplitude-free —
that is what lets interleaves differing only in gradient amplitude share it — so
expressing an origin inside it means storing
$b + k_0/(A\,n_\text{samples}\Delta t)$. And the origin lives, precisely, on the
axis the readout does *not* drive: a stack-of-spirals readout carries its whole
$k_z$ in the origin while its $z$ gradient amplitude is zero for the entire
window. $A = 0$ on the axis that needs it, so there is nothing to divide by. The
zero row and the origin are complementary by construction, which is why the
reader interns one zero row for the whole scan and puts the constant beside it.

**The shape is one row, not three.** All of $x$, then $y$, then $z$ go into a
single registered shape, so an origin that varies on *any* axis makes the whole
three-axis row distinct. A stack of spirals stores one base shape for the entire
scan however many arms and partitions it has; baking the origin in would make
that one shape per partition — a full copy of the arm to carry one scalar, and
growing with the partition count.

Both are properties of this layout rather than laws, and either could be
traded away: store absolute $k$ in 1/m and the amplitude factorisation goes with
it, or split the shape per axis and the granularity improves. Neither buys
anything here, because the origin is three floats the consumer already has.

The origin term does **not** drop out of a deferred shift. What it drops out of
is the *stored shapes*: the residual and the RF phase shape are differences
against a reference time, which is what makes the plan memoizable. A shift phase
that omitted $\Delta r\cdot k_0$ would be referenced to each readout's own
entry point rather than to a common origin, and the readouts would be
inconsistent with each other.

```{note}
Nothing forces $\Delta r\cdot k_0$ onto the consumer. It is a design-time
constant even for a deferred readout — $\Delta r$ is logical and known when the
file is written, and $k_0$ composed with the earlier blocks' rotations is a
property of the sequence — so it could be folded into that readout's ADC
`phase_offset` and only the swept term left deferred. It is not, because doing
so would save the consumer nothing: the origin is needed for the **trajectory
the consumer reports**, shift or no shift, which is the second job below.
Making the consumer origin-free is a question about shipping the origin in the
file, not about where the shift's constant is applied.
```

## One row, two jobs

The base trajectory is written whenever `compat` is cleared — shift or no shift,
and a scan prescribed at isocentre gets it too. The row answers two questions,
and only one involves a prescription.

```{figure} ../assets/transform_fov/two_jobs.png
The stored row, and the two things read off it. Steps 1 and 2 are the same on
both paths; what differs is what is done with the result.
```

**Finishing a deferred shift**, as above.

**Reporting the k-space.** A non-Cartesian reconstruction needs the readout's
$k$ whether or not the volume was ever moved, and steps 1 and 2 give it
directly. This is the job that makes the origin load-bearing: a stored base is
block-relative by construction — that is what lets one row serve every readout
that plays it — so absolute $k$ needs the origin from somewhere, and a shift is
not what puts it there. This is what lets a client
{doc}`enrich an acquisition <../sequence_model/mrd_architecture>` by reading the
sequence file rather than re-deriving anything: the trajectory it attaches per
readout is composed from a stored row, an amplitude, an origin and a quaternion,
and no gradient waveform is integrated at acquisition time. The offline route —
`calculate_kspace()` over the whole sequence — computes the same numbers and is
the right tool away from the scanner, but it is a pass over the scan rather than
three arithmetic steps per readout, and a streaming client has neither the time
nor the sequence object.

That second job is why the row is not conditional on a shift. Gating it on one
left a spiral prescribed at isocentre with no trajectory for the reconstruction
to use.

## What the overload costs

`phase_modulation` is the only field of the right shape, and putting a
trajectory in it is an overload of a field the format defines as a phase. A
toolbox that applies it literally gets a plausible wrong image rather than an
error — so a file that carries base trajectories *says* so, in
`[DEFINITIONS] PhaseModulationMode`. Unknown definitions are ignorable, so the
marker costs nothing to a reader that does not care.

Compatibility with a stock Pulseq reconstruction is lost for exactly those
readouts, and it is recovered two ways: the overload is opt-out, so
`compat=True` bakes every shift the way the reference does and the file needs
nothing downstream; and a consumer that wants both can read the marker and
rescale, which is the same arithmetic it would have done anyway to turn a phase
back into a position.
