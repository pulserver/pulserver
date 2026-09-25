# The scanner IR

A Pulseq file lists every block of a scan in play order, with each event stored
once in a library and referenced by id. A scanner's pulse generator is
programmed differently: waveform memory is allocated once per distinct
waveform, and a scan is played as repetitions of a small number of instruction
sequences, with amplitudes, phases and rotations updated between repetitions.
Pulserver computes this intermediate representation (IR) of a sequence on the
host and writes it into a binary cache beside the sequence file. The
interpreter loads the cache rather than parsing the Pulseq file.

## Subsequences

{func}`~pulserver.ir.convert` reads the `NextSequence` chain starting at a
sequence file, text or binary, with `pypulseqpp.Sequence`. Each file of the
chain is a subsequence of one scan, played in order; a prescan and the imaging
sequence are typically two subsequences. Each subsequence is analysed on its
own and the results are chained into one collection.

## Prescription

A sequence application writes its files in the logical frame, about the
isocentre. The prescribed field-of-view offset $\mathbf{d}$ reaches the host in
the `fov_offset_x`, `fov_offset_y` and `fov_offset_z` entries of the protocol,
in mm along the logical readout, phase-encoding and slice axes, and
{func}`~pulserver.ir.prescribe` applies it to every file of the chain before
the passes run. No gradient is changed:

- an RF pulse played under a gradient $G$ along an axis of the offset receives
  the frequency offset $G d$, with $G$ in Hz/m, and a phase offset; where the
  gradient varies during the pulse, the remaining phase is added as a phase
  shape;
- a readout receives the frequency and phase offsets, and where the gradient
  varies during it a phase modulation, so that the phase of each sample
  relative to its excitation is $2\pi\,\mathbf{d}\cdot\mathbf{k}(t)$, with
  $\mathbf{k}$ the k-space location of the sample in 1/m;
- blocks labelled `NOPOS` keep the phases they were designed with.

A block that carries a rotation extension plays its drawn gradients turned by
that rotation, and it is moved by the gradients it plays.

The readouts are therefore demodulated to the prescribed centre as they are
acquired, and the reconstruction receives an object at $\mathbf{d}$ at the
centre of its field of view. The cache carries each readout's frequency and
phase offsets and its phase modulation, one phase per sample, which
`pulseg_get_cursor_adc_phase_modulation` returns; a playout whose receiver
cannot vary its phase within a readout applies the modulation to the samples
before they are sent. A chain whose modulation does not hold one phase per ADC
sample is refused. The rotation of the prescription is not applied
to the cache: the scanner plays it through its rotation matrix, composed after
each block's own rotation. Two offsets make two caches of one design, and two
designs ({doc}`designs`).

## Checks

The gradient coils are limited per physical axis, and the peak a gradient
reaches on one axis depends on the orientation it is played in: two logical
axes at 0.8 of the amplitude limit each put $0.8\sqrt{2}$ of it on one physical
axis at 45°. So the checks are made in the physical frame. The prescription's
rotation $R$, from logical to physical axes, reaches the host in the nine
`fov_rotation_ij` entries of the protocol, element $(i, j)$ of $R$, and each
file is rotated by it as the scanner plays it: composed after each block's own
rotation, with blocks labelled `NOROT` left unrotated. A prescription with a
reflection in it is checked as it plays, reflection included.

Before a chain is converted, {func}`~pulserver.ir.check` runs pypulseqpp's
timing check, gradient continuity included, and its gradient amplitude and
slew-rate checks on every file, against the gradient limits, dead times and
ringdown time of the scanner. Where the call's limits carry them
({class}`~pulserver.ir.CheckLimits`), it also runs pypulseqpp's PNS check
under the scanner's nerve model and its mechanical-resonance check against the
scanner's forbidden gradient bands. The waveforms are timed by the rasters the
file declares. The PSD passes these limits with every design call
({doc}`../user-guide/running`); it computes the SAR and the gradient heating.

No design is stored for a generated design or an imported chain that fails a
check: `generate` and `import` reply with the problems, as they do with a
design error. The checks compute estimates; passing them does not
establish scanner or patient safety.

## SAR against a reference pulse

The PSD computes SAR under its own calibration of the transmit chain. Where
local SAR is computed from virtual observation points (VOPs), the energy a
pulse deposits at VOP $v$ is $\int \mathbf{b}(t)^H Q_v\, \mathbf{b}(t)\,dt$,
with $\mathbf{b}$ the drive of each transmit channel and $Q_v$ the VOP's
matrix, and it depends on the shape of the pulse and its channel weights. The
host evaluates it against a reference: the hard pulse of 180° and 1 ms, played
in the default channel weights. For each repetition $w$ of a subsequence, the
blocks before the first repetition and after the last included, as
pypulseqpp's SAR check averages over them, {func}`~pulserver.ir.sar_ratios`
computes

$$
r = \max_w \max_v \frac{E_{v,w}}{N_w\, E_v^{\mathrm{ref}}},
$$

with $E_{v,w}$ the energy of the repetition at VOP $v$, $N_w$ the number of
pulses it plays and $E_v^{\mathrm{ref}}$ the energy of the reference pulse
there: the energy of the repetition over that of the same repetition with each
of its pulses replaced by the reference. A 1 ms hard pulse of 90° counts a
quarter of the reference, and the ratio is 1 for a repetition of reference
pulses. The global SAR matrix of the VOP file gives the same ratio for global
SAR. A scale common to every channel's drive and to the VOPs cancels in both.

The cache carries the two ratios of each subsequence in its
`pulseg_subseq_info`, zero without VOPs or without RF, and the PSD computes
the SAR of the subsequence as the ratio times its SAR for the reference
repetition.

## Passes

| Pass | Result |
| --- | --- |
| Event deduplication | A library of distinct RF, gradient and ADC definitions, and a per-block instance table recording the definition each block plays and its amplitude |
| Repetition detection | The repeating unit of each subsequence, its repetition time (TR), and the preparation and cool-down regions around it |
| Segmentation | The repeating unit divided into segments at block boundaries where every gradient waveform is zero |
| Execution stream | The order in which segments are played over the whole scan |
| Label table | The Pulseq labels of every readout, three of which fill the ADC label columns |
| RF statistics | For each RF definition: its transmit channels, flip angle and energy, as pypulseqpp counts them; its duration from the envelope; and the bandwidth and bands of a multiband pulse from its spectrum |

The limits and rasters of the scanner (`pypulseqpp.Opts`) are those under which
the scan is segmented. A segment boundary falls where every gradient is zero,
judged by the first and last values each arbitrary gradient's library row
stores for the event's edges; a trapezoid starts and ends at zero. The
spectral statistics are measured by pypulseqpp's
`calc_rf_bandwidth` when the chain is read, with the Pulseq recipe of the width
at half the spectral peak; a band is a run of the spectrum above 30 % of its
peak, and its offset is measured from the carrier. A dynamic pTx pulse holds
its channels one after another over one time base. Its channel count is
pypulseqpp's `Sequence.rf_channels`: the number of samples at its first sample
time, when the times are that many identical copies. Its flip angle is
pypulseqpp's `Sequence.rf_flip_angles`, the channels' integrals summed
coherently, the flip where every channel has unit, in-phase sensitivity; the
envelope for the power statistics is the root sum of squares of the channels.
The energy is held as the integral of the squared envelope scaled to unit
peak, in seconds: pypulseqpp's `calc_rf_power` energy, the integral of
$|b_1|^2$, over its peak power, both summed over the channels.
A pulse a file leaves unlabelled takes the use pypulseqpp detects for it when
the chain is read. `label_column_map` selects the three labels the
interpreter records per readout, as indices in the order SLC, PHS, REP, AVG,
SEG, SET, ECO, PAR, LIN, ACQ.

## Cache file

The cache has the name of the first sequence file with its extension replaced:
`.pseg` by default and `.pge` on GE. It is divided into sections that a
consumer loads independently. The pulse-generation stage of a playout reads the
definitions and their waveforms; the scan loop also reads the per-block
instances, the rotations and the execution stream, whose size scales with the
scan length. Integer and float fields are 4 bytes. The byte order is recorded
in the file, and a reader on a machine of the other byte order swaps on load. A
cache whose format version differs from the reader's is rejected rather than
read in part.

A cache may be tagged with a vendor code (`PULSEG_VENDOR_*`), which selects the
reader built for that vendor. {func}`~pulserver.ir.summary` reports the
subsequences, segments and readouts of a sequence, computed from the chain or
loaded from a vendor-neutral cache.

## Playback

{func}`~pulserver.ir.play` loads a cache with the C library and walks its
execution stream with the cursor a playout uses, resolving each block as the
scanner plays it: its duration, the RF and ADC frequency and phase offsets,
the RF use, the ADC window, the gradient amplitudes and the rotation, and, on
request, the gradient waveforms each instance plays. It stands in for the
interpreter, so the cache can be compared with the file it was converted from
without a scanner; the test suite holds every played block of each fixture to
the block its file designs, and the trajectory the waveforms trace to the one
the file designs ({doc}`virtual-scanner`).

## Language constraint

The IR is built by C++ passes in `src/cpp/ir/`, which run only on the host. The
cache reader, and the accessors a playout uses to walk the loaded collection,
are compiled into the interpreter, whose vendor toolchains accept ANSI C. They
are therefore C89, in `src/c/`, and call nothing in `src/cpp/`. The test suite
compiles `src/c/` as a scanner build does, 32-bit and vendor-tagged, and reads
back a cache written on the host.

## See also

* {doc}`../api/ir` — the conversion interface.
* {doc}`protocol` — the protocol entries the offset arrives in.
* [`src/c/include/pulseg/`](https://github.com/pulserver/pulserver/tree/main/src/c/include/pulseg)
  — the public C headers the interpreter includes.
* {doc}`/generated/gallery/02-scanner-ir/01_segmentation` — the segmentation of shipped sequences, executed.
