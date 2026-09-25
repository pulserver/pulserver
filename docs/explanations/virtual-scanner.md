# The virtual scanner

The IR, the raw-data header contract and the reconstruction path are tested
against data a scanner acquires. Data sampled at the k-space locations the
enrichment computes cannot test the enrichment, and a design compared only
with itself cannot test the IR. The virtual scanner stands in for the scanner
and nothing else: the design calls, the IR, the reconstruction proxy and the
reconstruction plugins it drives are the production code. Its data are
acquired from an analytic phantom along the trajectory the cache plays, and
sent as the scanner's reconstruction client sends them.

## Stand-ins

| Stand-in | Replaces | Built from | Exercises |
| --- | --- | --- | --- |
| Design call | The PSD host process's call | `pulserver design generate`, or {func}`~pulserver.host.call` | Request and reply blocks, presets, design errors, check failures |
| Virtual interpreter | The playout | The C library's cursor over the cache, {func}`~pulserver.ir.play` | Segmentation, execution stream, the waveforms and offsets the cache carries |
| Physics | Magnet, coils and subject | {class}`~pulserver.virtual.Phantom`, {func}`~pulserver.virtual.acquire` | The trajectory the enrichment has to state, the demodulation of the prescription, the timing of every echo, the RF frequencies ppm offsets resolve to |
| Reconstruction client | The scanner's reconstruction client | {func}`~pulserver.virtual.send` | The header and acquisition contract of {doc}`../user-guide/reconstruction-client` |

## Played trajectory

{func}`~pulserver.ir.play` walks the cache with the cursor a playout uses and,
with `waveforms`, returns each played block's gradients: the instance's
amplitude times the waveform that instance plays, timed from the block's
start. The accessors that answer for a segment position,
`pulseg_get_grad_amplitude` and `pulseg_get_grad_time_us`, answer through
its representative, the instance of largest energy; where the instances of
one position play distinct shapes, as the interleaves of a spiral drawn as
distinct shapes do, `pulseg_get_cursor_grad_waveform` returns the shape of
the instance at the cursor. It also returns each played RF pulse: the
instance's amplitude times the magnitude and phase shapes of its definition,
which `pulseg_get_rf_magnitude` and `pulseg_get_rf_phase` return, the phase in
cycles as Pulseq stores it.

{func}`~pulserver.virtual.trajectory` integrates those gradients into the
k-space location $\mathbf{k}$ of every ADC sample, in 1/m along the physical
axes. The rotation $R$ of the prescription, from logical to physical axes, is
not in the cache ({doc}`ir-cache`): the virtual interpreter is given it, as a
playout is, and turns each block's gradients by the block's own rotation and
then by $R$, except in blocks labelled `NOROT`, which play turned by their own
rotation alone. This is the frame {func}`~pulserver.ir.check` checks in. Under
the identity the physical axes are the logical ones. An excitation returns
$\mathbf{k}$ to zero, and a refocusing pulse negates it, at the RF centre the
design records, which the cache carries for each RF event with its use; each
file of a chain starts from zero. The test suite holds this trajectory to the
one each file designs, turned as `pypulseqpp.TransformFOV` turns it for the
checks, and the trajectory of the identity to the one the enrichment states.

## Signal model

The phantom lies in the physical frame, where a rotation and a position place
it ({class}`~pulserver.virtual.Phantom`). Each of its ellipses holds spins of
one chemical shift $\sigma$, in ppm from water, which precess at
$f_\sigma = 10^{-6}\sigma\gamma B_0 + \Delta f$ from the scanner's centre
frequency: the shift resolved at the magnet's field $B_0$ with pypulseqpp's
default $\gamma$, in Hz/T, plus an off-resonance $\Delta f$ common to every
spin.

After an excitation of RF phase $\phi_e$ at its centre, the magnetization is
transverse, at the phantom's density $\rho_\sigma(\mathbf{r})$ whatever the
flip angle, with phase $-\phi_e - \pi/2$, and it precesses as
$\exp(-2\pi i\,(\mathbf{k}\cdot\mathbf{r} + f_\sigma\tau))$, with
$\mathbf{r}$ and $\mathbf{k}$ along the physical axes and $\tau$ the time it
has precessed freely: counted from the excitation's centre and, as
$\mathbf{k}$ is, negated about the centre of each refocusing pulse, so that a
spin echo refocuses it. A refocusing pulse of phase $\phi_r$ conjugates the
magnetization about $-\phi_r$. The RF phase at the centre is the pulse's
phase offset plus its frequency offset times the time since the pulse began.
Coil $c$, of sensitivity $s_c(\mathbf{r})$, receives

$$
S_c(t) = e^{i\psi} \sum_\sigma m_\sigma \int \rho_\sigma(\mathbf{r})\,
s_c(\mathbf{r})\, e^{-2\pi i\,(\mathbf{k}(t)\cdot\mathbf{r} + f_\sigma\tau(t))}\,
d\mathbf{r},
$$

with $\psi$ the phase the RF pulses left and $m_\sigma$ the longitudinal
magnetization of shift $\sigma$ the excitation tipped, as a fraction of
equilibrium. A saturation pulse scales $m_\sigma$ by the $z$ component it
leaves at $f_\sigma$, from $+z$, in pypulseqpp's relaxation-free Bloch
simulation (`pypulseqpp.sim_bloch`) of the pulse the cache plays, in the frame
of its frequency. The next excitation tips what remains, and the
magnetization is at equilibrium again after it. A saturation pulse played
under a gradient saturates a band in space, which a species of the phantom
does not hold, and is refused. The playout demodulates $S_c$ by
$\exp(i\theta(t))$, where the receiver phase $\theta$ is the ADC phase
offset at the ADC's start, advancing at its frequency offset, plus its phase
modulation.

These are the conventions under which the field-of-view translation applied
when the IR is built ({doc}`ir-cache`) recentres an object. Where every block
is turned by $R$, $\mathbf{k}$ is $R\mathbf{k}_L$, with $\mathbf{k}_L$ its
value along the logical axes. An object displaced to the prescribed centre
$R\mathbf{d}$, for the offset $\mathbf{d}$ along the logical axes, gains the
phase $-2\pi\,(R\mathbf{d})\cdot(R\mathbf{k}_L) = -2\pi\,\mathbf{d}\cdot\mathbf{k}_L$,
which the demodulation removes, and an object turned by $R$ presents at
$R\mathbf{k}_L$ its own transform at $\mathbf{k}_L$. An object posed at the
prescribed centre, its axes turned by $R$, is therefore acquired as the same
object at the isocentre under the identity, a reflection in $R$ included. The
phantom's ellipses and its sensitivities, sums of plane waves, have analytic
transforms, so $S_c$ is evaluated exactly at every sample.

Relaxation, diffusion, slice profiles and every RF use other than
excitation, refocusing and saturation are not modelled. Without relaxation, a
saturation pulse leaves spins at its own frequency at the cosine of its flip
angle: pypulseqpp's `FatSaturation`, of 110° by default, leaves fat at
$\cos 110° \approx -0.34$ of its magnetization. A readout before the first
excitation of its file acquires zeros.

## External simulators

A Bloch simulator that reads Pulseq files, KomaMRI for example, models what
the signal model above leaves out: relaxation, slice profiles and the action
of every RF pulse on the magnetization. A simulation of the design would test
the design alone; {func}`~pulserver.virtual.export` writes the blocks the
cache plays instead, so that a simulation of the file tests the IR as the
virtual interpreter does. Each played block becomes one block of a Pulseq 1.4.1 file,
of the duration it plays:

- The RF pulse the cache plays, on the RF raster, with its phase offset and
  its frequency offset from the pulse's start applied to the samples, so that
  the file's RF offsets are zero and no convention for them is left to the
  simulator. The channels of a pTx pulse are summed, as at unit, in-phase
  sensitivity.
- The gradients along the physical axes, turned by the block's rotation and
  then by $R$ except in blocks labelled `NOROT`, as the virtual interpreter
  turns them. A time-shaped gradient through the corners of all three axes
  carries each, and a step from or to zero at a gradient's first or last
  corner is a ramp 10 ns wide, since a time shape holds one value per time.
  The file therefore needs no rotation extension, which revision 1.4.1 does
  not have.
- The ADC window without its offsets. Revision 1.4.1 cannot carry a phase
  modulation, so the receiver phase $\theta$ of every sample is returned
  instead, and the simulated samples are demodulated by it as the playout
  demodulates.

The file holds the standard sections alone. It carries neither the RF use nor
the RF centre, which a Bloch simulation does not read. The text format writes
a gradient's amplitude to six significant figures, and a turned gradient's
amplitude is written as the rotation leaves it, so the k-space of a file
exported under an oblique prescription agrees with the played trajectory to a
relative $10^{-5}$ of its extent.

## What a run establishes

- The trajectory the cache plays is the one each file designs, for the
  fixtures and for every sequence pypulseqpp ships, at small sizes, and under
  an oblique and a reflected prescription the one `pypulseqpp.TransformFOV`
  turns the design into for the checks. A block labelled `NOROT` plays turned
  by its own rotation alone.
- The cache carries the RF centre a design records where it is away from the
  magnitude peak, and plays every RF pulse of a fat-saturated EPI and of a
  spin echo as the design draws it.
- Off resonance, every sample of the fixtures and of every shipped sequence
  accrues the phase its file's excitation, refocusing and ADC timing give it,
  so the cache plays each echo as the design times it.
- Fat precesses at its chemical shift at the magnet's field. A fat saturation
  leaves water and fat what pypulseqpp's Bloch simulation of the designed
  pulse leaves them, and one converted at another field misses the fat.
- The enrichment states the trajectory of the identity, along the logical
  axes, for every readout.
- An object posed where an axial, an oblique or a reflected prescription
  places the field of view is acquired by every shipped sequence as it is at
  the isocentre, EPI and spiral readouts, which the translation gives a phase
  modulation, and readouts turned by a rotation extension included; an object
  away from the offset, or turned otherwise than prescribed, is not.
- A series streamed through the proxy under an axial, an oblique and a
  reflected prescription is reconstructed into the image of the phantom at the
  isocentre, and the image carries the prescribed centre and the columns of
  $R$ as its read, phase and slice directions; a series short of a readout is
  refused.
- The exported file of every fixture and every shipped sequence, read and
  integrated by pypulseqpp, has the trajectory the cache plays, under an
  axial, an oblique and a reflected prescription.

## See also

* {doc}`ir-cache` — the IR, its prescription and its playback.
* {doc}`../user-guide/reconstruction-client` — the stream the virtual reconstruction client sends.
* {doc}`../api/virtual` — the phantom, the acquisition, the client and the export.
