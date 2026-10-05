# The virtual scanner

```{admonition} TL;DR
:class: tldr

- The virtual scanner replaces the scanner and nothing else: the design calls,
  the IR, the proxy and the reconstruction plugins it drives are the
  production code.
- It plays the IR cache through the C library's playout stages, integrates
  the played gradients into the k-space trajectory under the prescription's
  rotation, and acquires an analytic phantom along it or simulates the blocks
  on isochromats with the Bloch equation.
- The playout demodulates by the ADC frequency and phase offsets; the cache
  carries no phase modulation, and the reconstruction proxy applies the rest of
  the field-of-view phase.
```

The IR, the raw-data header contract and the reconstruction path are tested
against data a scanner acquires. Data sampled at the k-space locations the
enrichment computes cannot test the enrichment, and a design compared only
with itself cannot test the IR. The virtual scanner stands in for the scanner
and nothing else: the design calls, the IR, the reconstruction proxy and the
reconstruction plugins it drives are the production code. Its data are
acquired along the trajectory the cache plays, from an analytic phantom or by a
Bloch simulation of the blocks the cache plays, and sent as the scanner's
reconstruction client sends them. A browser build of its console is published
at <https://pulserver.github.io/MaRGE/>.

## Stand-ins

| Stand-in | Replaces | Built from | Exercises |
| --- | --- | --- | --- |
| Design call | The interpreter host process's call | `pulserver design generate`, or {func}`~pulserver.host.call` | Request and reply blocks, presets, design errors, check failures |
| Virtual interpreter | The playout | The C library's two playout stages over a recording backend, {func}`~pulserver.ir.playout` | Segmentation, the events each segment position is prepared with, the registers the scan loop sets on each block, the rotation and trigger of each segment instance, the waves and the waveform memory they are loaded into |
| Physics | Magnet, coils and subject | {class}`~pulserver.virtual.Phantom`, {func}`~pulserver.virtual.acquire`, {func}`~pulserver.virtual.simulate` | The trajectory the enrichment has to state, the demodulation of the prescription, the timing of every echo, the RF frequencies ppm offsets resolve to; in the Bloch simulation, every RF pulse, gradient and receiver phase the cache plays |
| Scan clock | The scanner's acquisition in real time, and its gradient coils' sound | {class}`~pulserver.virtual.Scan` | The rate at which readouts reach the reconstruction; the sound of the gradients the cache plays |
| Reconstruction client | The scanner's reconstruction client | {func}`~pulserver.virtual.send` | The header and acquisition contract of {doc}`../user-guide/reconstruction-client` |

## Played trajectory

{func}`~pulserver.ir.playout` plays the cache through the two stages of a
playout ({doc}`../developer-guide/internals/ir-cache`) and, with `waveforms`,
returns the gradients and the RF pulse each block plays, as the scan loop sets
them; {doc}`../developer-guide/internals/virtual-scanner` states which
prepared events and waves they are taken from.

{func}`~pulserver.virtual.trajectory` integrates those gradients into the
k-space location $\mathbf{k}$ of every ADC sample, in 1/m along the physical
axes. The rotation $R$ of the prescription, from logical to physical axes, is
not in the cache ({doc}`scanner-representation`): the virtual interpreter is given it, as a
playout is, and turns each block's gradients, which carry the block's own
rotation, by $R$, except in the segment instances whose blocks are labelled
`NOROT`, which play as the cache holds them. This is the frame
{func}`~pulserver.ir.check` checks in. Under
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
flip angle, with phase $\pi/2 - \phi_e$, and it precesses as
$\exp(-2\pi i\,(\mathbf{k}\cdot\mathbf{r} + f_\sigma\tau))$, with
$\mathbf{r}$ and $\mathbf{k}$ along the physical axes and $\tau$ the time it
has precessed freely: counted from the excitation's centre and, as
$\mathbf{k}$ is, negated about the centre of each refocusing pulse, so that a
spin echo refocuses it. A pulse of phase zero turns the magnetization from
$+z$ towards $+y$, as it turns the magnetic moment of a nucleus of positive
gyromagnetic ratio. A refocusing pulse of phase $\phi_r$ conjugates the
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
offset at the ADC's start, advancing at its frequency offset. The cache
carries no ADC phase modulation: the reconstruction proxy applies it to the
received samples ({doc}`reconstruction`).

These are the conventions under which the field-of-view translation applied
when the IR is built ({doc}`scanner-representation`) recentres an object. Where every block
is turned by $R$, $\mathbf{k}$ is $R\mathbf{k}_L$, with $\mathbf{k}_L$ its
value along the logical axes. An object displaced to the prescribed centre
$R\mathbf{d}$, for the offset $\mathbf{d}$ along the logical axes, gains the
phase $-2\pi\,(R\mathbf{d})\cdot(R\mathbf{k}_L) = -2\pi\,\mathbf{d}\cdot\mathbf{k}_L$,
which the demodulation removes where the readout gradient holds one value
across the sampling window; under one that varies, the demodulation removes
the line through the window centre and the proxy the rest. An object turned by $R$ presents at
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

## Bloch simulation

The signal model leaves out what an RF pulse does beyond an ideal excitation,
refocusing or saturation, and what the magnetization does between them.
{func}`~pulserver.virtual.simulate` plays the blocks
{func}`~pulserver.ir.playout` returns on isochromats instead
({class}`~pulserver.virtual.Isochromats`, {doc}`../developer-guide/internals/bloch-engine`): flip
angles, slice profiles, relaxation and the coherences one repetition leaves to
the next act as the Bloch equation gives them, and the magnetization is carried
from each block to the next, across the files of a chain. Each block's
gradients are those of the played trajectory, turned by $R$ except in blocks
labelled `NOROT`. Its RF pulse and its ADC play as the RF and ADC events of a
Pulseq block, with the frequency and phase offsets the playout sets, and each
sample is demodulated by $\exp(i\theta(t))$ as above. A 90°
excitation of phase $\phi_e$ leaves the magnetization at the phase
$\pi/2 - \phi_e$ of the signal model.

### The Bloch equation in the rotating frame

In the frame rotating at the scanner's reference frequency, an isochromat at
position $\mathbf{r}$ sees the field

$$
\mathbf{b}(t) = \bigl(\operatorname{Re} b_1(t),\ \operatorname{Im} b_1(t),\
\mathbf{G}(t)\cdot\mathbf{r} + \Delta f\bigr),
$$

in Hz: $b_1$ is the transverse RF field, $\mathbf{G}$ the gradient in Hz/m and
$\Delta f$ the isochromat's off-resonance, field inhomogeneity and chemical
shift together. The magnetisation obeys

$$
\frac{d\mathbf{M}}{dt} = 2\pi\,\mathbf{M}\times\mathbf{b}
- \frac{M_x\hat{\mathbf{x}} + M_y\hat{\mathbf{y}}}{T_2}
- \frac{(M_z - M_0)\,\hat{\mathbf{z}}}{T_1},
$$

with $M_0$ the proton density. The cross product in this order is the
precession of a nucleus of positive gyromagnetic ratio: the magnetisation
turns about $\mathbf{b}$ clockwise, seen from the tip of $\mathbf{b}$. A 90°
pulse along $+x$ takes $+z$ to $+y$, and free precession gives the transverse
magnetisation $M_{xy} = M_x + iM_y$ the factor $e^{-2\pi i\,b_z t}$.

pypulseqpp's relaxation-free {func}`~pypulseqpp.sim_bloch` turns the
magnetisation the other way, right-handedly about $\mathbf{b}$. The two are mirror images under $y \to -y$: the engine's
result for a field $b_1$ is `sim_bloch`'s for $b_1^*$ with $M_y$ negated.

### Pulseq events as fields

{meth}`~pulserver.virtual.Isochromats.play` plays one block's events as
follows, with the ppm offsets resolved at the system it is given.
{func}`~pulserver.virtual.simulate` gives it the events of each block the cache
plays, with the offsets the playout sets.

Gradients
: The corners of each axis's gradient, turned by the rotation the block is
  given, on the axes the isochromats' positions are given along: in the
  virtual scanner, the physical axes. Each turned axis is the sum on the union
  of the corners, where it is exact.

RF pulses
: Pulseq defines a pulse's complex waveform as its amplitude times its
  magnitude and phase shapes, times a carrier of the phase offset advancing at
  the frequency offset from the pulse's start,
  $w(t) = a\,m(t)\,e^{2\pi i\,p(t)}\,e^{i(\phi + 2\pi f t)}$, with the ppm
  offsets resolved at the system's `gamma` and `B0`. The engine plays
  $b_1 = w^*$. A phase ramp in $p(t)$ and an equal frequency offset are then
  one pulse, as the format defines them, and a positive $f$ excites
  isochromats at a positive $\Delta f$: under a positive slice-selection
  gradient, a slice at a positive position. Samples on the RF raster are held
  over their raster intervals. A pulse with a time shape is joined linearly
  between its samples and held over steps of the RF raster, or of its shortest
  interval where that is shorter.

Parallel transmission
: With transmit sensitivities $S_c(\mathbf{r})$, a pTx pulse's channels add as
  $b_1 = \sum_c S_c\,b_{1,c}$, one channel per sensitivity. Without them, the
  channels are summed: every channel has unit, in-phase sensitivity, and a
  pulse turns by the flip angle {meth}`~pypulseqpp.Sequence.rf_flip_angles`
  reports. {func}`~pulserver.virtual.simulate` plays a single-channel pulse on
  every channel of a coil, weighted by the block's RF shim or, without one, by
  the coil's default shim.

ADC samples
: Coil $c$ receives
  $s_c(t) = \sum_j R_{jc}\,M_{xy,j}(t)$, with $R_{jc}$ the receive sensitivity
  of isochromat $j$. Each sample is multiplied by $e^{i\theta}$, where $\theta$
  is the ADC phase offset plus its phase modulation, advancing at its frequency
  offset from the start of the window.

With these conventions, an ADC phase offset equal to the phase offset of the
excitation cancels it, which RF spoiling relies on. Isochromats excited by a
90° pulse along $+x$, without relaxation, give

$$
s(t) = i \sum_j \rho_j\, e^{-2\pi i\,\mathbf{k}(t)\cdot\mathbf{r}_j},
$$

with $\mathbf{k}$ the trajectory {meth}`~pypulseqpp.Sequence.calculate_kspace`
reports, and the inverse discrete Fourier transform of Cartesian samples places
each isochromat at its own position.

## Coils and the subject's field

The coils of the signal model are the phantom's own. A scan on the Bloch
simulation can take one of the scanner's {obj}`~pulserver.virtual.COILS`
instead, fixed in the physical frame, each a transmit coil and a receive coil,
named `transmit/receive`: the body coil both ways, taken to be one channel of
unit sensitivity each way; the body coil transmitting to a 48-channel receive
head array, `body/head48`; and an 8-channel head coil for parallel transmission
with a 32-channel receive head array, `head8/head32`. The head coils'
sensitivities are BART's coil models rather than electromagnetic simulations:
the 8-channel coil is `HEAD_2D_8CH`, constant along $z$, and the arrays are the
first 48 and 32 channels of `HEAD_3D_64CH`, each sampled by bartorch over a cube
25.6 cm wide about the isocentre and interpolated linearly between the samples.
The receive sensitivities $s_c$ have a root sum of squares of 1 at the
isocentre. The transmit sensitivities $s^+_c$ are the complex conjugates of the
8-channel coil's receive sensitivities, the quasi-static limit of reciprocity;
the models show none of the dielectric effects of a wavelength comparable to
the head.

Each transmit channel plays the pulse the cache holds for it, scaled at each
isochromat by $s^+_c(\mathbf{r})$, and the channels' fields add. A pTx pulse
holds one waveform per channel. A single-channel pulse plays on every channel,
weighted by its block's RF shim or, without one, by the coil's default shim,
the unit weights that bring the channels into phase at the isocentre, where the
pulse then has its nominal amplitude. A shim of another number of channels is
refused.

A console given field maps takes the coils, and the subject's field, from
electromagnetic and susceptibility models of BrainWeb's head instead
({doc}`../developer-guide/internals/virtual-scanner`).

## See also

* {doc}`scanner-representation` — the IR and its prescription.
* {doc}`../user-guide/reconstruction-client` — the stream the virtual reconstruction client sends.
* {doc}`../api/virtual` — the phantom, the acquisition, the Bloch simulation, the scan clock, the client and the export.
* {doc}`../developer-guide/internals/virtual-scanner` — the played waveforms, runs of repetitions, external simulators, coils from field maps, the scan clock and the sound.
* {doc}`../developer-guide/internals/bloch-engine` — how the Bloch engine samples a phantom, integrates free precession and RF pulses, and plays repeated blocks.
