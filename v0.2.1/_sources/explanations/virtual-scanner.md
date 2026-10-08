# The virtual scanner

```{admonition} TL;DR
:class: tldr

- The virtual scanner replaces the scanner and nothing else: the design calls,
  the IR, the proxy and the reconstruction plugins it drives are the
  production code.
- It plays the IR cache through the C library's playout stages, integrates
  the played gradients into the k-space trajectory under the prescription's
  rotation, and acquires a phantom's tissue along it with the Fourier engine:
  extended phase graphs of each class of tissue over the pulses and gradient
  moments the cache plays, and the images their signals weight encoded along
  the trajectory by a NUFFT.
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
acquired of a phantom's tissue along the trajectory the cache plays, under the
pulses and gradients the cache plays, and sent as the scanner's
reconstruction client sends them. A browser build of its console is published
at <https://pulserver.github.io/MaRGE/>.

## Stand-ins

| Stand-in | Replaces | Built from | Exercises |
| --- | --- | --- | --- |
| Design call | The interpreter host process's call | `pulserver design generate`, or {func}`~pulserver.host.call` | Request and reply blocks, presets, design errors, check failures |
| Virtual interpreter | The playout | The C library's two playout stages over a recording backend, {func}`~pulserver.ir.playout` | Segmentation, the events each segment position is prepared with, the registers the scan loop sets on each block, the rotation and trigger of each segment instance, the waves and the waveform memory they are loaded into |
| Physics | Magnet, coils and subject | {class}`~pulserver.virtual.Phantom`, {class}`~pulserver.virtual.Tissue`, {func}`~pulserver.virtual.simulate` | The trajectory the enrichment has to state, the demodulation of the prescription, the timing of every echo, the RF frequencies ppm offsets resolve to, and the flip angle, phase and band of every RF pulse, the gradient moments and the receiver phase the cache plays |
| Scan clock | The scanner's acquisition in real time, and its gradient coils' sound | {class}`~pulserver.virtual.Scan` | The rate at which readouts reach the reconstruction; the sound of the gradients the cache plays |
| Reconstruction client | The scanner's reconstruction client | {func}`~pulserver.virtual.send` | The header and acquisition contract of {doc}`../user-guide/reconstruction-client` |

## Played trajectory

{func}`~pulserver.ir.playout` plays the cache through the two stages of a
playout ({doc}`../developer-guide/internals/ir-cache`) and, with `waveforms`,
returns the gradients and the RF pulse each block plays, as the scan loop sets
them; {doc}`../developer-guide/internals/virtual-scanner` states which
prepared events and waves they are taken from.

The virtual scanner integrates those gradients into the k-space location
$\mathbf{k}$ of every ADC sample, in 1/m along the physical axes. The rotation $R$ of the prescription, from logical to physical axes, is
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

The Fourier engine ({class}`~pulserver.virtual.FourierPlayer`) acquires a
phantom's tissue under the pulses, gradients and readouts the cache plays. It
separates what happens to the magnetization of each kind of tissue over time,
which extended phase graphs give, from where that tissue is, which the
trajectory encodes.

### Tissue

The phantom lies in the physical frame, where a rotation and a position place
it ({class}`~pulserver.virtual.Phantom`).
{meth}`~pulserver.virtual.Phantom.tissue` samples it as cubes of uniform
magnetization ({class}`~pulserver.virtual.Tissue`), $\Delta$ wide along the
axes the phantom spans: a square in the plane of a phantom of ellipses, a
cube of BrainWeb's head. Each cube $j$ holds a proton density $\rho_j$,
relaxation times $T_{1,j}$, $T_{2,j}$ and $T'_{2,j}$, and a frequency $f_j$
from the scanner's centre frequency: its chemical shift $\sigma$, in ppm from
water, resolved at the magnet's field $B_0$ with pypulseqpp's default
$\gamma$, in Hz/T, plus the precession in the field $\Delta B_j$, in T, the
subject adds and an off-resonance $\Delta f$ common to every cube,

$$
f_j = 10^{-6}\sigma_j\gamma B_0 + \gamma\,\Delta B_j + \Delta f .
$$

Cubes of the same $T_1$, $T_2$ and $T'_2$ form a class of tissue.

### RF pulses

Each RF pulse turns the magnetization as a rotation at its centre. On
resonance, its flip angle $\alpha$ and the phase of the axis it turns about
are those of pypulseqpp's relaxation-free Bloch simulation
(`pypulseqpp.sim_bloch`) of the waveform the cache plays, the phase taken at
the pulse's centre: its phase offset plus its frequency offset times the time
since the pulse began. Its profile $P(\delta)$ is the angle the same
simulation tips magnetization at rest through at a detuning $\delta$ from the
pulse's frequency, over the angle on resonance. A pulse played under a
gradient $\mathbf{g}$, in Hz/m, held during it turns cube $j$ through

$$
\alpha_j = \alpha\,|b^+(\mathbf{r}_j)|\,P(\mathbf{g}\cdot\mathbf{r}_j + f_j - f_\mathrm{rf}),
$$

about its axis turned by the phase of the transmit field $b^+$, relative to
the pulse's nominal amplitude; a pulse played under no gradient, or under one
that changes during it, selects by frequency alone. A pulse of phase zero turns
$+z$ towards $+y$: after a 90° excitation of phase $\phi_e$ the magnetization
is transverse at the phase $\pi/2 - \phi_e$. Cubes that every pulse turns
through about the same angle form a group.

### Configuration states

Between two events, the net moment $\Delta\mathbf{m}$ of the gradients, in
1/m, dephases a voxel $\mathbf{v}$ by $\Delta\mathbf{m}\cdot\mathbf{v}$ cycles
along the axes the tissue spans. The voxel is the resolution the widest $k$
reaches along each logical axis, the thickness of the slab an excitation
selects along an axis $k$ does not move along, and the tissue's extent where
neither bounds it. An interval that dephases the voxel by half a cycle or more
shifts the configuration states by one order: the magnetization of the
voxel is then described by its dephased pathways rather than by the
magnetization of each position in it, as extended phase graphs describe it
(Weigel, J Magn Reson Imaging 41:266, 2015,
doi:[10.1002/jmri.24619](https://doi.org/10.1002/jmri.24619)). A phase
encoding that the intervals between two pulses sum to zero is balanced and
shifts nothing; a moment that changes from one repetition to the next and is
not cancelled, as a radial readout's turned prephaser is not, counts. A
readout reads the pathway that passes the centre of k-space during it: the
free induction, or the echo of an earlier interval, as an SSFP-echo readout
reads one. Its echo is the sample at which that pathway passes nearest the
centre.

The excitations under one selection, one gradient, frequency and profile, and
the readouts that follow them form a station.
For each station, the stream of its events, the pulses that act on its groups
and its readouts with the shifts between them, is simulated by blochsim's
extended phase graphs for each class of tissue, at the frequencies and
transmit-field magnitudes the cubes interpolate between. The frequency enters
the states only where an interval between two pulses shifts nothing, as in a
balanced or a refocused sequence; elsewhere it turns the samples alone.

### Temporal and readout bases

At each readout's echo, the signals of a station's groups and classes across
its readouts are spanned by a few temporal terms $u_p$. Across the readout,
from the echo at $\tau = 0$, a cube decays as $e^{-\tau/T_2}$, dephases as
$e^{-|t_u|/T'_2}$, with $t_u$ the time the magnetization has gone
unrefocused, and precesses as $e^{-2\pi i f_j\tau}$, or as
$e^{-2\pi i f_j t_u}$ where the frequency does not enter the states; those
factors are
spanned by a few terms $v_q$ of the samples' times. Each cube weighs every
pair $(u_p, v_q)$, and the weights of all cubes are spanned by fewer
combinations of the pairs, the terms $t$. Each term is an image
$m_t(\mathbf{r})$, the cubes' weights in it times their density, so that coil
$c$, of receive sensitivity $s_c(\mathbf{r})$, receives at sample $n$

$$
S_c(n) = e^{i\psi_n} \sum_t w_t(n) \int s_c(\mathbf{r})\, m_t(\mathbf{r})\,
e^{-2\pi i\,\mathbf{k}(n)\cdot\mathbf{r}}\, d\mathbf{r},
$$

with $w_t(n)$ the term's temporal and readout factor at the sample and
$\psi_n$ the phase the pulses left, which the receiver phase demodulates. The
images lie on a grid along the logical axes the trajectory encodes, at the
resolution its widest $k$ reaches, over the cubes the station excites; each
cube is uniform over its width, so its transform weighs every sample. Each
image times each coil's sensitivity is transformed to the samples by
bartorch's NUFFT. The spacing of the tissue sets how closely the cubes follow
the object; the resolution of the images is the trajectory's whatever it is.

### Demodulation and the field-of-view offset

The playout demodulates $S_c$ by $\exp(i\theta(t))$, where the receiver phase
$\theta$ is the ADC phase offset at the ADC's start, advancing at its
frequency offset. The cache carries no ADC phase modulation: the
reconstruction proxy applies it to the received samples
({doc}`reconstruction`).

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
object at the isocentre under the identity, a reflection in $R$ included.

### Diffusion, motion and the gradient response

A tissue class with a diffusion coefficient $D$ loses $e^{-bD}$ of its signal
at each readout, where $b = (2\pi)^2\int |\mathbf{k}(t)|^2\,dt$ is integrated
since the last excitation over the moment measured from the readout's origins,
so a refocusing pulse mirrors it. Diffusion is isotropic. It is left out where
no class diffuses, or where the largest $b$ of the scan times the largest $D$
is below a hundredth.

A subject in rigid motion, {class}`~pulserver.virtual.RigidMotion`, is held in
the pose it has at each readout's echo for the whole readout. The pose turns
the readout's k-space and shifts its phase, as a phantom placed in that pose
would be acquired; the coils, the transmit field and the relaxation move with
the object.

A gradient impulse response, {class}`~pulserver.virtual.Girf`, filters the
moment the gradients play on each physical axis, so the trajectory, the
origins and the dephasing all follow the filtered gradients.

### What is not modelled

A pulse acts at its centre, with no
relaxation during it, and turns each cube by an ideal rotation, so the phase a
selective pulse leaves across its slab beyond that rotation is left out. The
channels of a pTx pulse are summed, as at unit, in-phase sensitivity. A
readout before the first excitation of its file acquires zeros.

## Coils and the subject's field

A phantom is received by coils of its own. Its tissue can be sampled in one of
the scanner's {obj}`~pulserver.virtual.COILS` instead, fixed in the physical frame, each a transmit coil and a receive coil,
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

A pulse plays on every transmit channel, weighted by the coil's default shim,
the unit weights that bring the channels into phase at the isocentre, where the
pulse then has its nominal amplitude: each cube sees the transmit field
$b^+(\mathbf{r}) = \sum_c w_c\,s^+_c(\mathbf{r})$, for the weights $w_c$.

A console given field maps takes the coils, and the subject's field, from
electromagnetic and susceptibility models of BrainWeb's head instead
({doc}`../developer-guide/internals/virtual-scanner`).

## See also

* {doc}`scanner-representation` — the IR and its prescription.
* {doc}`../user-guide/reconstruction-client` — the stream the virtual reconstruction client sends.
* {doc}`../api/virtual` — the phantom, the Fourier engine, the scan clock, the client and the export.
* {doc}`../developer-guide/internals/virtual-scanner` — the played waveforms, the Fourier engine's streams, bases and grids, external simulators, coils from field maps, the scan clock and the sound.
