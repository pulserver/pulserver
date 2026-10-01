# Bloch simulation on isochromats

A Bloch simulation computes the magnetisation that a sequence leaves in an
object and the signal it induces in the receive coils. The object is
represented by isochromats: spin packets that share one position, one
precession frequency and one pair of relaxation times. The virtual scanner
plays the blocks of an IR cache on isochromats
({class}`~pulserver.virtual.Isochromats`, {func}`~pulserver.virtual.simulate`).
This page states the equation the engine integrates, how each Pulseq event
enters it, how the resulting signal relates to the k-space trajectory, and how
repeated blocks are played without integrating each of them.

## The Bloch equation in the rotating frame

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

## Free precession

Between RF steps, $b_z = \mathbf{G}(t)\cdot\mathbf{r} + \Delta f$ turns the
magnetisation about $z$ only, and the turns add. Over an interval $\tau$,

$$
M_{xy} \leftarrow M_{xy}\, e^{-2\pi i\,(\mathbf{r}\cdot\mathbf{A} + \Delta f\,\tau)}\,e^{-\tau/T_2},
\qquad
M_z \leftarrow M_z\,e^{-\tau/T_1} + M_0\,(1 - e^{-\tau/T_1}),
$$

where $\mathbf{A} = \int \mathbf{G}\,dt$ is the gradient area over the
interval, in 1/m. A Pulseq gradient is piecewise linear, so $\mathbf{A}$ is
exact from its corners, and so is the update, for any $\tau$. The engine
accumulates $\mathbf{A}$ and $\tau$ and applies them when the magnetisation is
next needed, by an RF pulse, an ADC sample or a read: blocks that hold neither
cost nothing per isochromat.

Within an ADC window the update is applied from one sample to the next, and
each coil's sample is its sensitivity times $M_{xy}$ summed over every
isochromat, at a cost proportional to the isochromats, the coils and the
samples. The engine forms these sums over tiles of isochromats in independent
partial sums, four samples at a time, with AVX-512, or AVX2 and FMA,
instructions on processors that have them. The order of the additions, and
with it the rounding of a sample, depends on the processor and on the number
of threads.

When the gradient area and the time from one sample to the next are the same
throughout the window, as under a gradient held over it, isochromat $j$ turns
by the same factor at every sample, and sample $k$ of coil $c$ is

$$
s_c(k) = \sum_{T_2} e^{-k\,\Delta t/T_2}
\sum_{j:\,T_{2,j} = T_2} R_{jc}\,M_{xy,j}\,e^{-2\pi i\,k\,u_j},
\qquad
u_j = \mathbf{r}_j\cdot\Delta\mathbf{A} + \Delta f_j\,\Delta t,
$$

with $\Delta\mathbf{A}$ and $\Delta t$ the area and time between samples and
$M_{xy,j}$ taken at the first. For the isochromats of each $T_2$ this is a sum
of exponentials at the integer frequencies $k$, a type-1 non-uniform FFT. The
engine spreads each isochromat onto a grid of twice the samples with the
exponential-of-semicircle kernel of Barnett, Magland and af Klinteberg (SIAM J
Sci Comput 2019), 13 grid points wide, transforms the grid and divides out the
kernel's transform. The samples agree with the sums to within about
$10^{-13}$ of $\sum_j |R_{jc}\,M_{xy,j}|$, at a cost proportional to the
isochromats, the kernel's width and the coils rather than to the samples;
played to a tolerance, the kernel is the narrowest that holds it. A
window is read this way where that costs less than turning every isochromat at
every sample and the isochromats hold at most 32 values of $T_2$.

When the gradient changes during the window, as on a spiral or on the ramps of
a readout, the sample at time $t$ from the first is

$$
s_c(t) = \sum_j R_{jc}\,M_{xy,j}\,e^{-2\pi i\,\mathbf{k}(t)\cdot\mathbf{r}_j}\,e^{-z_j t},
\qquad
z_j = 1/T_{2,j} + 2\pi i\,\Delta f_j,
$$

with $\mathbf{k}(t)$ the gradient area from the first sample. The engine reads
such a window to within a tolerance $\varepsilon$ of the sum of the magnitudes
of the terms a sample sums: the one the window is played to, or $10^{-11}$ at
zero. Where the isochromats lie on a lattice, to within $10^{-6}$ of its
spacing, along each axis $\mathbf{k}$ moves along, those at one lattice point
$\mathbf{r}_q$ share $e^{-2\pi i\,\mathbf{k}\cdot\mathbf{r}_q}$. Along an axis
on which $\mathbf{k}$ stays so close to zero that the phase it adds across the
isochromats stays within $\varepsilon/8$, as through a slice, the positions
are taken at their middle. Around the middle $\bar z$ of the rates,
$e^{-z_j t}$ is interpolated between $L$ Chebyshev points $t_l$ of the window,
with Lagrange basis $B_l$:

$$
s_c(t) \approx e^{-\bar z t} \sum_{l=1}^{L} B_l(t)
\sum_q e^{-2\pi i\,\mathbf{k}(t)\cdot\mathbf{r}_q}\,S_{cl}(q),
\qquad
S_{cl}(q) = \sum_{j \in q} R_{jc}\,M_{xy,j}\,e^{-(z_j - \bar z)\,t_l}.
$$

The lattice sums $S_{cl}(q)$ are exact. The sum over the lattice at each
sample's $\mathbf{k}$ is a type-2 non-uniform FFT, which FINUFFT (Barnett,
Magland and af Klinteberg, SIAM J Sci Comput 2019) computes for every coil and
Chebyshev point. $L$ is the fewest points that interpolate $e^{-zt}$ to within
$\varepsilon/4$ for every rate on the boundary of the rectangle the rates
span, where the error of the interpolation is largest. FINUFFT is given
$\varepsilon/4$ divided by the Lebesgue constant of the points and by the most
the sums grow over the window from the middle rate, and works in single
precision where that lies above $2 \times 10^{-6}$. Where it does, the lattice
sums are formed from the sensitivities rounded to single precision, read from
a copy in the order of the lattice points that is made at the first such window
and kept: 8 bytes per isochromat and coil, beside the 16 the sensitivities take
in double precision. The cost is proportional
to the isochromats times the coils and $L$, and to the coils times $L$ times
the lattice points, rather than to the isochromats times the coils and the
samples; a window is read this way where that costs less. Either way, the
window leaves each isochromat as it stands at the last sample.

## RF pulses

During an RF pulse $b_1$ varies, and the field no longer points along $z$. The
engine holds $b_1$ constant over steps of duration $\Delta t$. Step $k$ turns
the magnetisation by $2\pi\,|\mathbf{b}_k|\,\Delta t$ about $\mathbf{b}_k$,
whose $z$ component is the mean over the step,
$\mathbf{r}\cdot\Delta\mathbf{A}_k/\Delta t + \Delta f$. Relaxation over the
step is split into two half steps around the rotation,

$$
\mathbf{M} \leftarrow E(\tfrac{\Delta t}{2})\,R_k\,E(\tfrac{\Delta t}{2})\,\mathbf{M},
$$

a symmetric splitting whose error is of second order in $\Delta t$. The steps
compose into one affine map, $\mathbf{M} \leftarrow A\,\mathbf{M} + \mathbf{c}\,M_0$,
per pulse.

The map depends on the isochromat only through $\Delta f$, $T_1$, $T_2$, the
transmit sensitivities and $\mathbf{r}\cdot\Delta\mathbf{A}_k$. When the steps'
gradient areas all lie along one direction, as they do under a slice-selection
gradient, the last depends on the position along that direction alone.
Isochromats equal in the others and within $10^{-12}$ m along that direction
share one map. For a slice of a phantom with a few tissues, a pulse then costs
a few maps rather than one per isochromat.

A pulse whose waveform is an earlier pulse's times $e^{i\varphi}$, played under
the same gradient, turns every step's transverse field by $\varphi$ about $z$,
and relaxation is symmetric about $z$, so its maps are the earlier ones turned
by $R_z(\varphi)$: $A \leftarrow R_z(\varphi)\,A\,R_z(\varphi)^{\mathsf T}$ and
$\mathbf{c} \leftarrow R_z(\varphi)\,\mathbf{c}$. The engine keeps the maps of
the pulses it has played and applies them turned, which is exact, to a pulse
that matches one of them to within $10^{-12}$ of its peak. The pulses of an
RF-spoiled train differ in phase alone, so an object whose isochromats each
see their own off-resonance, such as a head in the field its susceptibility
adds, pays one map per isochromat once for each distinct pulse rather than for
every pulse played.

That map need not be computed for each isochromat either. A pulse played
without transmit sensitivities, under no gradient or one that holds its
amplitude $G$ along its direction $\hat{\mathbf{n}}$ throughout, gives each
isochromat one field along $z$ for the whole pulse,
$\nu = \Delta f + G\,\mathbf{r}\cdot\hat{\mathbf{n}}$, so its map depends on
$\nu$, $T_1$ and $T_2$ alone. Free precession over half the pulse on either
side, $P(\nu)$, varies with $\nu$ faster than anything else in the map; the
rest, $\tilde A = P^{-1} A P^{-1}$ and $\tilde{\mathbf{c}} = P^{-1}\mathbf{c}$,
varies only as the pulse's response does. The engine computes $\tilde A$ and
$\tilde{\mathbf{c}}$ on a grid of $\nu$, 64 points per $1/T$ for a pulse of
duration $T$, for each pair of relaxation times, interpolates between the
four points around each isochromat's $\nu$ with a cubic, and applies its own
$P(\nu)$ exactly. Against the map stepped for each isochromat, the
magnetisation after a slice-selective pulse agrees to within about $10^{-7}$ of
$M_0$. The grid is used where it costs fewer maps than the isochromats'
groups, as it does under a slice-selection gradient across a head. The maps it
gives each group are kept as a stepped pulse's are, so a pulse that differs
from one played by its phase alone reuses them; a pulse of the same waveform
otherwise reuses the grid.

With transmit sensitivities, a pulse whose channels all play one waveform
$w(t)$ times a weight $a_c$ of their own, as a pulse shimmed onto the channels
does, gives isochromat $j$ the transmit field $d_j\,w(t)$, with the drive
$d_j = \sum_c S_c(\mathbf{r}_j)\,a_c$. The drive's phase turns the map about
$z$ as a pulse's phase does, and its magnitude scales the field, so the map
depends on $\nu$, $|d_j|$, $T_1$ and $T_2$ alone, and the grid spans $|d|$ as
well. Its points along $|d|$ are as far apart in the turn the pulse makes,
$2\pi\,|d|\,\Delta t \sum_k |w_k|$, as its points along $\nu$ are in the
precession over the pulse, $2\pi/64$. The map turns with $|d|$ as fast as the
precession, none of which can be taken out along $|d|$, so it is interpolated
there with a quintic through six points, which keeps the agreement with the
stepped map at about $10^{-7}$ of $M_0$.

## Pulseq events as fields

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

## Repeated blocks

A scan often plays one sequence of blocks many times, changing only the phase
offsets of its pulses and ADC events, its phase-encoding gradients and the
direction of its readout.
{meth}`~pulserver.virtual.Isochromats.repetitions` plays such a scan without
integrating every block of every repetition.

Every event the engine plays maps an isochromat's magnetisation affinely, so a
repetition does too: $\mathbf{M} \leftarrow A_j\,\mathbf{M} + \mathbf{b}_j$ for
isochromat $j$, and its transverse magnetisation at the first sample of each
ADC window is $\mathbf{u}_j\cdot\mathbf{M} + v_j$, $\mathbf{M}$ taken at the
repetition's start. Four plays of the blocks, from no magnetisation and from a
unit magnetisation along each axis, give $A_j$, $\mathbf{b}_j$,
$\mathbf{u}_j$ and $v_j$ for every isochromat.

A repetition whose pulses have their phase offsets larger by $\varphi_n$ plays
every transverse field turned about $z$ by $\theta_n = -\varphi_n$, and
relaxation is symmetric about $z$, so its maps are the first repetition's
turned by $R_z(\theta_n)$, as a pulse's maps turn with its phase. In the frame
turned with the pulses, $\mathbf{m} = R_z(-\theta_n)\,\mathbf{M}$, repetition
$n$ takes $\mathbf{m}$ to $R_z(\theta_n - \theta_{n+1})\,(A_j\,\mathbf{m} +
\mathbf{b}_j)$.

A phase encoding is a gradient that differs from the first repetition's by a
waveform that is zero during every pulse and every ADC window and plays no
area by the start of any pulse. It turns the transverse magnetisation about
$z$ by the area it has played, $e^{-2\pi i\,\mathbf{a}\cdot\mathbf{r}}$, so it
leaves $A_j$ and $\mathbf{b}_j$ as they are and multiplies the transverse
magnetisation at a window's first sample by $e^{-2\pi i\,\mathbf{a}_n\cdot
\mathbf{r}_j}$, with $\mathbf{a}_n$ its area up to that sample. Under the
gradient held over the window, isochromat $j$ then turns and decays by one
factor $z_j$ from each sample to the next. The area $\mathbf{c}_n$ it leaves
over the repetition, zero where it is rewound, turns isochromat $j$ at the
repetition's end by $-2\pi\,\mathbf{c}_n\cdot\mathbf{r}_j$ as well:
repetition $n$ takes $\mathbf{m}$ to $R_z(\theta_n - \theta_{n+1} -
2\pi\,\mathbf{c}_n\cdot\mathbf{r}_j)\,(A_j\,\mathbf{m} + \mathbf{b}_j)$.

A turned readout differs from the first repetition's by a waveform zero
during every pulse, of no area by the start of any, and held through its ADC
window at $\Delta\mathbf{g}_n$: a radial spoke turned with its prephaser. It
acts as a phase encoding does up to the window's first sample and over the
repetition, and from one sample to the next it turns isochromat $j$ by
$e^{-2\pi i\,\Delta t\,\Delta\mathbf{g}_n\cdot\mathbf{r}_j}$ more, with
$\Delta t$ the dwell time: each repetition reads the window along its own
direction. The non-uniform FFT a window is read with then spreads each
isochromat onto grid points found anew for every repetition.

### Fixed points and transients

Where the phase offsets step by one increment, $\theta_{n+1} - \theta_n =
\delta$ for every $n$, no readout is turned and no area is left over a
repetition, the map in the turned frame is the same for every repetition,
$A'_j = R_z(-\delta)\,A_j$, and has the fixed point
$\mathbf{m}^*_j = (I - A'_j)^{-1}R_z(-\delta)\,\mathbf{b}_j$. A sequence file
holds a phase offset to about $10^{-5}$ rad, so $\delta$ is the mean step, and
every step must lie within $10^{-4}$ rad of it. The magnetisation is the fixed
point plus a transient, $\mathbf{m} = \mathbf{m}^*_j + \mathbf{d}$, and the
transient decays as $\mathbf{d} \leftarrow A'_j\,\mathbf{d}$.

In the frame of each repetition's pulses, before demodulation, the fixed
points send the same samples in every repetition but for the phase encoding:

$$
s_c(n, k) = \sum_j R_{jc}\,(\mathbf{u}_j\cdot\mathbf{m}^*_j + v_j)\,z_j^{\,k}\,
e^{-2\pi i\,\mathbf{a}_n\cdot\mathbf{r}_j}.
$$

When the phase encodings run along at most two axes along which the
isochromats take few coordinates, as on the lattice a phantom is sampled on,
the isochromats that share their coordinates along those axes form columns,
and $e^{-2\pi i\,\mathbf{a}_n\cdot\mathbf{r}_j}$ is the same for every
isochromat of a column. Each column's samples are summed once, over its
isochromats, by the non-uniform FFT a window is read with; each repetition's
samples are then a sum over the columns, and with two axes the sum along the
one of fewer distinct areas is taken once per area. The fixed points cost a
term per column and repetition rather than one per isochromat and repetition.

The transients are carried repetition by repetition, 16 repetitions of an
isochromat at a time. An isochromat whose transient falls below the tolerance
times its proton density is dropped, and its magnetisation stands at its fixed
point from then on. The transients decay with the relaxation times, so after a
few $T_1$ a scan costs the fixed points' samples alone.

### Tolerance

With a tolerance of zero, every isochromat is carried in double precision
through every repetition, and the samples agree with the blocks played one by
one to rounding. A tolerance above zero, relative to the sum of the magnitudes
of the terms a sample sums, reads the transients' windows by the narrowest
kernel that holds it, carries the magnetisation in single precision from
$10^{-4}$ on, and drops transients below it. The fixed points' samples are
summed by the widest kernel at any tolerance.

## Relation to KomaMRI

KomaMRI turns the magnetisation in the same sense, and plays a pulse's
frequency offset as a frame rotating in the same sense as here. Its Pulseq
reader adds a pulse's phase offset and phase shape to $b_1$ with the opposite
sign, refers the frequency offset's phase to the pulse's centre, and
demodulates a sample by the ADC phase offset alone, with the opposite sign and
without the ADC frequency offset or phase modulation. The two therefore agree,
up to their time discretisation, where every pulse has a real waveform, a
phase offset of 0 or π and no frequency offset, and every ADC has a phase
offset of 0 or π and no frequency offset or phase modulation. Elsewhere the
phase of a pulse or of the demodulation differs between them.

## What is not modelled

The engine computes the signal of an ideal system playing the sequence as
written: no gradient delays, eddy currents, gradient nonlinearity, concomitant
fields or receiver noise. Isochromats neither diffuse, flow nor move, and the
decay of a voxel's signal by intravoxel dephasing appears only where several
isochromats represent the voxel.

## See also

* {doc}`virtual-scanner` — the cache played on isochromats, the phantoms and
  the coils they are sampled with, and what a run establishes.
* {doc}`../api/virtual` — the isochromats, their repetitions and the
  simulation of the blocks a cache plays.
* {func}`pypulseqpp.sim_rf` — the off-resonance profile of one RF pulse.
