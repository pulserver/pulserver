# Bloch engine

A Bloch simulation computes the magnetisation that a sequence leaves in an
object and the signal it induces in the receive coils. The object is
represented by isochromats: spin packets that share one position, one
precession frequency and one pair of relaxation times. The virtual scanner
plays the blocks of an IR cache on isochromats
({class}`~pulserver.virtual.Isochromats`, {func}`~pulserver.virtual.simulate`).
The equation the engine integrates and how each Pulseq event enters it are
stated in {doc}`../../explanations/virtual-scanner`; this page states how the
engine integrates free precession and RF pulses, how repeated blocks are
played without integrating each of them, and how voxels, motion and diffusion
are represented.

## Isochromats of a phantom

{meth}`~pulserver.virtual.Phantom.isochromats` samples the phantom on a square
grid aligned with its axes. Each ellipse contributes the points of the grid
inside it, each an isochromat of proton density the ellipse's intensity times
the area of a grid cell, relaxing with the ellipse's $T_1$ and $T_2$ and
precessing at its $f_\sigma$; the coil sensitivities are sampled at the same
points. The sum over the isochromats approximates each ellipse's transform
below the grid's Nyquist frequency, $1/(2\Delta)$ for a spacing $\Delta$, and
converges to it as the grid is refined, so a grid several times finer than the
image's pixel acquires the analytic phantom wherever the signal model holds.

An isochromat that no excitation pulse tips from rest adds nothing to the
readouts, so a scan is simulated on the isochromats its excitation pulses
excite. {func}`~pulserver.virtual.excited` finds their slabs: each pulse the
cache labels an excitation is played, at its nominal amplitude and at twice
it, on a line of isochromats along its gradient, turned as the scan turns it,
and its slab holds the fields at which it changes the magnetization at rest by
at least a hundredth of the most it changes it, sidelobes included. An
isochromat at $\mathbf{r}$ precessing at $f$ sees the field
$\mathbf{g} \cdot \mathbf{r} + f$ during a pulse played under the gradient
$\mathbf{g}$, so a chemical shift or the subject's field moves an isochromat's
slab as it moves the slice, and each slice of a multislice scan is a slab of
its own. A pulse played without a gradient, or under one that changes during
it, excites the whole phantom, as a volumetric excitation does. So do pulses
played under more than 16 gradients, as a ZTE scan plays one under each
spoke's: between them, their slabs leave out next to none of the object. What other
pulses tip into the transverse plane outside the slabs, such as the free
induction decay of an imperfect refocusing pulse, is left out. The phantoms'
`isochromats` keep those a region, such as {class}`~pulserver.virtual.Slabs`,
answers for.

Played on isochromats, the cache samples what the design samples when its
blocks, turned as the checks turn it, are played one by one on the same
isochromats with the gradients `pypulseqpp.Sequence.waveforms` reports, to the
single precision of the cache's waveforms. A gradient turned by a block's
rotation is stored as the rotation leaves it, in single precision, and in a
sequence that leaves its transverse magnetization unspoiled from one repetition
to the next, the phase that rounding accrues is what separates the two.

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

Isochromats given a CUDA device read every such window there instead. The
engine finds the lattice, the Chebyshev points, the basis and each sample's
coordinates, as above, and hands the device the window's transverse
magnetisation; a Triton kernel forms the lattice sums $S_{cl}(q)$ for a block
of coils at a time, and cuFINUFFT transforms them, planned as FINUFFT is, in
the same precision. The sensitivities are held on the device in the order of
the lattice points from the first window on. The engine takes each
isochromat to the last sample itself, so a window leaves the same state
wherever it is read.

Every other window outside a run, which the engine would read by its own
transform or sample by sample, is summed term by term on the device: the sum
$s_c(t)$ above, at every sample. This covers a window under a held gradient,
as a Cartesian line; one whose isochromats lie off any lattice, as a moving
subject's or a scattered phantom's do; and one whose lattice has more points
than a window is summed onto, as an oblique slab's can. A second
Triton kernel takes the isochromats in blocks of 64, in the order of a
Z-order curve through their positions, so that each block's lie close
together. It forms each term's phase, in cycles, as its offset from the
block's centre times $\mathbf{k}$ plus its off-resonance from the middle one
times $t$, in the sums' precision, and adds the centre's phase
$\mathbf{k}\cdot\mathbf{r}_{\text{centre}}$ and the middle off-resonance's,
computed in double precision and reduced to one cycle. Each phase is then
rounded in proportion to the block's extent rather than to the field of
view. The sums are formed in single precision at a tolerance of $10^{-5}$
or more, unless rounding the turn of the off-resonance farthest from the
middle over the window would reach half the tolerance. The order, the blocks'
centres and the sensitivities in that order are found once for each
revision of the positions, so a moving subject's are found again at each
block. The cost is proportional to the isochromats times the samples times
the coils, as the engine's own reading sample by sample is. A window smaller
than a least size, $2^{22}$ isochromats times samples times coils by default
on a CUDA device, is left to the engine, which reads it before the device
would have started.

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

Groups are made for one gradient direction, and the engine keeps those of four
directions. Once it holds four, a pulse under a direction none of them is made
for, as each spoke of a ZTE scan plays under its own readout gradient, is
played from the grid isochromat by isochromat: each map is interpolated at the
isochromat's own $\nu$ and $|d_j|$ and applied at once, with no grouping along
the new direction and no maps kept for a later pulse. A pulse the grid cannot
serve with fewer new points than there are isochromats is grouped instead.

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

A first block that plays its pulse under a gradient held through the block,
larger by $\Delta\mathbf{g}_n$ than the first repetition's, as each spoke of a
ZTE scan plays its pulse under its own readout gradient, gives isochromat $j$
the field $\nu_j + \Delta\mathbf{g}_n\cdot\mathbf{r}_j$ through the block, and
a map that differs from one repetition to the next by more than a turn about
$z$. Each repetition then applies the block's map read off the pulse's grid at
that field and $|d_j|$, as a pulse past the groupings kept is, with the free
precession at that field before and after the pulse. The four plays give the
maps of the blocks after it, and the areas the difference leaves are those
over these blocks. The grid spans the fields of every repetition; a pulse it
cannot serve with fewer points than the isochromats times the repetitions is
refused.

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

### Blocks played between repetitions

Blocks played between two repetitions of a run leave a magnetisation the run
did not carry, as each shell's closing spoke and the ramp onto the next
shell's first spoke do between the spokes of a ZTE scan's shells. The run
resumes from it, turned into the frame of its next repetition's pulses,
$\mathbf{m} = R_z(-\theta_n)\,\mathbf{M}$. Its maps depend on the blocks of a
repetition alone, so they hold, and the scan pays for them once rather than
for every shell. A run split into fixed points and transients does not
resume: an isochromat whose transient was dropped no longer carries one.

### Runs carried on a device

Isochromats given a CUDA device carry a run there from its first play. The
engine hands the device, for each isochromat it carries, the magnetisation,
the maps, each window's coefficients, the limit below which its transient is
dropped, the coordinates its phase encodings depend on, and the grid points
and weights by which each window's non-uniform FFT spreads it, in the
precision the engine carries them in; where the first block's pulse is read
off its grid, it hands the grid, each isochromat's field and position, and the
weights of the grid's rows around its drive. For each tile of 16 repetitions,
a Triton kernel carries every isochromat through the tile, reading such a
pulse's map off the grid at each repetition, and writes its transverse
magnetisation at each window's first sample, times the phase encoding, at
every repetition. A second kernel spreads these onto each
window's grid. For each grid point, T2 class and block of coils, it sums the
isochromats of that class whose kernel reaches the point, grouped by the first
grid point they spread onto, as products of a matrix of their weights times
their sensitivities, a row per coil, by one of their transverse
magnetisations, a column per repetition. A turned window spreads each
isochromat onto grid points found anew for every repetition, from its
position and the repetition's readout as the engine finds them, in double
precision. At each repetition the isochromats are sorted by T2 class and the
first grid point they reach, and each block of 32 grid points sums those that
reach it as products of a matrix of their weights times their transverse
magnetisations, a row per grid point, by one of their sensitivities, a column
per coil. The engine reads the grids as it reads its own, and the device
writes the magnetisation back to the engine at the end of each play and takes
the engine's where the run resumes.
Transients are dropped as the engine drops them, and once more than a quarter
have been dropped the device keeps only the rest. A run of fewer than
$2^{18}$ isochromats times coils, by default, and one whose arrays would take
more than half the memory free on the device, are carried by the engine.

## Voxels, motion and diffusion

### A voxel's isochromats

A phantom's voxel is one isochromat by default, at the voxel's centre
(`spins=1`). With several (`spins` of
{meth}`Phantom.isochromats <pulserver.virtual.Phantom.isochromats>` and
{meth}`BrainWeb.isochromats <pulserver.virtual.BrainWeb.isochromats>`), they
share the voxel's proton density and differ in two respects.

*Position.* A `"point"` voxel keeps them at its centre; a `"box"` voxel places
them at the centres of a grid of cells filling it, $m$ to a side. A gradient
that winds the phase through a whole number of cycles across a box voxel along
one of its axes, and not a multiple of $m$, leaves the voxel no signal, as it
leaves a uniform voxel none; a point voxel, and a lattice of them, keeps its
signal under any such winding. The cells of all voxels lie on one lattice,
$m$ times finer, on which windows under a changing gradient are still read.

*Frequency.* T2′ is the decay of a voxel's signal by a static distribution of
frequencies within it, which a spin echo refocuses. A Lorentzian line of half
width $1/(2\pi T_2')$ gives the decay $e^{-|t|/T_2'}$ of the free induction and
$e^{-|t - T_E|/T_2'}$ about a spin echo at $T_E$. The isochromats of a voxel
take one frequency each from strata of equal probability of that line, the
strata shifted together by one uniform draw per voxel: over many voxels the
mean signal decays as $e^{-|t|/T_2'}$, while each voxel's own decay departs
from it by a spread that falls with the number of isochromats. The line is cut
at 32 half widths, which keeps a
voxel's frequencies within $\pm 32/(2\pi T_2')$ of its own and takes 2% of
the line, so that the decay starts quadratically for $|t| \lesssim T_2'/32$.
One isochromat per voxel precesses at the voxel's frequency, whatever its T2′.
BrainWeb's tissue classes take the T2′ that the T2 and T2* of BrainWeb's
simulator leave at 1.5 T, $R_2' = R_2^* - R_2$, with $R_2'$ in proportion to
the field, as in the static dephasing regime; their T1 grows with the field as
measured for white and grey matter, skeletal muscle and adipose tissue, and
CSF's holds ({meth}`BrainWeb.relaxation <pulserver.virtual.BrainWeb.relaxation>`).

### Motion

A subject moves as a function of the scan clock: `motion(t, positions)`
returns where the positions at rest lie at time $t$,
{class}`~pulserver.virtual.RigidMotion` for a rigid body. Each isochromat
keeps its proton density, relaxation times, off-resonance and transmit and
receive sensitivities as it moves, so that the sensitivities move with the
subject rather than stay with the coils.

Before each block the isochromats are placed where the motion puts them at the
block's start, and they are read there through the block. After the block,
each is turned by the phase the gradient adds along its path,

$$
\phi = 2\pi \int_{t_s}^{T} \mathbf{G}(t)\cdot\bigl(\mathbf{r}(t) - \mathbf{r}(0)\bigr)\,dt,
$$

from the end of the block's RF pulse $t_s$, or its start without one, to its
end $T$, integrated by three Gauss–Legendre points on each stretch of at most
0.5 ms on which the gradient is linear. A block without RF therefore leaves
each isochromat exactly as its motion through the block would, the first
moment of a bipolar gradient included; samples within a block, and the phase
accrued before the end of a pulse, are those of the isochromats where they
stood at the block's start.

### Diffusion

An isochromat with a diffusion coefficient $D$ follows a Brownian walk
$\mathbf{w}(t)$ of its own, of independent increments of variance $2D\,dt$
along each axis. The walk acts through the phase alone: the isochromat keeps
its position, and after each block it is turned by
$2\pi\int_{t_s}^{T}\mathbf{G}(t)\cdot\mathbf{w}(t)\,dt$. With $K(t)$ the area
the gradient plays from $t$ to $T$, that phase is
$2\pi\,K(t_s)\cdot\mathbf{w}(t_s) + 2\pi\int_{t_s}^{T} K(t)\cdot
d\mathbf{w}(t)$; the second term and the walk's displacement over the block
are jointly normal, of variances $2D\int K^2\,dt$ and $2D(T-t_s)$ and
covariance $2D\int K\,dt$ per axis, and are drawn together. The phase is
therefore exact in distribution, however the gradient is cut into blocks, and
the mean over a voxel's isochromats of their transverse magnetisation after a
diffusion encoding is $e^{-bD}$ times its value without one, with
$b = \int (2\pi k(t))^2\,dt$. Its spread about that mean is about
$1/\sqrt{2N}$ for $N$ isochromats, so that a voxel's attenuation needs
many isochromats to be resolved. The walk does not carry magnetisation from
one place to another.

Isochromats that move or diffuse play block by block: runs of repetitions are
not used for them.

## Relation to MRzero

MRzero computes the same signal from a phase distribution graph: each state of
the magnetisation carries the time and the gradient area since it was
dephased, from which it applies $e^{-|\tau|/T_2'}$, $e^{-bD}$ and a voxel's
dephasing function exactly, and moving voxels accrue the phase of their path
evaluated at each event. Its isochromat simulation, against which it checks
the graph, spreads each voxel over isochromats at shared positions within a box
and at shared quantiles of the Lorentzian line. Here the same voxel
properties act through isochromats: T2′ and a box voxel through the isochromats
of a voxel, motion through their positions and phases, and diffusion through
each one's walk. Where a voxel holds many isochromats, the two agree to the
spread of the voxel's isochromats about their mean.

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
fields or receiver noise. The decay of a voxel's signal by intravoxel
dephasing and by T2′ appears only where several isochromats represent the
voxel, and diffusion attenuates a voxel's signal only on average over its
isochromats.

## See also

* {doc}`../../explanations/virtual-scanner` — the cache played on isochromats, the phantoms and
  the coils they are sampled with, and what a run establishes.
* {doc}`../../api/virtual` — the isochromats, their repetitions and the
  simulation of the blocks a cache plays.
* {func}`pypulseqpp.sim_rf` — the off-resonance profile of one RF pulse.
