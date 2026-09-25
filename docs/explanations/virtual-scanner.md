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
| Physics | Magnet, coils and subject | {class}`~pulserver.virtual.Phantom`, {func}`~pulserver.virtual.acquire` | The trajectory the enrichment has to state, the demodulation of the prescription |
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
the instance at the cursor.

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
it ({class}`~pulserver.virtual.Phantom`). After an excitation of RF phase
$\phi_e$ at its centre, the magnetization is transverse, at the phantom's
density $\rho(\mathbf{r})$ whatever the flip angle, with phase
$-\phi_e - \pi/2$, and it precesses as
$\exp(-2\pi i\,\mathbf{k}\cdot\mathbf{r})$, with $\mathbf{r}$ and
$\mathbf{k}$ along the physical axes. A refocusing pulse of phase
$\phi_r$ conjugates it about $-\phi_r$. The RF phase at the centre is the
pulse's phase offset plus its frequency offset times the time since the pulse
began. Coil $c$, of sensitivity $s_c(\mathbf{r})$, receives

$$
S_c(t) = e^{i\psi} \int \rho(\mathbf{r})\, s_c(\mathbf{r})\,
e^{-2\pi i\,\mathbf{k}(t)\cdot\mathbf{r}}\, d\mathbf{r},
$$

with $\psi$ the phase the RF pulses left, and the playout demodulates it by
$\exp(i\theta(t))$, where the receiver phase $\theta$ is the ADC phase offset at
the ADC's start, advancing at its frequency offset, plus its phase
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

Relaxation, off-resonance, diffusion, slice profiles and every RF use other
than excitation and refocusing are not modelled. A readout before the first
excitation of its file acquires zeros.

## What a run establishes

- The trajectory the cache plays is the one each file designs, for the
  fixtures and for every sequence pypulseqpp ships, at small sizes, and under
  an oblique and a reflected prescription the one `pypulseqpp.TransformFOV`
  turns the design into for the checks. A block labelled `NOROT` plays turned
  by its own rotation alone.
- The cache carries the RF centre a design records where it is away from the
  magnitude peak.
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

## See also

* {doc}`ir-cache` — the IR, its prescription and its playback.
* {doc}`../user-guide/reconstruction-client` — the stream the virtual reconstruction client sends.
* {doc}`../api/virtual` — the phantom, the acquisition and the client.
