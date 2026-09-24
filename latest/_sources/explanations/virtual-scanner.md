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

{func}`~pulserver.virtual.trajectory` integrates those gradients, rotated by
each block's rotation, into the k-space location $\mathbf{k}$ of every ADC
sample, in 1/m along the logical axes. An excitation returns $\mathbf{k}$ to
zero at the RF magnitude peak and a refocusing pulse negates it there, by the
use the cache records for each RF event; each file of a chain starts from
zero. The test suite holds this trajectory to the one each fixture's file
designs, and to the one the enrichment states.

## Signal model

After an excitation of RF phase $\phi_e$ at its magnitude peak, the
magnetization is transverse, at the phantom's density $\rho(\mathbf{r})$
whatever the flip angle, with phase $-\phi_e - \pi/2$, and it precesses as
$\exp(-2\pi i\,\mathbf{k}\cdot\mathbf{r})$. A refocusing pulse of phase
$\phi_r$ conjugates it about $-\phi_r$. The RF phase at the peak is the
pulse's phase offset plus its frequency offset times the time since the pulse
began. Coil $c$, of sensitivity $s_c(\mathbf{r})$, receives

$$
S_c(t) = e^{i\psi} \int \rho(\mathbf{r})\, s_c(\mathbf{r})\,
e^{-2\pi i\,\mathbf{k}(t)\cdot\mathbf{r}}\, d\mathbf{r},
$$

with $\psi$ the phase the RF pulses left, and the playout demodulates it by
$\exp(i\theta(t))$, where the receiver phase $\theta$ is the ADC phase offset at
the ADC's start, advancing at its frequency offset, plus its phase
modulation. These are the
conventions under which the field-of-view translation applied when the IR is
built ({doc}`ir-cache`) recentres an object: an object at the prescribed
offset is acquired as the same object at the centre. The phantom's ellipses
and its sensitivities, sums of plane waves, have analytic transforms, so
$S_c$ is evaluated exactly at every sample.

Relaxation, off-resonance, diffusion, slice profiles and every RF use other
than excitation and refocusing are not modelled. A readout before the first
excitation of its file acquires zeros.

## What a run establishes

- The trajectory the cache plays is the one each file designs, for the
  Cartesian, EPI, spiral and radial fixtures.
- The enrichment states that trajectory for every readout.
- An object at the prescribed offset is acquired centred by a Cartesian
  gradient echo and spin echo, and by EPI and spiral readouts, which the
  translation gives a phase modulation; an object away from it is not.
- A series streamed through the proxy is reconstructed into the phantom's
  image, at the prescribed position, and a series short of a readout is
  refused.

pypulseqpp's field-of-view translation moves a block that carries a rotation
extension by its unrotated gradients, so an off-centre design whose spokes are
rotations of one readout, as the ZTE fixture's are, is acquired off its ideal;
the test suite records it as an expected failure.

## See also

* {doc}`ir-cache` — the IR, its prescription and its playback.
* {doc}`../user-guide/reconstruction-client` — the stream the virtual reconstruction client sends.
* {doc}`../api/virtual` — the phantom, the acquisition and the client.
