# Calibration, noise and coil compression

```{admonition} TL;DR
:class: tldr

- Sensitivity maps estimated from whitened and compressed data describe only
  data with the same channels, whitening, compression and voxel grid.
- Prewhitening and coil compression are linear transforms of the channels;
  bartorch computes them and the estimate, and `pulserver.recon` holds,
  compares and routes the results.
- Stored maps are reused only where every recorded field matches the unit;
  nothing is resampled, and a mismatch is reported.
```

A reconstruction from a receive-coil array needs the sensitivity of each
channel, estimated from calibration data, and combines the channels with equal
weight, which is the maximum-likelihood combination only where their noise is
uncorrelated and of equal variance. Prewhitening makes the noise of the
channels so, and coil compression reduces their number. Each of the three is
expressed in a basis of the channels, and a quantity estimated in one basis
does not describe data in another. This page states the model, the conditions
under which ``pulserver.recon`` reuses sensitivity maps, and the interfaces
that hold them. The transforms and the estimate are computed by bartorch; the
reconstruction plugin passes the estimate to {func}`~pulserver.recon.coil_maps`.

## Signal model

The signal of each channel $c$ of $N_c$ is the image $x$ weighted by the
channel's sensitivity $S_c$ on the voxel grid, Fourier encoded, plus complex
Gaussian noise $\mathbf{n}$ of channel covariance
$\Psi = \mathrm{E}[\mathbf{n}\mathbf{n}^H]$: the encoding model bartorch's
[encoding](https://pulserver.github.io/bartorch/latest/explanation/encoding.html)
page states. A linear transform $B$ of the
channels, $\tilde{\mathbf{y}} = B\mathbf{y}$, follows the same model with the
sensitivities $B\mathbf{S}(\mathbf{r})$, $\mathbf{S} = (S_1, \dots, S_{N_c})$,
and the noise covariance $B\Psi B^H$. Two transforms are used:

- prewhitening, $B = W$ with $W\Psi W^H = I$, after which the noise of the
  channels is uncorrelated with unit variance;
- coil compression, $B = A$, the $n \times N_c$ matrix whose rows are the
  conjugated eigenvectors of the channel covariance of the calibration data
  that belong to its $n$ largest eigenvalues, so that the $n$ virtual channels
  are uncorrelated and ordered by decreasing power.

Data that is whitened and then compressed is transformed by $AW$, and
sensitivities estimated from it are $AW\mathbf{S}$ on the voxel grid of the
calibration k-space. A set of maps therefore describes only data with the same
channels in the same order, the same whitening, the same compression and the
same voxel grid at the same position and orientation.

## Noise and prewhitening

Readouts flagged `IS_NOISE_MEASUREMENT` sample $\mathbf{n}$ with the RF
off. From $K$ samples the covariance is estimated as
$\hat\Psi = (K-1)^{-1}\sum_k \mathbf{n}_k\mathbf{n}_k^H$, and $W = L^{-1}$ for
its Cholesky factorisation $\hat\Psi = LL^H$. Estimating $\hat\Psi$ takes more
noise samples than channels.

The variance of the noise in a sample is inversely proportional to the dwell
time. A readout of dwell time $\tau$ is multiplied by $\sqrt{\tau/\tau_0}\,W$,
$\tau_0$ being the dwell time of the noise readouts, and by $W$ alone where
either dwell time is not stated. No other difference between the noise
measurement and the readout is modelled.

{class}`~pulserver.recon.Prewhiten` takes $W$ from the first of these:

1. the noise readouts that precede the first readout of any other kind in the
   stream;
2. the exam's `NOISE_COVARIANCE`, stored by a noise series, when its coil
   labels are those of the header;
3. none, in which case readouts pass unchanged and a warning is logged once,
   or {class}`~pulserver.recon.MissingCalibration` is raised for
   `required=True`.

Noise readouts after the first readout of another kind are consumed and not
used. The gadget applies $W$ on arrival, so the imaging and the calibration
readouts of every unit are whitened alike.

## Coil compression

The basis $A$ is that of the channel covariance
$\sum_k \mathbf{y}_k\mathbf{y}_k^H$ over the placed samples of the calibration
k-space of the first unit the stream compresses, or of its imaging k-space
where that unit holds no calibration k-space. One basis is estimated per
stream, and every unit is projected onto it, so the virtual channels of
successive units are the same.

If the signal of the channels lies in a subspace of dimension at most $n$,
compression discards none of it, and SENSE with compressed data and compressed
maps reconstructs the image of the full set of channels to round-off.
Otherwise the discarded signal is that of the $N_c - n$ smallest eigenvalues.
Maps estimated from uncompressed data are in another basis than compressed
data, and are not transformed into this one.

## Coil sensitivity maps

### Calibration k-space

The calibration k-space of a unit is {attr}`ReconData.ref
<pulserver.recon.ReconData.ref>`. It comes from one of three arrangements.

| Arrangement | Readouts | Maps are estimated by |
| --- | --- | --- |
| Embedded calibration | Lines of the imaging space flagged `IS_PARALLEL_CALIBRATION_AND_IMAGING` | The imaging unit, which holds both `data` and `ref` |
| Calibration unit | Readouts flagged `IS_PARALLEL_CALIBRATION` in a unit with no imaging readouts, as in an encoding space of their own | That unit, whose `data.data` is `None`; the imaging units of the same slice take the maps from the stream |
| Calibration series | A series of its own | The reconstruction of that series, which stores the maps in the exam |

The estimate receives the calibration k-space of a Cartesian encoding space
zero-filled onto the whole grid of the space, the lines its readouts cover
being at their grid positions, and returns maps on that grid. For a
non-Cartesian space {func}`~pulserver.recon.coil_maps` raises `ValueError`.

### Resolution

{func}`~pulserver.recon.coil_maps` returns the maps of a unit from the first of
these sources, the location being the unit's `slice` counter:

1. The unit's own calibration k-space, from which the maps are estimated and
   stored for the slice, replacing any stored there.
2. The maps stored in the stream for the slice.
3. The exam's `COIL_SENSITIVITIES`, which holds one set of maps.
4. No source: {class}`~pulserver.recon.MissingCalibration` is raised, or
   `None` returned for `required=False`.

The slice is the location because a calibration unit and the imaging units that
use its maps may be in different encoding spaces. A unit that needs maps before
the unit that estimates them has closed finds none in the first three sources.

### Reuse

Stored maps are returned only to a unit for which every field of the
{class}`~pulserver.recon.CoilSensitivities` is equal to that of the unit. The
fields are compared in this order, and the first that differs is reported.

| Field | Equal to |
| --- | --- |
| `coils` | The coil labels of the header in channel order |
| `noise` | The identifier of the prewhitening in `context.noise`, or `None` on both sides |
| `basis` | The identifier of the basis in `context.coil_compression`, or `None` on both sides |
| `geometry.frame_of_reference` | The `frameOfReferenceUID` of the header |
| `geometry.table_position`, `geometry.position`, `geometry.read_dir`, `geometry.phase_dir`, `geometry.slice_dir` | The same fields of the reference acquisition of the unit's k-space, in the units the acquisition states |
| `geometry.fov`, `geometry.matrix` | The field of view in m and the matrix of the reconstruction space of the unit's encoding space |
| `geometry.grid` | The shape of the voxel axes of the whole k-space of the unit's encoding space |

`method` and `source`, the estimate and the `measurementID` of the series that
measured the maps, are recorded and appear in a refusal; they are not compared.
The comparison establishes that the maps and the unit share the listed
quantities. It does not test the accuracy of the maps.

Nothing is resampled, regridded or resliced to make maps fit. Maps on a grid or
at a position other than the unit's are not returned, and the message of
{class}`~pulserver.recon.MissingCalibration` names each source and why it was
not used:

```text
no coil sensitivities for slice 0; sources consulted:
  this unit: it holds no calibration readouts
  this series, slice 0: no maps stored
  this exam: geometry.matrix differs: stored (96, 96), this unit (192, 192) (estimated by nlinv_maps from measurement 7)
```

## Software abstraction

| Quantity | Interface | State |
| --- | --- | --- |
| Whitening $W$ | {class}`~pulserver.recon.Prewhiten`, a gadget | `context.noise`; the exam key {obj}`~pulserver.recon.NOISE_COVARIANCE` |
| Compression basis $A$ | {class}`~pulserver.recon.CoilCompression`, called from `recon` | `context.coil_compression` |
| Maps $\mathbf{S}$ | {func}`~pulserver.recon.coil_maps`, called from `recon`; {class}`~pulserver.recon.CoilSensitivities` | `context.coil_maps` by slice; the exam key {obj}`~pulserver.recon.COIL_SENSITIVITIES`, read and written as `context.coil_sensitivities` |

The state in `context` belongs to one stream. What a series leaves to the next
is what it stores in `context.exam` under the exam keys, which
{doc}`reconstruction` describes. A calibration series publishes its whitening
with {meth}`Prewhiten.publish <pulserver.recon.Prewhiten.publish>` and its maps
by assigning `context.coil_sensitivities`.

Whitening runs on arrival, in the `gadgets` of the plugin. Within `recon`, a
unit is compressed before its maps are requested, because the maps record the
basis in force when they are estimated. The whitening, the compression basis
and the estimate are bartorch's (`tools.whiten`, `tools.cc` and `tools.ccapply`,
and the estimate the plugin names, such as `apps.nlinv_maps`); the interfaces
above hold their results, compare them and route them between units and series.
{doc}`../user-guide/reconstruction-plugins` shows the calls.

## See also

* {doc}`../user-guide/reconstruction-plugins` — requesting maps, prewhitening and compressing in a plugin.
* {doc}`reconstruction` — units, the exam cache and workers.
* {doc}`../api/recon` — the interfaces named above.
