"""
======================================
Raw-data enrichment and reconstruction
======================================

The scope of this notebook is to enrich a simulated series the way the
reconstruction proxy does, and to reconstruct it with a reconstruction plugin:
what the sequence supplies to an MRD stream that carries no encoding
information, and the displacement that demodulation to the prescription centre
removes.

The series is simulated rather than acquired: a phantom of three disks, whose
k-space is known analytically, sampled at the k-space locations of a
pypulseqpp 2D gradient echo and placed away from the isocentre. The enrichment
is described in :doc:`/explanations/reconstruction`.

Outline:

#. **Sequence and readout table.** The readouts of the sequence, as the proxy
   tabulates them from a revision.
#. **Simulated series.** A phantom away from the isocentre, streamed without
   encoding counters, flags or encoding spaces.
#. **Enrichment.** The header and acquisition fields the table supplies.
#. **Reconstruction.** The shipped Cartesian FFT plugin, with and without
   demodulation to the prescription centre.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 8.6  # inches, the width of the documentation column
# sphinx_gallery_end_ignore

# %%
# Sequence and readout table
# --------------------------
#
# The sequence is written as a generated revision holds it, in the binary
# Pulseq form, and tabulated with :class:`~pulserver.vre.SequenceTable`: one
# row per readout in play order, with its encoding counters, flags, dwell time
# and the k-space location of every sample in 1/m.

import tempfile
from pathlib import Path

import ismrmrd
import ismrmrd.xsd as xsd
import numpy as np
import pypulseqpp as pp
from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp

from pulserver.vre import (
    FOV_OFFSET_PARAMETER,
    SequenceTable,
    enrich_acquisition,
    enrich_header,
    fov_offset_m,
)

system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
work = Path(tempfile.mkdtemp())

app = Gre2DApp(system, n_x=64, n_y=64, te=None, tr=None, n_dummy=0)
files = app.write(work / "sequence.seq", offline=False)
table = SequenceTable.read(files[0])

print(f"{len(table)} readouts of {table.num_samples[0]} samples")
print("LIN of the first readouts:", table.counters["LIN"][:6])
print("encoding spaces:", table.spaces)

# %%
# Simulated series
# ----------------
#
# The phantom is a disk of radius 50 mm containing two smaller disks. The
# Fourier transform of a disk of radius :math:`R` and unit intensity centred
# at :math:`\mathbf{c}` is
#
# .. math::
#
#    \frac{R\, J_1(2\pi R |\mathbf{k}|)}{|\mathbf{k}|}\,
#    e^{-i 2\pi \mathbf{k}\cdot\mathbf{c}},
#
# so the signal of every sample follows from its k-space location without a
# discretized object. The phantom is displaced by
# :math:`\mathbf{d} = (30, -90)` mm from the isocentre, which exceeds half the
# 220 mm field of view along the phase-encoding axis.

from scipy.special import j1

DISKS = ((0.0, 0.0, 0.05, 1.0), (0.02, 0.01, 0.015, -0.5), (-0.02, -0.015, 0.01, 0.5))
offset = np.array([0.03, -0.09, 0.0])


def phantom_signal(k, shift):
    radius_k = np.hypot(k[0], k[1])
    signal = np.zeros(k.shape[1], complex)
    for cx, cy, radius, intensity in DISKS:
        with np.errstate(invalid="ignore", divide="ignore"):
            disk = np.where(
                radius_k > 0,
                radius * j1(2 * np.pi * radius * radius_k) / radius_k,
                np.pi * radius**2,
            )
        phase = np.exp(-2j * np.pi * (k[0] * (cx + shift[0]) + k[1] * (cy + shift[1])))
        signal += intensity * disk * phase
    return signal.astype(np.complex64)


samples = phantom_signal(table.k, offset)

# %%
# The stream carries what a vendor reconstruction client supplies: one
# receiver channel, the readout samples, and a header whose only
# pulserver-specific entry is the prescription centre in mm.


def vendor_stream(centre_mm):
    header = xsd.ismrmrdHeader(
        experimentalConditions=xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=127_000_000
        ),
        encoding=[],
    )
    header.acquisitionSystemInformation = xsd.acquisitionSystemInformationType(
        receiverChannels=1
    )
    if centre_mm is not None:
        header.userParameters = xsd.userParametersType(
            userParameterString=[
                xsd.userParameterStringType(
                    name=FOV_OFFSET_PARAMETER,
                    value=" ".join(f"{v:g}" for v in centre_mm),
                )
            ]
        )
    acquisitions = []
    for row in range(len(table)):
        start, count = int(table.sample_offset[row]), int(table.num_samples[row])
        data = samples[start : start + count][np.newaxis]
        acquisitions.append(ismrmrd.Acquisition.from_array(data))
    return header, acquisitions


header, acquisitions = vendor_stream(offset * 1e3)
print("encodings in the header:", len(header.encoding))
print("LIN of acquisition 10:", acquisitions[10].idx.kspace_encode_step_1)
print("flags of acquisition 63:", acquisitions[63].flags)

# %%
# Enrichment
# ----------
#
# :func:`~pulserver.vre.enrich_header` describes the table's encoding space in
# the header. :func:`~pulserver.vre.enrich_acquisition` applies one table row
# to each acquisition, in stream order, and demodulates it to the prescription
# centre the header records.

enrich_header(header, table)
centre = fov_offset_m(header)
for row, acquisition in enumerate(acquisitions):
    enrich_acquisition(acquisition, table, row, centre)

encoding = header.encoding[0]
matrix = encoding.encodedSpace.matrixSize
fov = encoding.reconSpace.fieldOfView_mm
print(
    f"encoded matrix {matrix.x} x {matrix.y} x {matrix.z}, FOV {fov.x:g} x {fov.y:g} mm"
)
print("phase-encoding limit:", encoding.encodingLimits.kspace_encoding_step_1.maximum)
print("LIN of acquisition 10:", acquisitions[10].idx.kspace_encode_step_1)
print(
    "acquisition 63 closes the slice:",
    acquisitions[63].isFlagSet(ismrmrd.ACQ_LAST_IN_SLICE),
)

# %%
# Reconstruction
# --------------
#
# The enriched stream is written to an ISMRMRD file and reconstructed in this
# process by the shipped two-dimensional Cartesian FFT plugin,
# :meth:`~pulserver.recon.ReconPlugin.run` driving the same hooks the proxy's
# workers drive. The plugin makes one image per slice when an acquisition
# carries ``LAST_IN_SLICE``, a flag enrichment supplies, and crops the
# oversampled readout to the reconstruction matrix of the header.
#
# The same series is reconstructed twice: with the prescription centre in the
# header, and without it, in which case no demodulation is applied.

from pulserver.recon.handlers.simplefft import SimpleFftRecon


def reconstruct(centre_mm, name):
    header, acquisitions = vendor_stream(centre_mm)
    enrich_header(header, table)
    centre = fov_offset_m(header)
    for row, acquisition in enumerate(acquisitions):
        enrich_acquisition(acquisition, table, row, centre)
    path = work / name
    dataset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=True)
    dataset.write_xml_header(xsd.ToXML(header))
    for acquisition in acquisitions:
        dataset.append_acquisition(acquisition)
    dataset.close()
    images = [
        out for out in SimpleFftRecon().run(str(path)) if isinstance(out, ismrmrd.Image)
    ]
    return np.squeeze(images[0].data)


uncorrected = reconstruct(None, "uncorrected.h5")
centred = reconstruct(offset * 1e3, "centred.h5")

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH * 0.8, 3.8), layout="constrained")
for ax, image, title in (
    (axes[0], uncorrected, "no prescription centre"),
    (axes[1], centred, "demodulated to d = (30, -90) mm"),
):
    ax.imshow(image, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
# sphinx_gallery_end_ignore

# %%
# Without demodulation the phantom appears at its displacement from the
# isocentre: shifted along the readout axis, and folded along the
# phase-encoding axis, where the 90 mm displacement exceeds half the field of
# view. Multiplying each readout by :math:`e^{+i 2\pi \mathbf{d}\cdot\mathbf{k}}`
# places it at the centre of the field of view, with no change to the
# trajectory.
