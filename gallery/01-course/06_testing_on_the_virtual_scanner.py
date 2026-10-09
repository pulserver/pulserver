"""
==========================================
6. Test your pair on the virtual scanner
==========================================

You now have a sequence plugin (lesson 2) and a recon plugin (lesson 5). On a
scanner they are two files in two folders, and the operator uses them without
knowing they are yours. In this lesson you put them in those files, load them
into the virtual scanner as a scanner would, and scan with them. Then you
check the images against the physics: if the flip angle, the timing or the
reconstruction were wrong, the contrast would show it.

**Learning objectives**

- Save your plugins as the files a scanner loads.
- Run the operator's flow with them: protocol, edit, scan, image.
- Compare the contrast of a flip-angle series with the closed-form signal of
  a spoiled gradient echo.

Previous: :doc:`05_recon_plugin`. This is the last lesson of the course; the
Tours start from here.
"""

# sphinx_gallery_start_ignore
import logging

import matplotlib.pyplot as plt

# The exam has no noise scan; the reconstruction's note about it is not part
# of the lesson.
logging.disable(logging.WARNING)
# sphinx_gallery_end_ignore

# %%
# Your plugins, as files
# ----------------------
#
# A sequence plugin is a ``.py`` file in a folder of sequence plugins, and
# the operator sees it under the file's name. A recon plugin is a ``.py``
# file in a folder of recon plugins, with a module-level ``PLUGIN``. Here are
# the plugin of lesson 2, with its ``evaluate``, and the plugin of lesson 5:
import tempfile
from pathlib import Path

work = Path(tempfile.mkdtemp())
(work / "sequences").mkdir()
(work / "recon").mkdir()

(work / "sequences" / "my_gre.py").write_text("""
from pypulseqpp.sequences import steady_state_dummies
from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d

from pulserver.design import Evaluation, FloatParam, IntParam, SequencePlugin, TimeParam
from pulserver.protocol import TEPreset, UIParam


class Gre(SequencePlugin):
    app = gre2d
    protocol = {
        UIParam.FLIP: FloatParam("flip_angle_deg", unit="deg", range_min=1, range_max=90),
        UIParam.TE: TimeParam(
            "te", range_min=1000, range_max=80000, presets={TEPreset.MINIMUM: None}
        ),
        UIParam.TR: TimeParam("tr", range_min=1000, range_max=5_000_000),
        UIParam.NY: IntParam("n_y", range_min=32, range_max=512, range_incr=2),
    }

    def evaluate(self, system, protocol):
        arguments = protocol.arguments
        n_y = arguments["n_y"]
        one_line = arguments | {"ry": n_y, "n_acs_y": 0, "n_dummy": 0}
        seq = self.app(system, **one_line)
        te = seq.definitions["TE"][0]
        tr = seq.definitions["TR"][0]
        dummies = steady_state_dummies(tr, arguments["flip_angle_deg"])
        return Evaluation(
            protocol.replace({UIParam.TE: te, UIParam.TR: tr}),
            duration=(dummies + n_y) * tr,
        )
""")

(work / "recon" / "my_pics.py").write_text("""
import torch
import bartorch.tools as bt

from bartorch import apps, priors
from pulserver import recon
from pulserver.mrd import AcquisitionFlag


class Pics(recon.ReconPlugin):
    def __init__(self):
        super().__init__(
            gadgets=[recon.RemoveReadoutOversampling()],
            triggers={"imaging": AcquisitionFlag.LAST_IN_SLICE},
        )

    def recon(self, context, branch, data):
        kspace = torch.from_numpy(data.data.kspace)[:, None]  # (coils, z, y, x)
        maps = bt.ecalib(kspace, maps=1)
        image = apps.pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005))
        return recon.ReconResult(image.abs().squeeze(0).numpy())


PLUGIN = Pics()
""")
print(sorted(str(path.relative_to(work)) for path in work.rglob("*.py")))

# %%
# The virtual scanner, with your plugins
# --------------------------------------
#
# The console of lesson 1 took no plugin folders and used the shipped
# plugins. Point it at your two folders, and ``my_gre`` and ``my_pics`` are
# what it offers. When the operator opens ``my_gre``, the scanner shows your
# four buttons, plus the number of averages pulserver adds:
from pulserver import virtual
from pulserver.protocol import (
    TEPreset,
    UIParam,
    format_values,
    parse_listing,
    parse_validation,
)

system = """[Limits]
B0: 3.0
max_grad: 40
grad_unit: mT/m
max_slew: 150
slew_unit: T/m/s
[Limits End]
"""
console = virtual.Console(
    plugins=[work / "sequences"],
    recon_plugins=[work / "recon"],
    limits=system,
    store=work / "designs",
)
listing = console.design("list", "my_gre")["reply"]
block = listing[listing.index("[Protocol]") :]
print("\n".join(line for line in block.splitlines() if "|off|" not in line))
entries = parse_listing(block)

# %%
# The phantom is the vials of lesson 1: seven vials of water whose T1 and T2
# grow around the circle, and one of fat in the middle. Your plugin
# estimates coil maps, which needs more than one receive coil, so the exam
# puts the phantom in the body coil to transmit and a 48-channel head coil to
# receive.
localizer = console.exam("vials", coil="body/head48")  # DICOM files

# %%
# A flip-angle series
# -------------------
#
# The operator sets a TR of 50 ms and the minimum TE, and scans four times
# with flip angles from 5° to 60°. Each scan is the flow of lesson 1: the
# edit is validated, the sequence generated, and the scan played and
# reconstructed by ``my_pics``.
import base64
import io

import numpy as np
import pydicom

FLIPS = (5, 15, 30, 60)


def scan(flip):
    """Validate, generate and scan my_gre at a flip angle; return the image and TE in s."""
    values = {key: entry.value for key, entry in entries.items()}
    values.update(
        {
            UIParam.FLIP: flip,
            UIParam.TR: 50_000,  # µs on the wire
            UIParam.TE: TEPreset.MINIMUM.value,
        }
    )
    request = format_values(values, entries)
    validation = parse_validation(
        console.design("validate", "my_gre", request)["reply"], entries
    )
    te_ms = validation.values[UIParam.TE] / 1e3
    print(f"{flip:2d}°: TE {te_ms:.2f} ms, scan time {validation.duration:.1f} s")

    design = console.design("generate", "my_gre", request)["design"]
    received = []
    console.scan(
        design,
        rotation=np.eye(3),
        centre_mm=(0.0, 0.0, 0.0),
        emit=received.append,
        recon="my_pics",
    )
    dicom = next(item["dicom"] for item in received if "dicom" in item)
    return pydicom.dcmread(io.BytesIO(base64.b64decode(dicom))), te_ms / 1e3


images = {flip: scan(flip) for flip in FLIPS}
te = images[FLIPS[0]][1]  # the minimum TE, the same for every flip angle

# sphinx_gallery_start_ignore
from figure_style import MUTED, PAGE_WIDTH, SERIES


def pixels(image):
    """The pixel values in the units the reconstruction returned them in."""
    slope = float(image.RescaleSlope)
    return image.pixel_array * slope + float(image.RescaleIntercept)


peak = max(pixels(image).max() for image, _ in images.values())
fig, axes = plt.subplots(1, len(FLIPS), figsize=(PAGE_WIDTH, 0.3 * PAGE_WIDTH))
for ax, (flip, (image, _)) in zip(axes, images.items(), strict=True):
    x0, y0 = (float(v) for v in image.ImagePositionPatient[:2])
    dy, dx = (float(v) for v in image.PixelSpacing)
    extent = (x0, x0 + image.Columns * dx, y0 + image.Rows * dy, y0)
    ax.imshow(pixels(image), cmap="gray", extent=extent, vmin=0, vmax=peak)
    ax.set_xlim(-75, 75)
    ax.set_ylim(75, -75)
    ax.set_title(f"{flip}°")
    ax.set_axis_off()
plt.show()
# sphinx_gallery_end_ignore

# %%
# The four images are drawn on one grey scale. At 5° all the water vials are
# dim and alike; as the flip angle grows, the vials of short T1 brighten and
# those of long T1 fade.
#
# Against the physics
# -------------------
#
# After its dummy TRs, an RF-spoiled gradient echo reaches a steady state
# whose signal is known in closed form:
#
# .. math::
#
#    S(\alpha) = M_0 \, \sin\alpha \,
#    \frac{1 - E_1}{1 - \cos\alpha \, E_1} \, e^{-\mathrm{TE}/T_2},
#    \qquad E_1 = e^{-\mathrm{TR}/T_1}.
#
# Each vial's T1 and T2 are known, so the curves below have no free parameter
# but one overall scale, the same for every vial and every scan, fitted by
# least squares. The points are the mean of each vial in your images.
TR = 0.05
t1 = np.geomspace(0.3, 2.0, 7)  # the vials' T1 and T2, in s
t2 = np.geomspace(0.04, 0.3, 7)


def steady_state(flip_deg, t1, t2):
    alpha = np.radians(flip_deg)
    e1 = np.exp(-TR / t1)
    return np.sin(alpha) * (1 - e1) / (1 - np.cos(alpha) * e1) * np.exp(-te / t2)


# sphinx_gallery_start_ignore
angles = 2 * np.pi * np.arange(7) / 7  # the vials sit 45 mm from the centre
centres_mm = 45 * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def vial_means(image):
    """The mean of each water vial, within 9 mm of its centre."""
    x0, y0 = (float(v) for v in image.ImagePositionPatient[:2])
    dy, dx = (float(v) for v in image.PixelSpacing)
    xx, yy = np.meshgrid(
        x0 + dx * np.arange(image.Columns), y0 + dy * np.arange(image.Rows)
    )
    values = pixels(image)
    return [values[np.hypot(xx - cx, yy - cy) < 9].mean() for cx, cy in centres_mm]


measured = np.array([vial_means(images[flip][0]) for flip in FLIPS])
model = np.array([steady_state(flip, t1, t2) for flip in FLIPS])
scale = (measured * model).sum() / (model**2).sum()

curve = np.linspace(1, 70, 200)
_, ax = plt.subplots(figsize=(PAGE_WIDTH, 0.5 * PAGE_WIDTH))
for vial in range(7):
    colour = SERIES[vial]
    ax.plot(
        curve,
        scale * steady_state(curve, t1[vial], t2[vial]),
        color=colour,
        lw=1.2,
        label=f"T1 {t1[vial] * 1e3:.0f} ms",
    )
    ax.plot(FLIPS, measured[:, vial], "o", color=colour, ms=5)
ax.set_xlabel("flip angle (°)")
ax.set_ylabel("vial signal (data units)")
ax.set_xlim(0, 70)
ax.set_ylim(0, None)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("lines: closed form    points: your images", color=MUTED)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Each vial peaks at its own Ernst angle, and the measured points follow the
# closed form for all seven vials with one shared scale. What remains,
# largest for the vials of long T2, is most likely the spoiling: a 117° RF
# phase increment does not remove the transverse magnetization completely,
# and the closed form assumes it does.
#
# That agreement tests the whole chain at once: the flip angle and the timing
# your sequence plugin designed, the conversion to the scanner's cache, the
# labels that sorted the raw data, and your reconstruction. A wrong unit in a
# ``TimeParam``, or a recon that scaled each scan differently, would break it.
#
# The virtual scanner is a model, so passing this test does not make the
# sequence safe to run on a person; it tells you that the pair does what you
# designed before a scanner is involved. :doc:`/explanations/virtual-scanner`
# describes what it models.
#
# On a scanner
# ------------
#
# The same two folders are what pulserver is pointed at on a scanner: the
# design service takes the folder of sequence plugins and the reconstruction
# proxy the folder of recon plugins. Nothing in the files changes.
# :doc:`/user-guide/virtual-scanner` runs the same console from the command
# line, with MaRGE's interface in front of it.
#
# This is the end of the course. The Tours take each piece further.
#
# As a spec
# ---------
#
# What this lesson did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Save the gre2d sequence plugin of lesson 2 as sequences/my_gre.py and the
#    pics recon plugin of lesson 5 as recon/my_pics.py. Load both folders into
#    a pulserver virtual.Console, start a vials exam in the body/head48 coil,
#    and scan my_gre with recon my_pics at a TR of 50 ms, the minimum TE and
#    flip angles of 5, 15, 30 and 60 degrees. Compare the mean of each water
#    vial with the closed-form spoiled gradient echo signal for its T1 and T2,
#    with one least-squares scale for all vials and scans.
