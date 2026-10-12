"""
==================
1. Your first scan
==================

In this lesson you run a whole scan without writing any plugin. You use a
sequence and a reconstruction that ship with pulserver (``gre2d``, a 2D
gradient echo, and ``pics``, a compressed-sensing reconstruction with BART),
and the virtual scanner plays the sequence on a phantom. You will do what a
scanner does: ask for the protocol, edit it, start the scan and get the image
back.

Every step you run here is one step of the README's *How it works*. The rest
of the course opens them one at a time, and replaces the shipped plugins with
your own.

**Learning objectives**

- Run the six steps of a scan in one notebook.
- See what the operator sees: the protocol, and what pulserver sends back
  after an edit.
- Know which lesson of the course covers each step.

Next: :doc:`02_sequence_plugin`, where you write the sequence plugin yourself.
"""

# sphinx_gallery_start_ignore
import logging

import matplotlib.pyplot as plt

# The vials exam has no noise scan; the reconstruction's note about it is not
# part of the lesson.
logging.disable(logging.WARNING)
# sphinx_gallery_end_ignore
# %%
# A scanner in your notebook
# --------------------------
#
# A :class:`~pulserver.virtual.Console` stands in for the scanner: it makes
# the same calls a scanner makes, plays the sequence on a phantom, and
# reconstructs in this process instead of in a separate reconstruction server.
#
# It needs two things. The **system** is the scanner's hardware: field
# strength, maximum gradient amplitude and slew rate. These are the arguments
# of Pulseq's ``system`` object (:class:`pypulseqpp.Opts`), written as the text
# block a scanner sends; raster times and dead times you leave out take their
# defaults. The **store** is the folder where finished sequences are kept.
import tempfile
from pathlib import Path

import numpy as np

from pulserver import virtual

system = """[Limits]
B0: 3.0
max_grad: 40
grad_unit: mT/m
max_slew: 150
slew_unit: T/m/s
[Limits End]
"""
store = Path(tempfile.mkdtemp())
console = virtual.Console(plugins=[], limits=system, store=store, recon_plugins=[])

# %%
# Step 1: the protocol
# --------------------
#
# When the operator opens a sequence, the scanner asks pulserver which
# parameters to show. This is the reply for ``gre2d``: one line per parameter,
# with its type, current value, range and unit. The scanner builds its UI from
# it.
from pulserver.protocol import UIParam, format_values, parse_listing, parse_validation

listing = console.design("list", "gre2d")["reply"]
block = listing[listing.index("[Protocol]") :]
print("\n".join(block.splitlines()[:12]) + "\n...")

# %%
# Step 2: an edit
# ---------------
#
# The operator changes the matrix to 64 by 64. The scanner sends the new values
# back, and pulserver asks the sequence what it can achieve with them. The
# reply says whether the protocol is valid, the scan time, and any value the
# sequence had to change: here the receiver bandwidth.
entries = parse_listing(block)
values = {key: entry.value for key, entry in entries.items()}
values.update({UIParam.NX: 64, UIParam.NY: 64})
request = format_values(values, entries)

validation = parse_validation(
    console.design("validate", "gre2d", request)["reply"], entries
)
print(f"valid: {validation.valid}, scan time {validation.duration:.1f} s")
for key, value in validation.values.items():
    if value != values[key]:
        print(f"{key}: requested {values[key]}, achieved {value}")

# %%
# Step 3: the sequence for the scanner
# ------------------------------------
#
# The operator presses *Scan*. pulserver designs the whole sequence, checks it
# against the limits, converts it into the cache the scanner plays and keeps
# both in the store under one identifier.
generated = console.design("generate", "gre2d", request)
design = generated["design"]
print(generated["reply"])
print(sorted(path.name for path in (store / design).iterdir()))

# %%
# Steps 4 to 6: scan and reconstruct
# ----------------------------------
#
# An exam puts a subject in the scanner. Here it is a phantom of eight vials,
# seven of water with different T1 and T2 and one of fat, in the body coil.
# The exam returns a localizer, a quick image of the phantom, as the scanner
# would.
#
# The scan then plays the stored sequence on the phantom, pulserver labels the
# raw data from the sequence, and ``pics`` reconstructs it. The images come
# back as DICOM, as they would on the console.
import base64
import io

import pydicom

localizer = [pydicom.dcmread(io.BytesIO(f)) for f in console.exam("vials", coil="body")]

received = []
status = console.scan(
    design,
    rotation=np.eye(3),
    centre_mm=(0.0, 0.0, 0.0),
    emit=received.append,
    recon="pics",
)
images = [
    pydicom.dcmread(io.BytesIO(base64.b64decode(item["dicom"])))
    for item in received
    if "dicom" in item
]
print(f"status {status}, {len(images)} image of {images[0].pixel_array.shape}")

# sphinx_gallery_start_ignore
from figure_style import MUTED, PAGE_WIDTH, SERIES

fig, axes = plt.subplots(1, 2, figsize=(0.8 * PAGE_WIDTH, 0.42 * PAGE_WIDTH))
for ax, image, title in zip(
    axes,
    (localizer[0], images[0]),
    ("the phantom (localizer)", "your scan: gre2d + pics"),
    strict=True,
):
    # Both drawn in millimetres from the image position, so the two fields of
    # view line up.
    x0, y0 = (float(v) for v in image.ImagePositionPatient[:2])
    dy, dx = (float(v) for v in image.PixelSpacing)
    extent = (x0, x0 + image.Columns * dx, y0 + image.Rows * dy, y0)
    ax.imshow(image.pixel_array, cmap="gray", extent=extent)
    ax.set_xlim(-128, 128)
    ax.set_ylim(128, -128)
    ax.set_title(title)
    ax.set_axis_off()
plt.show()
# sphinx_gallery_end_ignore

# %%
# The localizer is drawn from the phantom's geometry. Your image is
# reconstructed from raw data the virtual scanner simulated, at the 64 x 64
# matrix you set. Both are drawn on the same millimetre scale: your image
# covers the protocol's 220 mm field of view, the localizer a wider one.
#
# What you just ran
# -----------------
#
# Each call above was one step of a scan. The rest of the course opens them in
# the same order:

# sphinx_gallery_start_ignore
from matplotlib.patches import FancyBboxPatch

steps = [
    ("1. protocol", "list", "lesson 2"),
    ("2. edit", "validate", "lesson 2"),
    ("3. checks +\nconversion", "generate", "lesson 3"),
    ("4. scanner\nplays", "scan", "lesson 6"),
    ("5. raw data\nlabelled", "scan", "lesson 4"),
    ("6. recon", "scan", "lesson 5"),
]
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 1.9))
ax.set_xlim(0, 6 * 17)
ax.set_ylim(0, 22)
ax.set_axis_off()
for i, (title, call, lesson) in enumerate(steps):
    x = i * 17 + 1
    colour = SERIES[0] if i < 3 else SERIES[2]
    ax.add_patch(
        FancyBboxPatch(
            (x, 6),
            14,
            13,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            fc=colour + "1f",
            ec=colour,
            lw=1.2,
        )
    )
    ax.text(x + 7, 15.5, title, ha="center", va="top", fontsize=10, color=colour)
    ax.text(x + 7, 8, call, ha="center", fontsize=8.5, color=MUTED, family="monospace")
    ax.text(x + 7, 1.5, lesson, ha="center", fontsize=9.5, color=MUTED, style="italic")
    if i < 5:
        ax.annotate(
            "",
            (x + 16.3, 12.5),
            (x + 14.7, 12.5),
            arrowprops={"arrowstyle": "-|>", "color": MUTED},
        )
plt.show()
# sphinx_gallery_end_ignore

# %%
# - Lesson 2 replaces the shipped ``gre2d`` plugin with your own: steps 1 and 2.
# - Lesson 3 looks inside what ``generate`` produced: step 3.
# - Lessons 4 and 5 follow the raw data into a reconstruction plugin you write:
#   steps 5 and 6.
# - Lesson 6 puts your two plugins together on the virtual scanner: step 4.
#
# As a spec
# ---------
#
# What this lesson did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    On pulserver's virtual console at 3 T (40 mT/m, 150 T/m/s), list the
#    shipped gre2d protocol, set a 64 x 64 matrix, generate the design, and
#    scan it on the vials phantom in the body coil with the pics
#    reconstruction. Show the localizer next to the image.
