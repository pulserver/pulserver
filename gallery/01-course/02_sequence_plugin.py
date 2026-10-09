"""
==================================
2. Your sequence in the scanner UI
==================================

In lesson 1 the operator edited the protocol of ``gre2d``, and pulserver
answered every edit with the TE, the TR and the scan time. Both the protocol
and the answers came from a plugin: a short Python class wrapped around a
sequence function. In this lesson you write that plugin yourself, one piece at
a time, and after each piece you look at what changed for the operator.

**Learning objectives**

- Turn a sequence function into a plugin the scanner UI can show.
- Choose the parameters the operator edits, with their units and ranges.
- Write an ``evaluate`` that tells the operator what the protocol achieves, and
  refuses what your sequence cannot play.
- Expose one more parameter, and watch the minimum TE follow it.

Previous: :doc:`01_protocol_to_image`. Next: :doc:`03_scanner_representation`,
where you look at what pulserver makes of your sequence before the scanner
plays it.
"""

# sphinx_gallery_start_ignore
import logging

import matplotlib.pyplot as plt

# validate logs a refused request with its traceback; the page prints the reply.
logging.disable(logging.WARNING)
# sphinx_gallery_end_ignore
# %%
# Your sequence
# -------------
#
# A sequence is a Python function. It takes the scanner's limits and keyword
# arguments, and returns a Pulseq sequence. ``gre2d``, the one you scanned in
# lesson 1, is such a function from pypulseqpp; any function you write in the
# same shape works the same way.
#
# The limits are a :class:`pypulseqpp.Opts`, the ``system`` object of Pulseq
# and PyPulseq: field strength, gradient amplitude and slew rate, raster
# times. On a scanner pulserver builds it from what the scanner sends; here you
# build it yourself.
#
# Here is one TR of ``gre2d``, designed with the shortest TE and TR its readout
# allows (``te=None``, ``tr=None``):
import pypulseqpp as pp
from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d

system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")

seq = gre2d(system, n_y=8, n_dummy=0, te=None, tr=None)
seq.paper_plot()

# sphinx_gallery_start_ignore
from figure_style import MUTED, PAGE_WIDTH, SERIES

ax = plt.gca()
ax.figure.set_size_inches(PAGE_WIDTH, 0.45 * PAGE_WIDTH)
excitation = seq.get_block(1).rf
start = excitation.delay + excitation.center
te = seq.definitions["TE"][0]
ax.annotate(
    "",
    (start + te, 9.9),
    (start, 9.9),
    arrowprops={"arrowstyle": "<->", "color": SERIES[0]},
)
ax.text(start + te / 2, 10.05, "TE", ha="center", va="bottom", color=SERIES[0])
for t in (start, start + te):
    ax.axvline(t, color=SERIES[0], lw=0.6, ls=":")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Each TR excites the slice, encodes one phase-encoding line, and reads out
# one echo. The scan repeats it once per line (128 by default), after a few
# dummy TRs that bring the magnetization to its steady state.
#
# The smallest plugin
# -------------------
#
# A plugin is a class with two attributes. ``app`` is your sequence function.
# ``protocol`` says which of its arguments the operator can edit. It is a
# dictionary, and each item pairs two things pulserver gives you:
#
# - the **key** is a button of the scanner UI, a member of
#   :class:`~pulserver.protocol.UIParam` such as ``UIParam.FLIP`` or
#   ``UIParam.TE``. The set of buttons is fixed: they are the parameters the
#   scanner's interpreter knows how to show.
# - the **value** says how that button drives your function: which argument
#   it sets (``"flip_angle_deg"``), its unit and the range the operator may
#   type. It is one of the ``*Param`` classes of :mod:`pulserver.design`,
#   chosen by the kind of number: :class:`~pulserver.design.FloatParam`,
#   :class:`~pulserver.design.IntParam`, :class:`~pulserver.design.TimeParam`
#   and a few more.
#
# Start with one item, the flip angle:
from pulserver.design import FloatParam, IntParam, SequencePlugin, TimeParam
from pulserver.protocol import UIParam, format_listing


class Gre(SequencePlugin):
    app = gre2d
    protocol = {
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1, range_max=90
        ),
    }


def ui(plugin):
    """Print the parameters the operator sees, as the scanner receives them."""
    listing = format_listing(plugin.listing())
    print("\n".join(line for line in listing.splitlines() if "|off|" not in line))


ui(Gre())

# sphinx_gallery_start_ignore
from matplotlib.patches import FancyBboxPatch

boxes = [
    ("key: the button", "UIParam.FLIP", "flip, on the scanner UI", SERIES[0]),
    (
        "value: how it maps",
        'FloatParam(\n  "flip_angle_deg",\n  unit="deg", 1 to 90)',
        "checked against the range",
        SERIES[1],
    ),
    (
        "your function",
        "gre2d(system,\n  flip_angle_deg=...)",
        "called by pulserver",
        SERIES[2],
    ),
]
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 2.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 20)
ax.set_axis_off()
for i, (title, code, note, colour) in enumerate(boxes):
    x = 1 + 34 * i
    ax.add_patch(
        FancyBboxPatch(
            (x, 3),
            29,
            14,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            fc=colour + "1f",
            ec=colour,
            lw=1.2,
        )
    )
    ax.text(x + 14.5, 15, title, ha="center", va="top", color=colour)
    ax.text(
        x + 14.5, 9.5, code, ha="center", va="center", family="monospace", fontsize=9.5
    )
    ax.text(x + 14.5, 4.2, note, ha="center", color=MUTED, fontsize=9, style="italic")
    if i < 2:
        ax.annotate(
            "",
            (x + 33.5, 10),
            (x + 30.5, 10),
            arrowprops={"arrowstyle": "-|>", "color": MUTED},
        )
plt.show()
# sphinx_gallery_end_ignore

# %%
# This is what the scanner receives when the operator opens your sequence: one
# line per parameter with its type, current value, range, step and unit. The
# current value, 12 degrees, is the default of ``gre2d``'s
# ``flip_angle_deg``. ``nex``, the number of averages, is added by pulserver,
# which plays the sequence that many times. Every other argument of ``gre2d``
# keeps its default and is out of the operator's reach.
#
# Units and ranges
# ----------------
#
# The scanner works in its own units, your function in SI. Each entry
# converts between the two. A :class:`~pulserver.design.TimeParam` is in
# seconds in your function and in integer microseconds between pulserver and
# the scanner, the unit GE stores times in; the operator sees and types
# milliseconds, and the scanner's interpreter converts. So ``range_min=1000``
# below is 1 ms on the UI. A
# :class:`~pulserver.design.FloatParam` takes a ``unit`` and a ``scale``, for
# example millimetres on the scanner and metres in your function
# (``scale=1e-3``). The range is what the operator is allowed to type.
#
# Add the echo time, the repetition time and the number of phase-encoding
# lines:


class Gre(SequencePlugin):
    app = gre2d
    protocol = {
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1, range_max=90
        ),
        UIParam.TE: TimeParam("te", range_min=1000, range_max=80000),
        UIParam.TR: TimeParam("tr", range_min=1000, range_max=5_000_000),
        UIParam.NY: IntParam("n_y", range_min=32, range_max=512, range_incr=2),
    }


plugin = Gre()
ui(plugin)

# %%
# What the operator gets back
# ---------------------------
#
# Every time the operator changes a value, the scanner sends the protocol to
# pulserver, and pulserver calls your plugin's ``validate``. Here the operator
# types a TE of 2 ms, which reaches pulserver as 2000 µs:
validation = plugin.validate(system, {UIParam.TE: 2000})
print(f"valid: {validation.valid}, scan time: {validation.duration}")

# %%
# The protocol is accepted, with no scan time. Your plugin has not told
# pulserver anything about the sequence yet, so pulserver accepts every value
# inside the ranges. The problem shows up only when the operator presses
# *Scan*, and pulserver designs the whole sequence:
import tempfile
from pathlib import Path

try:
    plugin.design(system, validation.values, Path(tempfile.mkdtemp()))
except ValueError as error:
    print(error)

# %%
# ``gre2d`` cannot reach a 2 ms echo time with this readout. The operator
# finds out after having set up the whole exam. The next step makes them find
# out while typing.
#
# Your evaluate
# -------------
#
# ``evaluate`` is the method pulserver calls on every edit. It receives the
# scanner limits and the protocol the operator typed, and returns an
# :class:`~pulserver.design.Evaluation`: the protocol as your sequence will
# play it, and the scan time. If the protocol is impossible, raise an
# exception; its message is what the operator reads.
#
# It runs on every keystroke, so it has to be fast: a few tens of
# milliseconds. Designing the whole scan is too slow, but ``gre2d`` designs one
# TR quickly, so this ``evaluate`` designs a single phase-encoding line and
# reads the TE and TR it achieved. The scan time is one TR per line, plus the
# dummy TRs ``gre2d`` plays before them.
#
# ``protocol.arguments`` is what your function receives: one item per entry,
# in SI units, with presets resolved. Arguments without an entry, such as
# ``ry``, are not in it and keep ``gre2d``'s defaults. To design one line,
# ``evaluate`` overrides three of them with the ``|`` of two dictionaries,
# which also wins over an entry the operator edits: ``ry=n_y`` keeps one line
# in every ``n_y``, and the calibration lines and dummy TRs are dropped.
#
# A *preset* adds a choice to the parameter's menu. ``TEPreset.MINIMUM`` lets
# the operator ask for the shortest TE; your function receives ``None`` for
# it, which ``gre2d`` reads as "as short as possible". On the wire a preset is
# a negative code (``TEPreset.MINIMUM.value``, -2), which the scanner shows as
# *Minimum* in the TE menu.
from pypulseqpp.sequences import steady_state_dummies

from pulserver.design import Evaluation
from pulserver.protocol import TEPreset


class Gre(SequencePlugin):
    app = gre2d
    protocol = {
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1, range_max=90
        ),
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


plugin = Gre()

# %%
# The same three edits, now answered by your ``evaluate``:
for typed, value in (
    ("8 ms", 8000),
    ("2 ms", 2000),
    ("Minimum", TEPreset.MINIMUM.value),
):
    validation = plugin.validate(system, {UIParam.TE: value})
    if validation.valid:
        te = validation.values[UIParam.TE] / 1e3
        print(f"{typed}: valid, TE {te} ms, scan time {validation.duration:.1f} s")
    else:
        print(f"{typed}: refused, {validation.info}")

# %%
# The 2 ms echo time is refused while the operator types it, with the reason
# ``gre2d`` gave. *Minimum* comes back as the 3.4 ms the readout allows, and
# that is the value the operator sees.
#
# What ``evaluate`` reports is your plugin's responsibility: pulserver passes
# it on to the operator as it is. Designing one TR is the simplest way to be
# right. A sequence whose timing is easy to compute can return the TE and TR
# from block durations without designing anything, and one that cannot design
# a short piece of itself can accept the typed values and only report the scan
# time.
#
# Your turn: expose the receiver bandwidth
# ----------------------------------------
#
# The shortest echo time depends on how long the readout lasts, and that is
# set by the receiver bandwidth. ``gre2d`` takes it as
# ``readout_bandwidth_hz``, but the operator cannot change it yet. Exposing it
# is one more entry; ``evaluate`` stays as it is, because it already designs
# whatever the operator typed.


class GreBandwidth(Gre):
    protocol = Gre.protocol | {
        UIParam.BANDWIDTH: FloatParam(
            "readout_bandwidth_hz", unit="Hz", range_min=1e3, range_max=1e6
        ),
    }


plugin = GreBandwidth()
ui(plugin)

# %%
# Now ask for the *Minimum* TE at a range of bandwidths, as an operator
# trading signal-to-noise for echo time would:
import numpy as np

bandwidths = np.arange(20e3, 251e3, 2.5e3)
minimum_te = [
    plugin.validate(
        system, {UIParam.BANDWIDTH: bw, UIParam.TE: TEPreset.MINIMUM.value}
    ).values[UIParam.TE]
    / 1e3
    for bw in bandwidths
]

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(0.75 * PAGE_WIDTH, 0.42 * PAGE_WIDTH))
ax.fill_between(
    bandwidths / 1e3,
    0,
    minimum_te,
    step="post",
    color=MUTED,
    alpha=0.18,
    lw=0,
    label="refused",
)
ax.step(
    bandwidths / 1e3,
    minimum_te,
    where="post",
    color=SERIES[0],
    lw=1.8,
    label="Minimum TE",
)
ax.set_xlabel("receiver bandwidth typed (kHz)")
ax.set_ylabel("TE (ms)")
ax.set_xlim(bandwidths[0] / 1e3, bandwidths[-1] / 1e3)
ax.set_ylim(0, 10)
ax.text(140, 1.6, "TE refused", color=MUTED, ha="center")
ax.text(140, 6.5, "TE accepted", color=MUTED, ha="center")
ax.legend(frameon=False, loc="upper right")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Below the curve the operator's TE is refused, above it accepted, and the
# *Minimum* preset sits on it. Lowering the bandwidth lengthens the readout and
# pushes the minimum TE up.
#
# The curve is a staircase because ``gre2d`` does not play every bandwidth it
# is asked for. Here its readout uses dwell times that are whole multiples of
# 10 µs, so it plays 100, 50, 33, 25 or 20 kHz, and a typed bandwidth is
# replaced by one of them; nothing faster than 100 kHz is played. The
# shipped ``gre2d`` plugin reports the bandwidth it plays back to the
# operator, as you saw in lesson 1. Your ``evaluate`` could do the same by
# reading the dwell time of the ADC event in ``seq``.
#
# In a file
# ---------
#
# On a scanner, a plugin is a ``.py`` file in a folder pulserver is pointed
# at, and its file name is the name the operator picks. The class is all the
# file needs:
#
# .. code-block:: python
#
#    # sequences/my_gre.py
#    from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d
#
#    from pulserver.design import Evaluation, FloatParam, SequencePlugin, TimeParam
#    from pulserver.protocol import TEPreset, UIParam
#
#
#    class Gre(SequencePlugin):
#        app = gre2d
#        protocol = {...}
#
#        def evaluate(self, system, protocol):
#            ...
#
# Lesson 6 loads your file into the virtual scanner and scans with it.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Write a pulserver SequencePlugin for pypulseqpp's gre2d. Let the operator
#    edit the flip angle (1-90 deg), TE (1-80 ms, with a Minimum preset), TR
#    (1 ms-5 s), the phase-encoding lines (32-512, even) and the receiver
#    bandwidth (1 kHz-1 MHz). In evaluate, design a single phase-encoding
#    line, report the TE and TR it achieves, and estimate the scan time as one
#    TR per line plus the steady-state dummies. Refuse a TE the readout cannot
#    reach, with gre2d's own message.
