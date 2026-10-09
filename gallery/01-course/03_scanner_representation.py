"""
=========================
3. What the scanner plays
=========================

When the operator presses *Scan*, pulserver does three things before the
scanner plays a single block: it designs the whole scan with your sequence
function, checks it against the scanner's limits, and converts it into the
form the scanner plays. In lesson 1 all three were one call,
``console.design("generate", ...)``. In this lesson you run them one at a
time and look at what each one produces.

**Learning objectives**

- See which checks a design must pass, and what the operator reads when one
  fails.
- Fix a failing design by designing it under gentler limits.
- See how a scan of hundreds of blocks becomes a couple of segments the
  scanner prepares once, and what changes each time it plays them.

Previous: :doc:`02_sequence_plugin`. Next: :doc:`04_reconstruction_plugin`,
where you follow the raw data on its way back.
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_end_ignore
# %%
# The whole scan
# --------------
#
# The scan is your sequence function called with the final protocol. Here it
# is ``gre2d`` at a 64 by 64 matrix, the protocol you scanned in lesson 1,
# written to a Pulseq file as pulserver writes it.
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pypulseqpp as pp
from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d

system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")

seq = gre2d(system, n_x=64, n_y=64)
path = Path(tempfile.mkdtemp()) / "gre2d.seq"
seq.write(path)
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")

# %%
# The checks
# ----------
#
# A sequence designed under the scanner's limits can still break them, for
# example on a rotated slice, where two gradient axes add up on one physical
# axis. So pulserver checks the design as the scanner will play it, rotated to
# the prescription, with :func:`~pulserver.ir.check`:
#
# - the timing: every block on its raster, dead times respected;
# - the gradient amplitude and the slew rate against ``system``;
# - peripheral nerve stimulation (PNS), mechanical resonances and acoustic
#   noise, when the scanner states a model for them.
#
# The PNS model belongs to the gradient coil, and the scanner sends it with
# its limits. Here is an example rheobase-chronaxie model:
from pypulseqpp import safety

from pulserver import ir

coil = safety.ChronaxieModel(chronaxie=360e-6, rheobase=20, alpha=0.333)
limits = ir.CheckLimits(pns=coil)

problems = ir.check(path, system, limits=limits)
print("\n".join(problems) or "no problems")

# %%
# The design is refused. On a scanner this message is what the operator reads
# instead of a scan starting, and nothing is played.
# :doc:`/explanations/safety-checks` lists every check and what each one
# needs from the scanner.
#
# The figure shows why. Over the first 12 ms of a TR, the gradients stay
# within 40 mT/m and the slew rate within 150 T/m/s, but the slice rephaser
# ramps fast enough to drive the nerve response just above its threshold.

# sphinx_gallery_start_ignore
from figure_style import MUTED, PAGE_WIDTH, SERIES

GAMMA = 42.576e6


def pns_trace(sequence):
    report = safety.check_pns(sequence, coil, trace=True)[1]
    return np.asarray(report.time), 100 * np.asarray(report.response)


tr = seq.definitions["TR"][0]
start = tr * (seq.duration()[0] // tr - 1)
window = (start, start + 0.012)
waves = seq.waveforms_and_times(time_range=window)[0]
fig, axes = plt.subplots(3, 1, figsize=(PAGE_WIDTH, 0.72 * PAGE_WIDTH), sharex=True)
for axis, (time, amplitude) in zip("xyz", waves[:3], strict=True):
    colour = SERIES["xyz".index(axis)]
    ms = (time - start) * 1e3
    axes[0].plot(ms, amplitude / GAMMA * 1e3, color=colour, label=f"G{axis}")
    slew = np.abs(np.diff(amplitude) / np.diff(time)) / GAMMA
    axes[1].step(ms[:-1], slew, where="post", color=colour)
time, response = pns_trace(seq)
shown = (time >= window[0]) & (time <= window[1])
axes[2].plot((time[shown] - start) * 1e3, response[shown], color=SERIES[0])
for ax, limit, label in zip(
    axes,
    (40, 150, 100),
    ("gradient (mT/m)", "slew rate (T/m/s)", "PNS (% of threshold)"),
    strict=True,
):
    ax.axhline(limit, color=MUTED, ls="--", lw=1)
    ax.set_ylabel(label)
axes[0].axhline(-40, color=MUTED, ls="--", lw=1)
axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
axes[2].set_xlabel("time in the TR (ms)")
axes[2].set_xlim(0, 12)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Designing under gentler limits
# ------------------------------
#
# The remedy is a change of design, not of the check: design the sequence
# with a lower slew rate, and check it against the same scanner. A scanner can send
# such design limits with its own (``design_max_slew``), and pulserver then
# hands your sequence function the gentler ``system``.
gentle = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=100, slew_unit="T/m/s")

seq = gre2d(gentle, n_x=64, n_y=64)
seq.write(path)
problems = ir.check(path, system, limits=limits)
print("\n".join(problems) or "no problems")

# sphinx_gallery_start_ignore
time_gentle, response_gentle = pns_trace(seq)
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 0.3 * PAGE_WIDTH))
for t, r, colour, label in (
    (time, response, SERIES[0], "150 T/m/s: refused"),
    (time_gentle, response_gentle, SERIES[2], "100 T/m/s: accepted"),
):
    shown = (t >= window[0]) & (t <= window[1])
    ax.plot((t[shown] - start) * 1e3, r[shown], color=colour, label=label)
ax.axhline(100, color=MUTED, ls="--", lw=1)
ax.set_xlim(0, 12)
ax.set_xlabel("time in the TR (ms)")
ax.set_ylabel("PNS (%)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The same TE and TR, with the nerve response under its threshold.
#
# What the scanner prepares
# -------------------------
#
# The scanner does not read the Pulseq file block by block while it plays.
# pulserver first finds what repeats: the TR, here 6 blocks played once per
# line and once per dummy TR (64 lines and 25 dummies). It cuts the TR into **segments**, short runs of
# blocks the scanner prepares once, and lists the order to play them in. The
# result is the **cache** the scanner loads, written beside the Pulseq file by
# :func:`~pulserver.ir.convert`:
print(ir.convert(path, system).name)

report = ir.summary(path, system, cache_ext=".pseg")
repetition = report["subsequences"][0]
print(
    f"{seq.num_blocks} blocks = {repetition['num_trs']} TRs "
    f"of {repetition['tr_size']} blocks"
)
for number, segment in enumerate(report["segments"]):
    kind = "a pure delay" if segment["pure_delay"] else "RF, gradients and ADC"
    print(f"segment {number}: {segment['num_blocks']} block(s), {kind}")

# %%
# Segment 0 excites, encodes and reads out; segment 1 is the wait until the
# next TR. Every TR plays the two of them in turn, as the scanner will:

# sphinx_gallery_start_ignore
played = ir.play(path)
blocks = repetition["tr_size"]
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 1.3))
shown_trs = 4
for index in range(shown_trs * blocks):
    segment = int(played["segment"][index])
    ax.add_patch(
        plt.Rectangle(
            (index, 0), 0.92, 1, color=SERIES[0] if segment == 0 else SERIES[1]
        )
    )
for tr_index in range(shown_trs):
    ax.text(
        tr_index * blocks + blocks / 2,
        -0.35,
        f"TR {tr_index + 1}",
        ha="center",
        va="top",
        color=MUTED,
    )
ax.text(shown_trs * blocks + 0.5, 0.5, "...", va="center", color=MUTED)
ax.set_xlim(0, shown_trs * blocks + 2)
ax.set_ylim(-1, 1.2)
ax.set_axis_off()
ax.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color=SERIES[0], label="segment 0"),
        plt.Rectangle((0, 0), 1, 1, color=SERIES[1], label="segment 1 (delay)"),
    ],
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.6),
)
plt.show()
# sphinx_gallery_end_ignore

# %%
# What changes from one TR to the next
# ------------------------------------
#
# Every TR plays the same segments, and what differs between TRs, such as the
# phase encoding, is carried by the segment instances. The scanner prepares a segment's timing and waveform shapes once. Each time it
# plays the segment, the *instance* sets everything else:
#
# - the amplitude of each gradient, and its rotation;
# - another gradient waveform in place of the prepared one, provided its
#   timing is the same;
# - the RF amplitude, its phase and frequency offsets, and the shim of each
#   transmit channel;
# - the ADC phase and frequency offsets, or no acquisition at all;
# - whether it waits for a trigger, and the digital outputs.
#
# :func:`~pulserver.ir.play` walks the cache with the same C library the
# scanner links, one entry per block played, so you can read these values:
gy = played["gradient_hz_per_m"][:, 1].reshape(-1, blocks)
phase_encoding = np.abs(gy).max(axis=0).argmax()
rf_phase = played["rf_phase_rad"].reshape(-1, blocks)[:, 0]
acquires = played["adc"].reshape(-1, blocks).any(axis=1)

# sphinx_gallery_start_ignore
fig, (top, middle, bottom) = plt.subplots(
    3, 1, figsize=(PAGE_WIDTH, 0.6 * PAGE_WIDTH), sharex=True
)
top.plot(gy[:, phase_encoding] / GAMMA * 1e3, ".", ms=4, color=SERIES[0])
top.set_ylabel("phase encoding\n(mT/m)")
middle.plot(np.degrees(rf_phase) % 360, ".", ms=4, color=SERIES[2])
middle.set_ylabel("RF phase\n(deg)")
bottom.step(np.arange(len(acquires)), acquires, where="mid", color=SERIES[1])
bottom.set_yticks([0, 1], ["off", "on"])
bottom.set_ylabel("ADC")
bottom.set_xlabel("TR")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The first TRs are the dummy TRs: the same segment, with no phase encoding
# and the ADC off. Then the phase-encoding amplitude steps through the 64
# lines. The RF phase follows the quadratic schedule of RF spoiling, its
# increment growing by 117° per TR. The scanner prepares segment 0 once and
# only changes these numbers, so its preparation does not grow with the
# length of the scan.
#
# Where the segment boundaries fall can be tuned to the scanner's interpreter;
# :doc:`/explanations/scanner-representation` explains the model.
#
# As a spec
# ---------
#
# What this lesson did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Design pypulseqpp's gre2d at 64 x 64 for a 40 mT/m, 150 T/m/s scanner and
#    check it with pulserver.ir.check against a chronaxie PNS model
#    (chronaxie 360 us, rheobase 20 T/m/s, alpha 0.333). If it is refused,
#    redesign it at 100 T/m/s and check again. Convert the accepted design
#    with pulserver.ir.convert, report its segments, and plot the
#    phase-encoding amplitude and RF phase of every TR from pulserver.ir.play.
