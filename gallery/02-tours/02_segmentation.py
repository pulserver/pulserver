"""
==========================================
Segmentation of a sequence for the scanner
==========================================

This Tour converts shipped pypulseqpp sequences into the scanner
representation and reads what the conversion reduces them to: the repetition
of each subsequence, the virtual segments it is cut into, and how those counts
scale with the length of the scan.

**Prerequisites:** lessons 1 and 3 of the :doc:`course </examples/course>`.

A scanner interpreter prepares each virtual segment once and plays the scan as
an execution stream of segment instances, so the number of virtual segments,
not the number of blocks, sets what it prepares. The model is described in
:doc:`/explanations/scanner-representation`.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
# sphinx_gallery_end_ignore
# %%
# Repetition of a spin echo
# -------------------------
#
# The sequence is pypulseqpp's shipped 2D spin echo at its shortest TR,
# written to a file and converted under the scanner limits. :func:`~pulserver.ir.convert` writes the
# IR cache beside the sequence file; :func:`~pulserver.ir.summary` reports the
# segmentation.
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pypulseqpp as pp
from figure_style import FAINT, INK, PAGE_WIDTH, SERIES
from pypulseqpp.sequences import se2D_sequence

from pulserver import ir

system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
work = Path(tempfile.mkdtemp())

seq = se2D_sequence(n_x=64, n_y=64, tr=None)
seq.write(work / "se2d.seq")
cache = ir.convert(work / "se2d.seq", system)
report = ir.summary(work / "se2d.seq", system)

unit = report["subsequences"][0]
print(f"{seq.num_blocks} blocks in the file; cache {cache.name}")
print(
    f"repetition: {unit['tr_size']} blocks, TR {unit['tr_duration_us'] / 1e3:.2f} ms, "
    f"played {unit['num_trs']} times"
)
for index, segment in enumerate(report["segments"]):
    first = segment["start_block"] + 1
    last = segment["start_block"] + segment["num_blocks"]
    kind = "pure delay" if segment["pure_delay"] else "events"
    print(
        f"virtual segment {index}: blocks {first}-{last}, "
        f"{segment['duration_us'] / 1e3:.2f} ms, {kind}"
    )

# %%
# The repetition is cut only at block boundaries where every gradient is at
# rest, so blocks joined by a gradient that does not rest stay in one virtual
# segment. A pure delay has no events: every pure delay of the repetition
# plays the one pure-delay virtual segment, with its own duration, as the
# virtual segment of each block, read from the execution stream, shows.
owner = ir.play(work / "se2d.seq")["segment"][: unit["tr_size"]]
print("virtual segment of each block of the repetition:", owner.tolist())

# sphinx_gallery_start_ignore
tr_blocks = unit["tr_size"]
durations = np.array([seq.block_durations[i] for i in range(1, tr_blocks + 1)])
edges = np.concatenate([[0.0], np.cumsum(durations)]) * 1e3
waves = seq.waveforms_and_times(append_RF=True, block_range=(1, tr_blocks))[0]
gamma = 42.576e6


fig, axes = plt.subplots(4, 1, figsize=(PAGE_WIDTH, 5.2), sharex=True)
rows = (
    (waves[3], "RF (Hz)", True),
    (waves[2], "Gz (mT/m)", False),
    (waves[1], "Gy (mT/m)", False),
    (waves[0], "Gx (mT/m)", False),
)
for ax, (data, label, is_rf) in zip(axes, rows, strict=True):
    for block in range(tr_blocks):
        ax.axvspan(
            edges[block], edges[block + 1], color=SERIES[owner[block]], alpha=0.12, lw=0
        )
    time = np.real(data[0]) * 1e3
    value = np.abs(data[1]) if is_rf else np.real(data[1]) / gamma * 1e3
    ax.plot(time, value, lw=1, color=INK)
    ax.set_ylabel(label)
    for edge in edges:
        ax.axvline(edge, color=FAINT, lw=0.6)
for index in range(len(report["segments"])):
    axes[0].plot([], [], lw=6, alpha=0.3, color=SERIES[index], label=f"segment {index}")
axes[-1].set_xlabel("time from the start of the TR (ms)")
axes[-1].set_xlim(0, edges[-1])
fig.legend(loc="outside upper center", ncols=len(report["segments"]))
plt.show()
# sphinx_gallery_end_ignore

# %%
# Scan length
# -----------
#
# The same conversion over the shipped 2D gradient echo, with a growing
# phase-encoding matrix and slice count. The number of blocks and the number
# of repetitions grow with the scan; the number of virtual segments does not,
# because every repetition plays the same base blocks with a different
# phase-encoding amplitude.

from pypulseqpp.sequences import gre2D_sequence

rows = []
for n_y, n_slices in ((64, 1), (128, 1), (256, 1), (256, 4)):
    path = work / f"gre_{n_y}_{n_slices}.seq"
    gre = gre2D_sequence(n_x=64, n_y=n_y, n_slices=n_slices, tr=None)
    gre.write(path)
    result = ir.summary(path, system)
    rows.append((n_y, n_slices, gre.num_blocks, result))

# sphinx_gallery_start_ignore
print(
    f"{'ny':>5} {'slices':>7} {'blocks':>8} {'repetitions':>12} {'virtual segments':>17}"
)
for n_y, n_slices, blocks, result in rows:
    trs = sum(sub["num_trs"] for sub in result["subsequences"])
    print(f"{n_y:5d} {n_slices:7d} {blocks:8d} {trs:12d} {result['num_segments']:17d}")
# sphinx_gallery_end_ignore

# %%
# Subsequences
# ------------
#
# pypulseqpp's echo planar sequence returns a reference prescan, one volume
# with the phase-encoding direction reversed, and the imaging sequence, as a
# chain. :func:`pypulseqpp.sequences.write` writes them as separate files
# linked by the ``NextSequence`` definition, and the conversion reads the chain
# as the subsequences of one scan.

from pypulseqpp import sequences
from pypulseqpp.sequences.sequence.epi2D_sequence import epi2d

files = sequences.write(work / "epi.seq", epi2d(system, n_x=64, n_y=64), offline=False)
print([Path(f).name for f in ir.chain(files[0])])

report = ir.summary(files[0], system)
for index, subsequence in enumerate(report["subsequences"]):
    print(
        f"subsequence {index}: {subsequence['tr_size']} blocks per repetition, "
        f"{subsequence['num_trs']} repetitions"
    )
print(f"virtual segments over the chain: {report['num_segments']}")

# %%
# Virtual segments are deduplicated across subsequences as well as within
# one. The reference prescan reverses the sign of the phase-encoding
# gradients, which is an amplitude of each segment instance rather than part
# of a base block, so both subsequences play the virtual segments printed
# above.
