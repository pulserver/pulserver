"""
==========================================
Protocol resolution for a 2D gradient echo
==========================================

In this Tour you resolve prescriptions of a 2D gradient echo the way a design
call does for the scanner UI. You see two things the resolved protocol depends
on: the receiver bandwidth the readout realizes on the sampling rasters, and
the shortest echo time the readout admits.

**Prerequisites:** lessons 1 and 2 of the :doc:`course </examples/course>`.

pulserver resolves a request by evaluating the protocol under the scanner
limits and reading back the values the design achieved
(:doc:`/explanations/protocol`). The sequence is pypulseqpp's ``gre2d``, bound
to five protocol entries. To keep the plugin short, its ``evaluate`` designs
the whole scan. The shipped plugins, and lesson 2, design one repetition, so
the time an edit takes does not grow with the matrix.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sphinx_gallery_end_ignore
# %%
# The plugin
# ----------
#
# The plugin maps the interpreter's parameter names to keyword arguments of
# ``gre2d``. Times are exchanged in integer microseconds and the field of view
# in mm. The *Minimum* preset of TE and TR asks for the shortest time the
# design admits.
#
# ``evaluate`` designs the sequence and reads back three values: the echo time
# and the repetition time, from the sequence's ``TE`` and ``TR`` definitions,
# and the receiver bandwidth, the inverse of the dwell time of its ADC event.
# The reply carries these values and the scan time.
import numpy as np
import pypulseqpp as pp
from figure_style import MUTED, PAGE_WIDTH
from pypulseqpp import sequences
from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d

from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    TimeParam,
)
from pulserver.protocol import TEPreset, TRPreset, UIParam, format_listing


class Gre2D(SequencePlugin):
    app = gre2d
    protocol = {
        UIParam.TE: TimeParam(
            "te", range_min=1000, range_max=80000, presets={TEPreset.MINIMUM: None}
        ),
        UIParam.TR: TimeParam(
            "tr", range_min=1000, range_max=5_000_000, presets={TRPreset.MINIMUM: None}
        ),
        UIParam.BANDWIDTH: FloatParam(
            "readout_bandwidth_hz", unit="Hz", range_min=1e3, range_max=1e6
        ),
        UIParam.FOV: FloatParam(
            "fov_x", unit="mm", scale=1e-3, range_min=50.0, range_max=500.0
        ),
        UIParam.NX: IntParam("n_x", range_min=32, range_max=512, range_incr=2),
    }

    def evaluate(self, system, protocol):
        seq = self.app(system, **protocol.arguments)
        dwell = seq.libraries().adc[0, 1]
        achieved = {
            UIParam.TE: seq.definitions["TE"][0],
            UIParam.TR: seq.definitions["TR"][0],
            UIParam.BANDWIDTH: 1.0 / dwell,
        }
        return Evaluation(protocol.replace(achieved), sequences.duration(seq))


gre = Gre2D()
print(format_listing(gre.listing()), end="")

# %%
# These are the scanner limits the interpreter's host process sends with each
# call. You state the rasters explicitly, because the achieved bandwidth
# depends on them.

system = pp.Opts(
    max_grad=40,
    grad_unit="mT/m",
    max_slew=150,
    slew_unit="T/m/s",
    grad_raster_time=10e-6,
    adc_raster_time=100e-9,
)

# %%
# Receiver bandwidth
# ------------------
#
# The readout samples at a dwell time that is a multiple of the ADC raster.
# Its duration, the number of samples times the dwell time, must be a whole
# number of gradient raster periods. The dwell time is the first one that
# satisfies both, searched upward from the requested one. The achieved
# bandwidth is its reciprocal, so a request between two achievable bandwidths
# resolves to the lower one.
#
# With twofold readout oversampling, a 64-column matrix is read with 128
# samples and a 100-column matrix with 200. For 128 samples, a duration that is
# a multiple of 10 µs needs a dwell time that is a multiple of 2.5 µs, so only
# 400, 200, 133, 100, 80 kHz and so on are achievable. For 200 samples, every
# multiple of the 100 ns ADC raster is achievable.

requested = np.linspace(60e3, 300e3, 121)
achieved = {
    nx: [
        gre.validate(system, {"nx": nx, "bandwidth": bw}).values["bandwidth"]
        for bw in requested
    ]
    for nx in (64, 100)
}

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 3.6))
for nx, values in achieved.items():
    ax.step(
        requested / 1e3,
        np.asarray(values) / 1e3,
        where="post",
        label=f"{2 * nx} samples (nx = {nx})",
    )
ax.plot(requested / 1e3, requested / 1e3, ls=":", lw=1, color=MUTED)
ax.set_xlabel("requested bandwidth (kHz)")
ax.set_ylabel("achieved bandwidth (kHz)")
fig.legend(loc="outside upper center", ncols=2)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The reply always carries the achieved value, so the scanner UI shows the
# bandwidth that is played, not the one typed in.

for bw in (150e3, 190e3):
    reply = gre.validate(system, {"nx": 64, "bandwidth": bw})
    print(
        f"requested {bw / 1e3:6.1f} kHz -> achieved {reply.values['bandwidth'] / 1e3:6.1f} kHz"
    )

# %%
# Shortest echo time
# ------------------
#
# The *Minimum* preset of TE resolves to the echo time of the design, in
# integer microseconds. Before the echo come the excitation, the
# slice-selection rephaser, the readout prewinder and the first half of the
# readout's samples. The shortest echo time therefore grows with the number of
# samples and with the dwell time: with the matrix size, and inversely with
# the bandwidth.

matrices = np.arange(64, 385, 32)
bandwidths = (100e3, 200e3, 400e3)
minimum_te = {
    bw: [
        gre.validate(
            system, {"TE": TEPreset.MINIMUM, "nx": int(nx), "bandwidth": bw}
        ).values["TE"]
        for nx in matrices
    ]
    for bw in bandwidths
}

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 3.6))
for bw, values in minimum_te.items():
    ax.plot(
        matrices,
        np.asarray(values) / 1e3,
        marker="o",
        ms=3,
        label=f"{bw / 1e3:.0f} kHz",
    )
ax.set_xlabel("readout matrix size nx")
ax.set_ylabel("minimum TE (ms)")
fig.legend(loc="outside upper center", ncols=3, title="requested bandwidth")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Infeasible requests and repeated resolution
# -------------------------------------------
#
# A prescription the design cannot realize is invalid. The reply carries the
# request unchanged and the message of the error the design raised. The
# interpreter shows that message to the operator.

reply = gre.validate(system, {"TE": 1500, "nx": 256})
print(reply.valid)
print(reply.info)

# %%
# A valid reply carries the resolved protocol at the precision a scanner
# parameter stores. Sending it back resolves to the same protocol, so a design
# can be identified by its resolved protocol
# (:doc:`/explanations/architecture`).

first = gre.validate(system, {"TE": TEPreset.MINIMUM, "nx": 192, "bandwidth": 150e3})
again = gre.validate(system, first.values)
for key in gre.protocol:
    print(f"{key}: {first.values[key]}")
print(again.values == first.values, f"scan time {first.duration:.1f} s")
