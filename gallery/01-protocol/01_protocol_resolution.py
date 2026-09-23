"""
==========================================
Protocol resolution for a 2D gradient echo
==========================================

The scope of this notebook is to resolve prescriptions of a two-dimensional
gradient echo the way the host daemon does for the scanner UI, and to show
what the resolved protocol depends on: the receiver bandwidth a readout can
realize on the sampling rasters, and the shortest echo time the readout
admits.

A request is resolved by designing the sequence under the scanner limits and
reading back the values the design achieved, as described in
:doc:`/explanations/protocol`. The sequence is pypulseqpp's shipped
``Gre2DApp``, bound to five protocol entries.

Outline:

#. **Scanner sequence.** The binding of the application to the protocol.
#. **Receiver bandwidth.** Requested and achieved bandwidth for two readout
   lengths.
#. **Shortest echo time.** The *Minimum* preset against matrix size and
   bandwidth.
#. **Infeasible requests and repeated resolution.** The reply to a
   prescription the design refuses, and to a resolved protocol sent back.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 8.6  # inches, the width of the documentation column
# sphinx_gallery_end_ignore

# %%
# Scanner sequence
# ----------------
#
# The binding maps interpreter parameter names to ``init_sequence`` arguments.
# Times are exchanged in integer microseconds, the field of view in mm, and the
# *Minimum* preset of TE and TR requests the shortest time the design admits.
# ``resolved`` reads the achieved echo time, repetition time and ADC sampling
# rate from the readout module, where the application keeps them.

import numpy as np
import pypulseqpp as pp
from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp

from pulserver.design import FloatParam, IntParam, ScannerSequence, TimeParam
from pulserver.protocol import TEPreset, TRPreset, UIParam, format_listing


class Gre2D(ScannerSequence):
    app = Gre2DApp
    ui = {
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

    def resolved(self, app):
        return {
            "te": app.ro.echo_time,
            "tr": app.repetition_time,
            "readout_bandwidth_hz": app.ro.bandwidth_hz,
        }


gre = Gre2D()
print(format_listing(gre.listing()), end="")

# %%
# The scanner limits are those a PSD host process opens its session with. The
# rasters are stated explicitly because the achieved bandwidth depends on
# them.

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
# The readout samples at a dwell time that is a multiple of the ADC raster,
# and its duration, the number of samples times the dwell time, must be a
# whole number of gradient raster periods. The dwell time is the first that
# satisfies both, searched upward from the requested one, and the achieved
# bandwidth is its reciprocal: a request between two achievable bandwidths
# resolves to the lower of them. With twofold readout
# oversampling, a 64-column matrix is read with 128 samples and a 100-column
# matrix with 200.
#
# For 128 samples, a duration that is a multiple of 10 µs requires a dwell
# time that is a multiple of 2.5 µs, so only 400, 200, 133, 100, 80 kHz and so
# on are achievable. For 200 samples, every multiple of the 100 ns ADC raster
# is achievable.

requested = np.linspace(60e3, 300e3, 121)
achieved = {
    nx: [
        gre.validate(system, {"nx": nx, "bandwidth": bw}).values["bandwidth"]
        for bw in requested
    ]
    for nx in (64, 100)
}

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 3.6), layout="constrained")
for nx, values in achieved.items():
    ax.step(
        requested / 1e3,
        np.asarray(values) / 1e3,
        where="post",
        label=f"{2 * nx} samples (nx = {nx})",
    )
ax.plot(requested / 1e3, requested / 1e3, ls=":", lw=1, color="#7d8996")
ax.set_xlabel("requested bandwidth (kHz)")
ax.set_ylabel("achieved bandwidth (kHz)")
fig.legend(loc="outside upper center", ncols=2, frameon=False)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The reply always carries the achieved value, so the scanner UI shows the
# bandwidth that will be played rather than the one typed in.

for bw in (150e3, 190e3):
    reply = gre.validate(system, {"nx": 64, "bandwidth": bw})
    print(
        f"requested {bw / 1e3:6.1f} kHz -> achieved {reply.values['bandwidth'] / 1e3:6.1f} kHz"
    )

# %%
# Shortest echo time
# ------------------
#
# The *Minimum* preset of TE is resolved to the echo time of the design, in
# integer microseconds. The echo is placed after the excitation, the
# slice-selection rephaser and the readout prewinder, and before it the
# readout acquires half of its samples. The shortest echo time therefore grows
# with the number of samples and with the dwell time, that is, with matrix size
# and inversely with bandwidth.

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
fig, ax = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 3.6), layout="constrained")
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
fig.legend(
    loc="outside upper center", ncols=3, frameon=False, title="requested bandwidth"
)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Infeasible requests and repeated resolution
# -------------------------------------------
#
# A prescription the design cannot realize is invalid. The reply carries the
# request unchanged and the error the application raised, which the
# interpreter shows to the operator.

reply = gre.validate(system, {"TE": 1500, "nx": 256})
print(reply.valid)
print(reply.info)

# %%
# A valid reply carries the resolved protocol at the precision a scanner
# control variable stores. Sending it back resolves to the same protocol,
# which is what lets the host daemon identify a revision by its resolved
# protocol (:doc:`/explanations/sessions`).

first = gre.validate(system, {"TE": TEPreset.MINIMUM, "nx": 192, "bandwidth": 150e3})
again = gre.validate(system, first.values)
print(first.values)
print(again.values == first.values, f"scan time {first.duration:.1f} s")
