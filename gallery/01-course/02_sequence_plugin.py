"""
======================================
2. A scanner sequence and its protocol
======================================

A scanner sequence binds a pypulseqpp sequence function to the entries of the
scanner protocol. The operator edits the entries; pulserver converts their
values into the function's arguments, evaluates the protocol under the scanner
limits, and returns the protocol the design achieves. The previous lesson used
the shipped ``gre2d``; this lesson writes one like it.

**Learning objectives**

- Declare a :class:`~pulserver.design.SequencePlugin`: the sequence function
  and the entries that bind its arguments, with their units and presets.
- Write an evaluation that designs one repetition of the scan and reads back
  the values the design achieves, the scan time and the RF of one TR.
- Validate requests, including a preset and a prescription the design
  refuses, and write the design.

The next lesson converts a design into the representation the scanner plays.
"""

# sphinx_gallery_start_ignore
import logging

import matplotlib

matplotlib.use("Agg")
# validate logs a refused request with its traceback; the page prints the reply.
logging.disable(logging.WARNING)
# sphinx_gallery_end_ignore
# %%
# Entries
# -------
#
# Each entry of :attr:`~pulserver.design.SequencePlugin.protocol` maps a
# protocol key, which names a parameter of the scanner UI, to a keyword
# argument of the sequence function. A :class:`~pulserver.design.TimeParam`
# is exchanged in integer microseconds and given to the function in seconds;
# its *Minimum* preset passes ``None``, for which ``gre2d`` designs its
# shortest time. A :class:`~pulserver.design.FloatParam` carries a unit and a
# scale, here mm on the UI and m for the function. Entries a protocol leaves
# out keep the function's defaults.
#
# :meth:`~pulserver.design.SequencePlugin.evaluate` answers every edit, so it
# designs one repetition, a single phase-encoding line without dummy scans,
# rather than the scan. The echo time and the repetition time are the ``TE``
# and ``TR`` definitions the design records; the scan time is the repetition
# time times the lines and dummy scans the scan plays. The RF layout is the RF
# of that one TR, its excitation's amplitude proportional to the flip angle.
import inspect

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d

from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    RfLayout,
    SequencePlugin,
    TimeParam,
)
from pulserver.protocol import TEPreset, TRPreset, UIParam, format_listing


class Gre2D(SequencePlugin):
    app = gre2d
    protocol = {
        UIParam.FLIP: FloatParam(
            "flip_angle_deg", unit="deg", range_min=1.0, range_max=90.0
        ),
        UIParam.TE: TimeParam(
            "te", range_min=1000, range_max=80000, presets={TEPreset.MINIMUM: None}
        ),
        UIParam.TR: TimeParam(
            "tr", range_min=1000, range_max=5_000_000, presets={TRPreset.MINIMUM: None}
        ),
        UIParam.FOV: FloatParam(
            "fov_x", unit="mm", scale=1e-3, range_min=50.0, range_max=500.0
        ),
        UIParam.NX: IntParam("n_x", range_min=32, range_max=512, range_incr=2),
        UIParam.NY: IntParam("n_y", range_min=32, range_max=512, range_incr=2),
    }

    def evaluate(self, system, protocol):
        bound = inspect.signature(self.app).bind_partial(system, **protocol.arguments)
        bound.apply_defaults()
        a = bound.arguments
        one = self.app(
            system,
            **(protocol.arguments | {"n_dummy": 0, "ry": a["n_y"], "n_acs_y": 0}),
        )
        tr = one.definitions["TR"][0]
        achieved = {UIParam.TE: one.definitions["TE"][0], UIParam.TR: tr}
        return Evaluation(
            protocol.replace(achieved),
            duration=tr * (a["n_dummy"] + a["n_y"]),
            rf_layout=RfLayout.of(one, UIParam.FLIP, period=tr),
        )


plugin = Gre2D()
print(format_listing(plugin.listing()), end="")

# %%
# Validation
# ----------
#
# A request is a mapping of protocol keys to wire values; the entries it
# leaves out take their initial values. Here the echo and repetition times
# request the *Minimum* preset, sent as its negative value, and the reply
# holds the times the design achieved.
system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")

shortest = plugin.validate(
    system,
    {
        UIParam.TE: TEPreset.MINIMUM.value,
        UIParam.TR: TRPreset.MINIMUM.value,
        UIParam.NY: 64,
    },
)
print(f"valid: {shortest.valid}, scan time {shortest.duration:.2f} s")
print(f"TE {shortest.values[UIParam.TE]} us, TR {shortest.values[UIParam.TR]} us")

# %%
# The RF layout states the definitions and the instances of one TR, from which
# a scanner estimates the RF of a prescription without designing it.
layout = shortest.rf_layout
print(
    f"{len(layout.instances.definitions)} RF definition, {len(layout.control)} instance per TR of {layout.period * 1e3:.2f} ms, controlled by {layout.control}"
)

# %%
# A prescription the sequence function cannot realize is invalid. The reply
# carries the request unchanged and, as its note, the error the design raised.
refused = plugin.validate(system, {UIParam.TE: 1000})
print(f"valid: {refused.valid}")
print(refused.info)

# %%
# Design
# ------
#
# :meth:`~pulserver.design.SequencePlugin.design` evaluates a request and
# writes the sequence of the requested protocol as signed binary Pulseq,
# designing the whole scan with :meth:`~pulserver.design.SequencePlugin.generate`.
# The design calls of the previous lesson do the same, then check and convert
# the design and store it.
import tempfile
from pathlib import Path

directory = Path(tempfile.mkdtemp())
validation, paths = plugin.design(system, shortest.values, directory)
print([Path(path).name for path in paths])
written = pp.Sequence()
written.read(paths[0])
print(f"{written.num_blocks} blocks")
