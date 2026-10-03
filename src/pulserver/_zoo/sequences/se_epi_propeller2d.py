"""pypulseqpp's 2D PROPELLER spin echo with EPI blades bound to the scanner UI."""

import math

import pypulseqpp as pp
from pypulseqpp.sequences.sequence.se_epi_propeller2D_sequence import (
    se_epi_propeller2d,
)

from pulserver._zoo._evaluation import achieved, arguments, rf_layout
from pulserver.design import (
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    TimeParam,
)
from pulserver.protocol import TEPreset, TRPreset, UIParam


class SeEpiPropeller2D(SequencePlugin):
    app = se_epi_propeller2d
    protocol = {
        UIParam.TE: TimeParam(
            "te",
            range_min=1000,
            range_max=200000,
            presets={TEPreset.MINIMUM: None},
            default=TEPreset.MINIMUM,
        ),
        UIParam.TR: TimeParam(
            "tr",
            range_min=10000,
            range_max=10_000_000,
            presets={TRPreset.MINIMUM: None},
        ),
        UIParam.BANDWIDTH: FloatParam(
            "readout_bandwidth_hz", unit="Hz", range_min=1e3, range_max=2e6
        ),
        UIParam.FOV: FloatParam(
            "fov", unit="mm", scale=1e-3, range_min=50.0, range_max=500.0
        ),
        UIParam.NX: IntParam("n_x", range_min=32, range_max=512, range_incr=2),
        UIParam.ETL: IntParam("blade_width", range_min=2, range_max=64),
        UIParam.NSLICES: IntParam("n_slices", range_min=1, range_max=64),
    }

    def evaluate(self, system, protocol):
        # One TR of the scan is a shot: an excitation, a refocusing pulse and
        # a blade, for each slice of a pass. The design of one slice and one
        # blade at the shortest TR is that shot, and the scan plays each pass
        # once per blade, the dummy blades included.
        a = arguments(self, protocol)
        one = {"n_slices": 1, "tr": None, "n_blades": 1, "n_dummy": 0}
        main = self.app(system, **(protocol.arguments | one))
        shot = main.definitions["TR"][0]
        sizes, passes = _passes(
            a["n_slices"], a["tr"], shot, system.block_duration_raster
        )
        blades = a["n_blades"] or max(
            1, math.ceil(math.pi * a["n_x"] / (2 * a["blade_width"]))
        )
        return Evaluation(
            protocol.replace(achieved(self, main) | {UIParam.TR: max(passes)}),
            (a["n_dummy"] + blades) * sum(passes),
            rf_layout=rf_layout(
                main, scaled=False, copies=max(sizes), period=max(passes)
            ),
        )


def _passes(
    n_slices: int, tr: float | None, shot: float, raster: float
) -> tuple[list[int], list[float]]:
    """Return the slices of each pass, and the duration of one repetition of each.

    Slices one ``tr`` cannot hold are dealt round-robin into passes, and each
    slice of a pass closes with the wait, on the block raster, that makes its
    shot of ``shot`` seconds ``tr`` over the slices of the pass.

    Raises
    ------
    ValueError
        If ``tr`` is shorter than one shot.
    """
    if tr is not None and tr < shot - 1e-9:
        raise ValueError(
            f"TR {tr * 1e3:.1f} ms is shorter than one blade takes "
            f"({shot * 1e3:.1f} ms)"
        )
    per_pass = n_slices if tr is None else max(1, int(tr / shot))
    count = -(-n_slices // per_pass)
    sizes = [len(range(start, n_slices, count)) for start in range(count)]
    waits = [
        0.0 if tr is None else max(pp.round_to_raster(tr / size - shot, raster), 0.0)
        for size in sizes
    ]
    return sizes, [
        size * (shot + wait) for size, wait in zip(sizes, waits, strict=True)
    ]
