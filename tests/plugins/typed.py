"""A scanner sequence with one entry of every kind: a choice, a user entry and a configuration key."""

import pypulseqpp as pp

from pulserver.design import (
    BoolParam,
    ChoiceParam,
    ConfigParam,
    Description,
    Evaluation,
    FloatParam,
    IntParam,
    SequencePlugin,
    TimeParam,
)
from pulserver.protocol import (
    ConfigKey,
    ImagingMode,
    TEPreset,
    TRPreset,
    UIParam,
    UserKey,
    UserNameKey,
)

SHORTEST_TE = 2.5e-3


def typed(
    system=None,
    *,
    te: float | None = 8e-3,
    tr: float | None = None,
    mode: str = "2d",
    n_repetitions: int = 4,
    fat_sat: bool = False,
    thickness: float = 5e-3,
):
    if te is not None and te < SHORTEST_TE:
        raise ValueError(
            f"the requested TE of {te * 1e3:.3f} ms is shorter than "
            f"{SHORTEST_TE * 1e3:.3f} ms"
        )
    te = SHORTEST_TE if te is None else te
    seq = pp.Sequence(system)
    seq.set_definition("TE", [te])
    for _ in range(n_repetitions):
        seq.add_block(pp.make_delay(te))
    return seq


class Typed(SequencePlugin):
    app = typed
    protocol = {
        UIParam.TE: TimeParam(
            "te",
            range_min=1000,
            range_max=80000,
            range_incr=100,
            presets={TEPreset.MINIMUM: None},
        ),
        UIParam.TR: TimeParam(
            "tr", range_min=1000, range_max=5_000_000, presets={TRPreset.MINIMUM: None}
        ),
        UIParam.IMAGING_MODE: ChoiceParam("mode", ImagingMode),
        UIParam.NX: IntParam("n_repetitions", range_min=1, range_max=64),
        UIParam.FAT_SAT: BoolParam("fat_sat"),
        UserNameKey.USER0: Description("Thickness"),
        UserKey.USER0: FloatParam(
            "thickness", unit="mm", scale=1e-3, range_min=1.0, range_max=50.0
        ),
        ConfigKey.ENABLE_SAR_BURST_MODE: ConfigParam(1),
    }

    def evaluate(self, system, protocol):
        arguments = protocol.arguments
        te = self.app(system, **arguments).definitions["TE"][0]
        return Evaluation(
            protocol.replace({UIParam.TE: te}), arguments["n_repetitions"] * te
        )
