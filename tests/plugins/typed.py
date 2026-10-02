"""A scanner sequence with one entry of every kind: a choice, a user entry and a configuration key."""

import pypulseqpp as pp
from pypulseqpp import sequences

from pulserver.design import (
    BoolParam,
    ChoiceParam,
    ConfigParam,
    Description,
    FloatParam,
    IntParam,
    ScannerSequence,
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


class TypedApp(sequences.SequenceApp):
    MAX_GRAD = 40.0
    MAX_SLEW = 150.0
    SHORTEST_TE = 2.5e-3

    def init_sequence(
        self,
        te: float | None = 8e-3,
        tr: float | None = None,
        mode: str = "2d",
        n_repetitions: int = 4,
        fat_sat: bool = False,
        thickness: float = 5e-3,
    ) -> None:
        if te is not None and te < self.SHORTEST_TE:
            raise ValueError(
                f"the requested TE of {te * 1e3:.3f} ms is shorter than "
                f"{self.SHORTEST_TE * 1e3:.3f} ms"
            )
        self.te = self.SHORTEST_TE if te is None else te
        self.tr = tr
        self.mode = mode
        self.n_repetitions = n_repetitions
        self.fat_sat = fat_sat
        self.thickness = thickness
        self.duration = n_repetitions * self.te
        self.resolve(te=self.te)

    def loop(self) -> None:
        for _ in range(self.n_repetitions):
            self.kernel()

    def kernel(self) -> None:
        self.seq.add_block(pp.make_delay(self.te))


class Typed(ScannerSequence):
    app = TypedApp
    ui = {
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
