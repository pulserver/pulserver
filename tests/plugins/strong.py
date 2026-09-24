"""A scanner sequence whose gradient is fixed at 60 mT/m, whatever the scanner allows."""

import pypulseqpp as pp
from pypulseqpp import sequences

from pulserver.design import ScannerSequence

DESIGNED_FOR = pp.Opts(max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")


class StrongApp(sequences.SequenceApp):
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0

    def init_sequence(self) -> None:
        self.duration = 2e-3

    def loop(self) -> None:
        self.kernel()

    def kernel(self) -> None:
        amplitude = 60e-3 * DESIGNED_FOR.gamma
        self.seq.add_block(
            pp.make_trapezoid(
                "x",
                amplitude=amplitude,
                flat_time=1e-3,
                rise_time=500e-6,
                system=DESIGNED_FOR,
            )
        )


class Strong(ScannerSequence):
    app = StrongApp
    ui = {}
