"""A plugin whose design kills the worker process."""

import os

from pypulseqpp import sequences

from pulserver.design import ScannerSequence, TimeParam
from pulserver.protocol import UIParam


class CrashApp(sequences.SequenceApp):
    MAX_GRAD = 40.0
    MAX_SLEW = 150.0

    def init_sequence(self, te: float = 8e-3) -> None:
        os._exit(1)

    def loop(self) -> None:
        pass

    def kernel(self) -> None:
        pass


class Crash(ScannerSequence):
    app = CrashApp
    ui = {UIParam.TE: TimeParam("te", range_min=1000, range_max=80000)}
