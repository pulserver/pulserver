"""A plugin whose design marks that it started, then does not return."""

import os
import time
from pathlib import Path

from pypulseqpp import sequences

from pulserver.design import ScannerSequence, TimeParam


class StallApp(sequences.SequenceApp):
    MAX_GRAD = 40.0
    MAX_SLEW = 150.0

    def init_sequence(self, te: float = 8e-3) -> None:
        Path(os.environ["PULSERVER_STALL_MARKER"]).touch()
        time.sleep(600)

    def loop(self) -> None:
        pass

    def kernel(self) -> None:
        pass


class Stall(ScannerSequence):
    app = StallApp
    ui = {"TE": TimeParam("te", range_min=1000, range_max=80000)}
