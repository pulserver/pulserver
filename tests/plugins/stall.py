"""A plugin whose design marks that it started, then does not return."""

import os
import time
from pathlib import Path

from pulserver.design import SequencePlugin, TimeParam
from pulserver.protocol import UIParam


def stall(system=None, *, te: float = 8e-3):
    Path(os.environ["PULSERVER_STALL_MARKER"]).touch()
    time.sleep(600)


class Stall(SequencePlugin):
    app = stall
    protocol = {UIParam.TE: TimeParam("te", range_min=1000, range_max=80000)}

    def evaluate(self, system, protocol):
        self.app(system, **protocol.arguments)
