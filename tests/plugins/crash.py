"""A plugin whose design kills the worker process."""

import os

from pulserver.design import SequencePlugin, TimeParam
from pulserver.protocol import UIParam


def crash(system=None, *, te: float = 8e-3):
    os._exit(1)


class Crash(SequencePlugin):
    app = crash
    protocol = {UIParam.TE: TimeParam("te", range_min=1000, range_max=80000)}

    def evaluate(self, system, protocol):
        self.app(system, **protocol.arguments)
