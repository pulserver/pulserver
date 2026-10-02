"""A reconstruction that raises on the first readout it is given."""

from pulserver.recon import Gadget, ReconPlugin


class Fail(Gadget):
    def __call__(self, acquisition, data):
        raise RuntimeError("this reconstruction always fails")


class CrashRecon(ReconPlugin):
    def __init__(self):
        super().__init__(gadgets=[Fail()])

    def recon(self, context, branch, data):
        return None


PLUGIN = CrashRecon()
