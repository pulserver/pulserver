"""A scanner sequence whose gradient is fixed at 60 mT/m, whatever the scanner allows."""

import pypulseqpp as pp
from pypulseqpp import sequences

from pulserver.design import Evaluation, SequencePlugin

DESIGNED_FOR = pp.Opts(max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")


def strong(system=None):
    seq = pp.Sequence(system)
    seq.add_block(
        pp.make_trapezoid(
            "x",
            amplitude=60e-3 * DESIGNED_FOR.gamma,
            flat_time=1e-3,
            rise_time=500e-6,
            system=DESIGNED_FOR,
        )
    )
    return seq


class Strong(SequencePlugin):
    app = strong
    protocol = {}

    def evaluate(self, system, protocol):
        return Evaluation(protocol, sequences.duration(self.app(system)))
