"""A minimal scanner sequence: one delay per repetition, with a shortest TE."""

import pypulseqpp as pp
from pypulseqpp import sequences

from pulserver.design import Evaluation, IntParam, SequencePlugin, TimeParam
from pulserver.protocol import TEPreset, UIParam

SHORTEST_TE = 2.5e-3


def tiny(system=None, *, te: float | None = 8e-3, n_repetitions: int = 4):
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


class Tiny(SequencePlugin):
    app = tiny
    protocol = {
        UIParam.TE: TimeParam(
            "te",
            range_min=1000,
            range_max=80000,
            range_incr=100,
            presets={TEPreset.MINIMUM: None},
        ),
        UIParam.NX: IntParam("n_repetitions", range_min=1, range_max=64),
    }

    def evaluate(self, system, protocol):
        sequence = self.app(system, **protocol.arguments)
        return Evaluation(
            protocol.replace({UIParam.TE: sequence.definitions["TE"][0]}),
            sequences.duration(sequence),
        )
