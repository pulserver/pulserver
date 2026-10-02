"""An echo train that states its RF layout: one excitation and its refocusing pulses."""

import numpy as np
import pypulseqpp as pp

from pulserver.design import Evaluation, FloatParam, IntParam, RfLayout, SequencePlugin
from pulserver.protocol import UIParam, UserKey


def echo_train(system, flip=90.0, refocusing=150.0, echoes=8):
    seq = pp.Sequence(system)
    seq.add_block(
        pp.make_block_pulse(
            np.deg2rad(flip), duration=1e-3, system=system, use="excitation"
        )
    )
    seq.add_block(pp.make_delay(2e-3))
    for _ in range(echoes):
        seq.add_block(
            pp.make_block_pulse(
                np.deg2rad(refocusing), duration=1e-3, system=system, use="refocusing"
            )
        )
        seq.add_block(pp.make_delay(4e-3))
    return seq


class RfTrain(SequencePlugin):
    app = echo_train
    protocol = {
        UIParam.FLIP: FloatParam("flip", unit="deg", range_min=0.0, range_max=180.0),
        UserKey.USER0: FloatParam(
            "refocusing", unit="deg", range_min=0.0, range_max=180.0
        ),
        UIParam.ETL: IntParam("echoes", range_min=1, range_max=64),
    }

    def evaluate(self, system, protocol):
        seq = echo_train(system, **protocol.arguments)
        controls = [UIParam.FLIP, *[UserKey.USER0] * protocol[UIParam.ETL]]
        return Evaluation(
            protocol, seq.duration()[0], rf_layout=RfLayout.of(seq, controls)
        )
