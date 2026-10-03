"""The 2D gradient echo whose protocol is its echo time and its matrix."""

from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d

from pulserver.design import IntParam, SequencePlugin, TimeParam
from pulserver.protocol import UIParam


class Gre2DRaw(SequencePlugin):
    app = gre2d
    protocol = {
        UIParam.TE: TimeParam("te", range_min=1000, range_max=80000, range_incr=10),
        UIParam.NX: IntParam("n_x", range_min=32, range_max=512, range_incr=2),
        UIParam.NY: IntParam("n_y", range_min=32, range_max=512, range_incr=2),
    }
