"""The 2D gradient echo without a bound reconstruction: the client's config chooses."""

from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp

from pulserver.design import IntParam, ScannerSequence, TimeParam
from pulserver.protocol import UIParam


class Gre2DRaw(ScannerSequence):
    app = Gre2DApp
    ui = {
        UIParam.TE: TimeParam("te", range_min=1000, range_max=80000, range_incr=10),
        UIParam.NX: IntParam("n_x", range_min=32, range_max=512, range_incr=2),
        UIParam.NY: IntParam("n_y", range_min=32, range_max=512, range_incr=2),
    }

    def resolved(self, app):
        return {"te": app.ro.echo_time}
