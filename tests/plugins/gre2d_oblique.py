"""The 2D gradient echo held within the tests' scanner limits in any orientation."""

import math

from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp

from pulserver.design import IntParam, ScannerSequence, TimeParam
from pulserver.protocol import UIParam


class ObliqueGre2DApp(Gre2DApp):
    # Each logical axis at 1/sqrt(3) of the 40 mT/m and 150 T/m/s the tests'
    # scanner allows keeps every physical axis within them after any rotation.
    MAX_GRAD = 40.0 / math.sqrt(3.0)
    MAX_SLEW = 150.0 / math.sqrt(3.0)


class ObliqueGre2D(ScannerSequence):
    app = ObliqueGre2DApp
    recon = "gre2d"
    ui = {
        UIParam.TE: TimeParam("te", range_min=1000, range_max=80000, range_incr=10),
        UIParam.NX: IntParam("n_x", range_min=32, range_max=512, range_incr=2),
        UIParam.NY: IntParam("n_y", range_min=32, range_max=512, range_incr=2),
    }
