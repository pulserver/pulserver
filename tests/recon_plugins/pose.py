"""States two poses, each absolute, then returns an image."""

import numpy as np

from pulserver.proxy._motion import Pose, pose_waveform
from pulserver.recon import ReconPlugin, ReconResult

#: A quarter turn about z, then a quarter turn about x: the second is where the
#: object is, not how far it moved since the first.
POSES = (
    (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0),
)


class PoseRecon(ReconPlugin):
    def recon(self, context, branch, data):
        return [
            *(pose_waveform(Pose(rotation=rotation)) for rotation in POSES),
            ReconResult(np.ones((2, 2), dtype=np.float32)),
        ]


PLUGIN = PoseRecon()
