"""Returns the index of the GPU its series was given, plus one, as its image; 0 on the host."""

import numpy as np

from pulserver.mrd import AcquisitionFlag
from pulserver.recon import ReconPlugin, ReconResult


class DeviceRecon(ReconPlugin):
    def __init__(self):
        super().__init__(branches={AcquisitionFlag.LAST_IN_MEASUREMENT: "device"})

    def recon(self, branch, context):
        index = 0 if context.device is None else int(context.device.split(":")[1]) + 1
        return ReconResult(np.full((2, 2), index, dtype=np.float32))


PLUGIN = DeviceRecon()
