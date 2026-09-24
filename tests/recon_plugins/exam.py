"""Counts the series of its exam in the exam cache, and returns the count as its image."""

import numpy as np

from pulserver.mrd import AcquisitionFlag
from pulserver.recon import ReconPlugin, ReconResult


class ExamCounter(ReconPlugin):
    def __init__(self):
        super().__init__(branches={AcquisitionFlag.LAST_IN_MEASUREMENT: "count"})

    def recon(self, branch, context):
        count = context.exam.get("series", 0) + 1
        context.exam["series"] = count
        return ReconResult(np.full((2, 2), count, dtype=np.float32))


PLUGIN = ExamCounter()
