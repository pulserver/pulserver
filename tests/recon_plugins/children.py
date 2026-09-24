"""Computes its image in a child process, as a reconstruction that parallelises with multiprocessing does."""

import multiprocessing

import numpy as np

from pulserver.mrd import AcquisitionFlag
from pulserver.recon import ReconPlugin, ReconResult


class ChildProcessRecon(ReconPlugin):
    def __init__(self):
        super().__init__(branches={AcquisitionFlag.LAST_IN_MEASUREMENT: "image"})

    def recon(self, branch, context):
        # A spawned child imports what it runs by module name, which this
        # file, loaded from a path, does not have.
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            return ReconResult(pool.apply(np.ones, ((2, 2),)))


PLUGIN = ChildProcessRecon()
