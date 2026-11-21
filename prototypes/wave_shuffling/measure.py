"""Replay a series through a plugin and report wall time, peak RAM and peak VRAM: measure.py RAW PLUGIN STORE OUT."""

import resource
import sys
import time

import torch

from pulserver._cli import main
import os
if os.environ.get("DEVICE"):
    # Offline replay has no proxy to hand out a GPU: give the series the one named.
    import pulserver.recon._runtime.offline as _offline
    _context = _offline.ReconContext
    _offline.ReconContext = lambda *a, **k: _context(*a, device=os.environ["DEVICE"], **k)

raw, plugin, store, out = sys.argv[1:5]
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t = time.perf_counter()
code = main(["replay", raw, "--plugin", plugin, "--store", store, "--output", out])
wall = time.perf_counter() - t
ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
vram = (
    torch.cuda.max_memory_allocated() / 2**20
    if torch.cuda.is_available()
    else float("nan")
)
print(
    f"MEASURE wall {wall:.1f} s, peak RAM {ram:.0f} MiB, peak VRAM {vram:.0f} MiB, exit {code}"
)
