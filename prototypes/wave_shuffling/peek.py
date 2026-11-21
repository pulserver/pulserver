import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fs = sys.argv[2:]
out = sys.argv[1]
f, a = plt.subplots(len(fs), 5, figsize=(13, 2.8 * len(fs)), squeeze=False)
for j, fn in enumerate(fs):
    e = np.abs(np.load(fn)["echoes"])
    for i, k in enumerate([0, 4, 9, 19, 39]):
        a[j, i].imshow(
            np.rot90(e[k, 24]), cmap="gray", vmax=np.percentile(e[k, 24], 99.5)
        )
        a[j, i].set_xticks([])
        a[j, i].set_yticks([])
        if j == 0:
            a[j, i].set_title(f"echo {k + 1}")
    a[j, 0].set_ylabel(fn.split("/")[-1], fontsize=8)
plt.tight_layout()
plt.savefig(out, dpi=60)
