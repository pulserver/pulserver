import sys

import numpy as np

a = np.load("h_ref/recon_truemaps.npz")["echoes"]
for f in sys.argv[1:]:
    b = np.load(f)["echoes"]
    print(
        f,
        [
            round(float(np.linalg.norm(b[e] - a[e]) / np.linalg.norm(a[e])), 3)
            for e in (0, 4, 9, 19, 29, 39)
        ],
    )
