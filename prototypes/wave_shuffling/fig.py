"""08-layout figure against the phantom's ground truth: fig.py OUT TITLE TRUTHCASE RECON.npz:LABEL ...

Rows: three echoes (images), each followed by its |error| x5 against that case's truth;
NRMSE in the head printed on each error panel. ROWS env picks entries of the truth's echo list.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, title, truthcase, *cols = sys.argv[1:]
z = np.load(f"{truthcase}/truth.npz"); T = z["echoes"]; keep = list(z["keep"])
rows_at = [int(r) for r in os.environ.get("ROWS", "0,2,4").split(",")]
label_of = os.environ.get("XLABEL", "echo")
cols = [c.split(":", 1) for c in cols]
from scipy.ndimage import binary_erosion
mask = binary_erosion(T[0] > 0.1 * T[0].max(), iterations=4)
ims = []
for f, _ in cols:
    R = np.abs(np.load(f)["echoes"])[keep]
    s = (R[:, mask] * T[:, mask]).sum() / (R[:, mask] ** 2).sum()
    ims.append(s * R)
sl = T.shape[1] // 2
names = ["Ground truth"] + [c[1] for c in cols]
fig, ax = plt.subplots(2 * len(rows_at), len(names), figsize=(2.5 * len(names), 2.5 * 2 * len(rows_at)), squeeze=False)
for i, j in enumerate(rows_at):
    v = np.percentile(T[j, sl], 99.5)
    for c in range(len(names)):
        img = T[j, sl] if c == 0 else ims[c - 1][j, sl]
        a, b = ax[2 * i, c], ax[2 * i + 1, c]
        a.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=v)
        if c == 0:
            b.text(0.5, 0.5, "reference", color="w", ha="center", va="center", transform=b.transAxes); b.set_facecolor("k")
        else:
            b.imshow(np.rot90(5 * np.abs(img - T[j, sl])), cmap="gray", vmin=0, vmax=v)
            n = np.linalg.norm((ims[c - 1][j] - T[j])[mask]) / np.linalg.norm(T[j][mask])
            b.text(0.03, 0.04, f"NRMSE {100 * n:.1f}%", color="yellow", fontsize=8, transform=b.transAxes)
        for q in (a, b): q.set_xticks([]); q.set_yticks([])
        if i == 0: a.set_title(names[c], fontsize=10)
    ax[2 * i, 0].set_ylabel(f"{label_of} {keep[j] + 1}", fontsize=9)
    ax[2 * i + 1, 0].set_ylabel(f"|error| x5\n{label_of} {keep[j] + 1}", fontsize=9)
fig.suptitle(title, fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
