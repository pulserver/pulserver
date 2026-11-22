"""08-layout figure: rows = time points (image, then |error| x5), columns = ground truth then reconstructions; each panel axial | sagittal.
fig2.py OUT TITLE TRUTH.npz RECON.npz:LABEL ...   (env ROWS picks truth entries, XLABEL, TIMES 'label,label,..')"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

out, title, truth, *cols = sys.argv[1:]
z = np.load(truth); T = z["echoes"]; keep = list(z["keep"])
rows_at = [int(r) for r in os.environ.get("ROWS", ",".join(map(str, range(len(keep))))).split(",")]
labels = os.environ.get("TIMES", "").split(",") if os.environ.get("TIMES") else [f"{os.environ.get('XLABEL', 'echo')} {k + 1}" for k in keep]
cols = [c.split(":", 1) for c in cols]
mask = binary_erosion(T.max(0) > 0.1 * T.max(), iterations=4)
ims = []
for f, _ in cols:
    q = np.load(f); R = np.abs(q["echoes"]); fr = list(q["frames"]) if "frames" in q else list(range(len(R)))
    R = R[[fr.index(k) for k in keep]]
    s = (R[:, mask] * T[:, mask]).sum() / (R[:, mask] ** 2).sum()
    ims.append(s * R)
nz, ny, nx = T.shape[1:]
ZAX = nz // 2 + int(os.environ.get("ZSHIFT", 3))
def view(v):
    """Axial (anterior up, subject's left on the right), central sagittal (superior up, anterior left), central coronal (superior up, left on the right).

    The arrays are (z superior, y posterior, x left), the phantom's physical frame.
    """
    ax = v[ZAX]
    sg = v[::-1, :, nx // 2]
    co = v[::-1, ny // 2, :]
    h = max(ax.shape[0], sg.shape[0], co.shape[0])
    pad = lambda a: np.pad(a, (((h - a.shape[0]) // 2, h - a.shape[0] - (h - a.shape[0]) // 2), (0, 0)))
    gap = np.zeros((h, 3))
    return np.concatenate([pad(ax), gap, pad(sg), gap, pad(co)], 1)
names = ["Ground truth"] + [c[1] for c in cols]
fig, axs = plt.subplots(2 * len(rows_at), len(names), figsize=(6.0 * len(names), 2.0 * 2 * len(rows_at)), squeeze=False)
for i, j in enumerate(rows_at):
    v = np.percentile(T[j][mask], 99.5)
    for c in range(len(names)):
        img = T[j] if c == 0 else ims[c - 1][j]
        a, b = axs[2 * i, c], axs[2 * i + 1, c]
        a.imshow(view(img), cmap="gray", vmin=0, vmax=v)
        if c == 0:
            b.text(0.5, 0.5, "reference", color="w", ha="center", va="center", transform=b.transAxes); b.set_facecolor("k")
        else:
            b.imshow(view(5 * np.abs(img - T[j]) * mask), cmap="gray", vmin=0, vmax=v)
            n = np.linalg.norm((img - T[j])[mask]) / np.linalg.norm(T[j][mask])
            b.text(0.02, 0.05, f"NRMSE {100 * n:.1f}%", color="yellow", fontsize=8, transform=b.transAxes)
        for q in (a, b): q.set_xticks([]); q.set_yticks([])
        if i == 0: a.set_title(names[c], fontsize=10)
    axs[2 * i, 0].set_ylabel(labels[j], fontsize=9)
    axs[2 * i + 1, 0].set_ylabel(f"|error| x5\n{labels[j]}", fontsize=9)
fig.suptitle(title, fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=100)
print("wrote", out)
