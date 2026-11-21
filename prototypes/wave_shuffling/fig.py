"""08-layout figure: fig.py OUT TITLE REF:LABEL CASE:LABEL ... (echo images of each recon, error x5 rows vs the first)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, title, *cases = sys.argv[1:]
ECHOES = [int(e) for e in __import__("os").environ.get("ECHOES", "1,9,24").split(",")]
ESP = float(__import__("os").environ.get("ESP_MS", "4.08"))
cols = [c.split(":", 1) for c in cases]
ims = [np.abs(np.load(f)["echoes"]) for f, _ in cols]
ref = ims[0]
z = ref.shape[1] // 2
rows = []
for e in ECHOES:
    rows.append((f"echo {e + 1}\nTE {ESP * (e + 1):.0f} ms", [im[e, z] for im in ims], False, e))
    rows.append((f"|error| x5\necho {e + 1}", [5 * np.abs(im[e, z] - ref[e, z]) for im in ims], True, e))
fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.6 * len(rows)), squeeze=False)
for i, (label, row, err, e) in enumerate(rows):
    for j, img in enumerate(row):
        a = ax[i, j]
        a.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=np.percentile(ref[e, z], 99.5))
        a.set_xticks([]); a.set_yticks([])
        if err and j > 0:
            m = np.abs(ims[j][e] - ref[e]); n = np.linalg.norm(m) / np.linalg.norm(ref[e])
            a.text(0.03, 0.04, f"NRMSE {100 * n:.1f}%", color="yellow", fontsize=8, transform=a.transAxes)
        if err and j == 0:
            a.text(0.5, 0.5, "reference", color="w", ha="center", va="center", transform=a.transAxes)
        if i == 0:
            a.set_title(cols[j][1], fontsize=10)
        if j == 0:
            a.set_ylabel(label, fontsize=9)
fig.suptitle(title, fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
