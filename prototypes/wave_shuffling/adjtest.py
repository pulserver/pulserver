
import numpy as np

exec(
    open("proto.py")
    .read()
    .split("# Frames = echo index")[0]
    .replace(
        "case, rank, lam, iters = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])",
        "case='fse'",
    )
    .replace(
        'basis_full = torch.load(sys.argv[5]) if len(sys.argv) > 5 else torch.load(f"{case}/basis.pt")',
        "",
    )
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Direct gridding of the wave-free lines + waved lines ignoring the wave, centred FFT recon, coil-combined.
k = np.zeros((VC, nz, ny, nro), np.complex64)
c = np.zeros((nz, ny))
for r in rows:
    k[:, r["e2"], r["e1"]] += comp(r["data"])
    c[r["e2"], r["e1"]] += 1
k /= np.maximum(c, 1)[None, :, :, None]
img = np.fft.fftshift(
    np.fft.ifftn(np.fft.ifftshift(k, axes=(1, 2, 3)), axes=(1, 2, 3)), axes=(1, 2, 3)
)
rss = np.sqrt((np.abs(img) ** 2).sum(0))
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(rss[24], cmap="gray")
ax[1].imshow(rss[:, 32], cmap="gray", aspect=2)
ax[2].imshow(np.abs(maps[0, 24]), cmap="gray")
fig.savefig("fse/adj.png", dpi=70)
print("center samples", sorted(set(r["center"] for r in rows)))
