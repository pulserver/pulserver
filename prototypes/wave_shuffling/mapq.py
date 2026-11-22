"""Coherence of compressed-calibration nlinv maps with the simulator's coils compressed the same way."""
import pickle, sys, time, numpy as np, torch
from bartorch import apps
from pulserver.virtual import COILS
from pulserver.recon.handlers.pics import nlinv_estimate
rows = pickle.load(open("h_fse48/dump.pkl", "rb")); nro = 256; nx = 128; NZ, NY = 120, 140
ref = [r for r in rows if r["flags"] & (1 << 19)]
Rr = np.stack([r["data"] for r in ref])
cc = np.linalg.svd(Rr.transpose(1, 0, 2).reshape(32, -1), full_matrices=False)[0][:, :8].conj().T.astype(np.complex64)
def deos(d):
    x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(d, -1), axis=-1), -1)[..., 64:192]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, -1), axis=-1), -1)
k = np.zeros((8, NZ, NY, nx), np.complex64)
for r in ref: k[:, r["e2"], r["e1"]] = deos(cc @ r["data"])
T = np.load("h_fses/truth.npz")["echoes"][0]; m = T > 0.1 * T.max()
zz, yy, xx = [(np.arange(n) - n // 2) * f / n for n, f in ((NZ, .24), (NY, .28), (nx, .256))]
Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
S = (cc @ np.asarray(COILS["head8/head32"].receive(np.stack([X[m], Y[m], Z[m]], 1))).T)
C = np.load("rss_helmet.npy")[m]
def coh(maps, label):
    a = maps.numpy()[:, m] if hasattr(maps, "numpy") else maps[:, m]
    c = np.abs((a.conj() * S).sum(0)) / (np.linalg.norm(a, axis=0) * np.linalg.norm(S, axis=0) + 1e-12)
    low = C < np.percentile(C, 30)
    print(f"{label:30s} coherence mean {c.mean():.3f}  centre(low RSS) {c[low].mean():.3f}", flush=True)
for size in (48, 96):
    t = time.perf_counter(); mp = apps.nlinv_maps(torch.from_numpy(k), None, size=size, radius=size / 2, phase_pole=0, sobolev=15) if len(sys.argv) > 1 else apps.nlinv_maps(torch.from_numpy(k), None, size=size, radius=size / 2, phase_pole=0)
    coh(mp, f"3D nlinv size {size} ({time.perf_counter()-t:.0f} s)")
