"""Reference weighted like a per-x compressed coil combination: wtruth_cf.py CASE TRUTH OUT VC"""
import pickle, sys, numpy as np
from pulserver.virtual import COILS
case, tpath, out, VC = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = pickle.load(open(f"{case}/dump.pkl", "rb")); nro = rows[0]["data"].shape[-1]; nx = nro // 2
ift = lambda d: np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(d, -1), axis=-1), -1)[..., nro // 2 - nx // 2: nro // 2 + nx // 2]
R = np.stack([ift(r["data"]) for r in rows if r["flags"] & (1 << 19)])
T = np.load(tpath); E = T["echoes"]; nz, ny, _ = E.shape[1:]
zz, yy, xx = [(np.arange(n) - n // 2) * f / n for n, f in ((nz, .24), (ny, .28), (nx, .256))]
Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
S = np.asarray(COILS["head8/head32"].receive(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))).T.reshape(-1, nz, ny, nx)
w = np.zeros((nz, ny, nx))
for x in range(nx):
    cc = np.linalg.svd(R[:, :, x].T, full_matrices=False)[0][:, :VC].conj().T
    sx = S[..., x].reshape(S.shape[0], -1)
    w[..., x] = (np.sqrt((np.abs(cc @ sx) ** 2).sum(0)) / np.maximum(np.sqrt((np.abs(sx) ** 2).sum(0)), 1e-12)).reshape(nz, ny)
np.savez(out, echoes=(E * w).astype(np.float32), keep=T["keep"]); print("weights p1/p50", np.percentile(w[E[0] > 0.05 * E[0].max()], [1, 50]))
