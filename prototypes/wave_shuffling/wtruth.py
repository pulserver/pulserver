"""Reference weighted like the recon's coil combination: truth * RSS(compressed true maps) / RSS(all coils). wtruth.py CASE TRUTH OUT VC"""
import pickle, sys, numpy as np
from pulserver.virtual import COILS
case, tpath, out, VC = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = pickle.load(open(f"{case}/dump.pkl", "rb"))
flat = [r["data"] for r in rows]  # no wave: every line is wave-free
calib = np.stack(flat)
u, s, _ = np.linalg.svd(calib.transpose(1, 0, 2).reshape(calib.shape[1], -1), full_matrices=False)
cc = u[:, :VC].conj().T
T = np.load(tpath); E = T["echoes"]; nz, ny, nx = E.shape[1:]
zz, yy, xx = [(np.arange(n) - n // 2 + float(__import__("os").environ.get("HALF", 0))) * f / n for n, f in ((nz, .24), (ny, .28), (nx, .256))]
Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
sens = np.asarray(COILS["head8/head32"].receive(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))).T
w = (np.sqrt((np.abs(cc @ sens) ** 2).sum(0)) / np.maximum(np.sqrt((np.abs(sens) ** 2).sum(0)), 1e-12)).reshape(nz, ny, nx)
print("weight range in head", np.round(np.percentile(w[E[0] > 0.1 * E[0].max()], [1, 50, 99]), 3))
np.savez(out, echoes=(E * w).astype(np.float32), keep=T["keep"])
