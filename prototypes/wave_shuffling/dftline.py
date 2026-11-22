import pickle, sys, numpy as np, torch
from pulserver.virtual import COILS
case = sys.argv[1]; VC = 8; NZ, NY, nx = 120, 140, 128
X = np.load("h_wfse/ctruth.npz")["echoes"][0]
rows = pickle.load(open(f"{case}/dump.pkl", "rb"))
ref = [r for r in rows if r["flags"] & (1 << 19)]
Rr = np.stack([r["data"] for r in ref]); cc = np.linalg.svd(Rr.transpose(1, 0, 2).reshape(32, -1), full_matrices=False)[0][:, :VC].conj().T
img = [r for r in rows if not r["flags"] & (1 << 19) and r["con"] == 0]
zz, yy, xx = [(np.arange(n) - n // 2 + 0.5) * f / n for n, f in ((NZ, .24), (NY, .28), (nx, .256))]
Z, Y, Xg = np.meshgrid(zz, yy, xx, indexing="ij")
m = np.abs(X) > 1e-3 * np.abs(X).max()
P = np.stack([Xg[m], Y[m], Z[m]], 1)
S = np.asarray(COILS["head8/head32"].receive(P)).T
v = torch.from_numpy((cc @ S) * X[m][None]).to(torch.complex128)  # (VC, n)
Pt = torch.from_numpy(P)
img.sort(key=lambda r: (r["e1"] - 70) ** 2 + (r["e2"] - 60) ** 2)
sel = [img[0], img[len(img) // 8], img[len(img) // 3]]
for r in sel:
    t = r["traj"]
    if t.size == 0:
        n_ = 256; t = np.stack([(np.arange(n_) - n_ // 2 + 0.5) / 0.512, np.full(n_, (r["e1"] - 70) / 0.28), np.full(n_, (r["e2"] - 60) / 0.24)], -1)
    tt = torch.from_numpy(t.astype(np.float64)); mo = np.zeros((VC, len(t)), complex)
    for a in range(0, Pt.shape[0], 100000):
        E = torch.exp(-2j * np.pi * (tt @ Pt[a:a + 100000].T)); mo += (v[:, a:a + 100000] @ E.T).numpy()
    me = cc @ r["data"]
    g = (mo.conj() * me).sum() / (np.abs(mo) ** 2).sum()
    print(case, r["e1"], r["e2"], "residual", round(float(np.linalg.norm(g * mo - me) / np.linalg.norm(me)), 3), "|g|", abs(g))
