"""3D residual of measured readouts against the signal equation at a few echoes: wresid.py CASE CTRUTH  (env SIGN, DELAY, NOWAVE, VC)"""
import os, pickle, sys, numpy as np, torch
from bartorch import linop
from pulserver.virtual import COILS
case, ct = sys.argv[1], sys.argv[2]
VC = int(os.environ.get("VC", 8)); NZ, NY = 120, 140; FOVY, FOVZ = 0.28, 0.24
T = np.load(ct); K = list(T["keep"]); X = T["echoes"]
X = (X * np.exp(-2j * np.pi * float(os.environ.get('KXSHIFT', 0.25)) * (np.arange(X.shape[-1]) - X.shape[-1] // 2) / X.shape[-1])).astype(np.complex64)
rows = pickle.load(open(f"{case}/dump.pkl", "rb")); nro = rows[0]["data"].shape[-1]; nx = nro // 2
ref = [r for r in rows if r["flags"] & (1 << 19)]
Rr = np.stack([r["data"] for r in ref])
cc = np.linalg.svd(Rr.transpose(1, 0, 2).reshape(Rr.shape[1], -1), full_matrices=False)[0][:, :VC].conj().T.astype(np.complex64)
img = [r for r in rows if not r["flags"] & (1 << 19) and r["con"] in K]
w = img[0]["traj"]; wk = np.stack([w[:, 2] - (img[0]["e2"] - NZ // 2) / FOVZ, w[:, 1] - (img[0]["e1"] - NY // 2) / FOVY], -1) if w.size else None
if wk is not None and os.environ.get("DELAY"):
    _t = np.arange(len(wk)); _dl = float(os.environ["DELAY"]); wk = np.stack([np.interp(_t - _dl, _t, wk[:, a]) for a in range(2)], -1)
if wk is not None and os.environ.get('WREV'): wk = wk[::-1].copy()
if wk is not None and os.environ.get('WSWAP'): wk = wk[:, ::-1].copy()
if os.environ.get('K1'): K = K[:1]; X = X[:1]
by = [[r for r in img if r["con"] == k] for k in K]; shots = max(len(b) for b in by)
pos = -np.ones((len(K), shots, 2), np.int64); y = np.zeros((VC, len(K), shots, nro), np.complex64)
for f, b in enumerate(by):
    for s_, r in enumerate(b): pos[f, s_] = (r["e2"], r["e1"]); y[:, f, s_] = cc @ r["data"]
zz, yy, xx = [(np.arange(n) - n // 2 + 0.5) * f / n for n, f in ((NZ, FOVZ), (NY, FOVY), (nx, 0.256))]
Z, Y, Xg = np.meshgrid(zz, yy, xx, indexing="ij")
S = np.asarray(COILS["head8/head32"].receive(np.stack([Xg.ravel(), Y.ravel(), Z.ravel()], 1))).T
maps = torch.from_numpy((cc @ S).reshape(VC, NZ, NY, nx).astype(np.complex64)); del S
if os.environ.get("NLMAPS"):
    from pulserver.recon.handlers.pics import nlinv_estimate
    def deos(d):
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(d, -1), axis=-1), -1)[..., nro // 2 - nx // 2: nro // 2 + nx // 2]
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, -1), axis=-1), -1)
    k3 = np.zeros((VC, NZ, NY, nx), np.complex64)
    for r in ref: k3[:, r["e2"], r["e1"]] = deos(cc @ r["data"])
    rssc = maps.abs().pow(2).sum(0).sqrt().numpy()
    maps = nlinv_estimate(torch.from_numpy(k3), sobolev=15.0).to(torch.complex64); del k3
    X = (X * rssc[None]).astype(np.complex64); print("nlinv maps")
sign = float(os.environ.get("SIGN", 1))
psf = torch.ones(NZ, NY, nro, dtype=torch.complex64) if (wk is None or os.environ.get("NOWAVE")) else linop.wave_psf(sign * wk, (NZ, NY, nx), (FOVZ, FOVY), centred=True).to(torch.complex64)
A = linop.WaveSense(maps, (len(K), NZ, NY, nx), nro, positions=torch.from_numpy(pos), basis=torch.eye(len(K), dtype=torch.complex64), centred=True, psf=psf)
mo = A(torch.from_numpy(X)).numpy(); valid = pos[..., 0] >= 0
g = (mo[:, valid].conj() * y[:, valid]).sum() / (np.abs(mo[:, valid]) ** 2).sum()
res = g * mo - y
print(f"{case} sign {sign} delay {os.environ.get('DELAY', 0)} nowave {bool(os.environ.get('NOWAVE'))}: overall {np.linalg.norm(res[:, valid]) / np.linalg.norm(y[:, valid]):.3f}",
      "by echo", [round(float(np.linalg.norm(res[:, f][:, valid[f]]) / np.linalg.norm(y[:, f][:, valid[f]])), 3) for f in range(len(K))])
for f in range(len(K)):
    a_, b_ = mo[:, f][:, valid[f]], y[:, f][:, valid[f]]
    gf = (a_.conj() * b_).sum() / (np.abs(a_) ** 2).sum()
    print(f"echo {K[f]+1}: own-fit |g|/g {abs(gf)/abs(g):.3f} phase {np.angle(gf/g):+.2f} residual {np.linalg.norm(gf*a_-b_)/np.linalg.norm(b_):.3f}")
