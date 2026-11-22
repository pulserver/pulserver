import pickle, numpy as np, torch
from bartorch import linop
from pulserver.virtual import COILS
VC = 8; NZ, NY, nx, nro = 120, 140, 128, 256
X = np.load("h_wfse/ctruth.npz")["echoes"][0]
rows = pickle.load(open("h_wfse/dump.pkl", "rb"))
ref = [r for r in rows if r["flags"] & (1 << 19)]
Rr = np.stack([r["data"] for r in ref]); cc = np.linalg.svd(Rr.transpose(1, 0, 2).reshape(32, -1), full_matrices=False)[0][:, :VC].conj().T
img = [r for r in rows if not r["flags"] & (1 << 19) and r["con"] == 0]
img.sort(key=lambda r: (r["e1"] - 70) ** 2 + (r["e2"] - 60) ** 2); r = img[0]
zz, yy, xx = [(np.arange(n) - n // 2 + 0.5) * f / n for n, f in ((NZ, .24), (NY, .28), (nx, .256))]
Z, Y, Xg = np.meshgrid(zz, yy, xx, indexing="ij")
S = np.asarray(COILS["head8/head32"].receive(np.stack([Xg.ravel(), Y.ravel(), Z.ravel()], 1))).T.reshape(32, NZ, NY, nx)
M = (np.einsum("vc,czyx->vzyx", cc, S)).astype(np.complex64); del S
t = r["traj"]; wk = np.stack([t[:, 2] - t[:, 2].mean(), t[:, 1] - t[:, 1].mean()], -1)
pos = torch.tensor([[[r["e2"], r["e1"]]]])
me = cc @ r["data"]
m = np.abs(X) > 1e-3 * np.abs(X).max(); Pt = torch.from_numpy(np.stack([Xg[m], Y[m], Z[m]], 1)); v = torch.from_numpy(M[:, m] * X[m][None]).to(torch.complex128)
tt = torch.from_numpy(t.astype(np.float64)); dft = np.zeros((VC, nro), complex)
for a in range(0, Pt.shape[0], 100000):
    dft += (v[:, a:a + 100000] @ torch.exp(-2j * np.pi * (tt @ Pt[a:a + 100000].T)).T).numpy()
def cmp(name, mo):
    g = (mo.conj() * me).sum() / (np.abs(mo) ** 2).sum(); h = (mo.conj() * dft).sum() / (np.abs(mo) ** 2).sum()
    print(f"{name:28s} vs data {np.linalg.norm(g * mo - me) / np.linalg.norm(me):.3f}  vs exact DFT {np.linalg.norm(h * mo - dft) / np.linalg.norm(dft):.3f}  |h| {abs(h):.3g}")
cmp("exact DFT", dft)
for name, w_ in [("psf +wk", wk), ("psf -wk", -wk), ("psf rev", wk[::-1].copy()), ("psf swap", wk[:, ::-1].copy()), ("no wave", 0 * wk)]:
    psf = linop.wave_psf(w_, (NZ, NY, nx), (0.24, 0.28), centred=True).to(torch.complex64)
    A = linop.WaveSense(torch.from_numpy(M), (1, NZ, NY, nx), nro, positions=pos, basis=torch.eye(1, dtype=torch.complex64), centred=True, psf=psf)
    mo = A(torch.from_numpy(X[None].astype(np.complex64))).numpy()[:, 0, 0]
    cmp(name, mo)
    if name == "psf +wk":
        for sh in (1, -1):
            cmp(f"  rolled {sh}", np.roll(mo, sh, -1))
        cmp("  reversed", mo[:, ::-1])
psf = linop.wave_psf(wk, (NZ, NY, nx), (0.24, 0.28), centred=True).numpy()
im = M * X[None]
pad = np.zeros((VC, NZ, NY, nro), np.complex64); pad[..., 64:192] = im
cft = lambda a, ax: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(a, ax), axis=ax, norm="ortho"), ax)
h = cft(pad, -1) * psf[None]
h = cft(cft(h, 1), 2)
cmp("manual chain", h[:, r["e2"], r["e1"]])
h2 = cft(pad, -1)
h2 = cft(cft(h2, 1), 2)
cmp("manual no wave", h2[:, r["e2"], r["e1"]])
# exact line from shifted hybrid: data(t) = sum_{z,y} H(z,y,t) exp(-2pi i (ky0 y + kz0 z)) psf
dx = 0.256 / nx
xp = (np.arange(nro) - nro // 2) * dx
kxt = t[:, 0]
for name, w_ in [("exactchain +wk", wk), ("exactchain no wave", 0 * wk)]:
    pad2 = pad * np.exp(-2j * np.pi * (0.5 / 0.512) * xp)[None, None, None]
    hh = cft(pad2, -1) * np.exp(-2j * np.pi * kxt * dx / 2)[None, None, None]
    zt = (np.arange(NZ) - NZ // 2 + 0.5) * 0.24 / NZ; yt = (np.arange(NY) - NY // 2 + 0.5) * 0.28 / NY
    ps = np.exp(-2j * np.pi * (w_[None, None, :, 0] * zt[:, None, None] + w_[None, None, :, 1] * yt[None, :, None]))
    hh = cft(cft(hh * ps[None], 1), 2)
    cmp(name, hh[:, r["e2"], r["e1"]])
