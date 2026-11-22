"""Compression-first 3D wave-shuffling subspace recon: wave3d.py CASE BASIS.pt RANK LAM ITERS TAG [SIGN]

SVD compression of the raw lines to VC virtual coils (from the wave-free calibration trains), 3D nlinv
on the compressed calibration trains, wave PSF from the streamed trajectory, one 3D solve.
"""
import os, pickle, resource, sys, time
import numpy as np, torch
from bartorch import linop, optim, priors
from pulserver.recon.handlers.pics import nlinv_estimate

case, bpath, rank, lam, iters, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sign = float(sys.argv[7]) if len(sys.argv) > 7 else -1.0
VC = int(os.environ.get("VC", 6)); NY, NZ = int(os.environ.get("NY", 140)), int(os.environ.get("NZ", 120))
FOVY, FOVZ = float(os.environ.get("FOVY", 0.28)), float(os.environ.get("FOVZ", 0.24))
KEEP = [int(v) for v in os.environ.get("KEEP", "0,4,9,19,29,39").split(",")]
sob = {"sobolev": float(os.environ["SOBOLEV"])} if os.environ.get("SOBOLEV") else {}
t0 = time.perf_counter()
rows = pickle.load(open(f"{case}/dump.pkl", "rb"))
nro = rows[0]["data"].shape[-1]; nx = nro // 2
ref = [r for r in rows if r["flags"] & (1 << 19)]
Rr = np.stack([r["data"] for r in ref])
cc = np.linalg.svd(Rr.transpose(1, 0, 2).reshape(Rr.shape[1], -1), full_matrices=False)[0][:, :VC].conj().T.astype(np.complex64)
img = [r for r in rows if not r["flags"] & (1 << 19)]
# Wave k-space along (z, y), cycles/m: a waved line's trajectory minus its nominal phase encode.
w = img[0]["traj"]
wk = np.stack([w[:, 2] - (img[0]["e2"] - NZ // 2) / FOVZ, w[:, 1] - (img[0]["e1"] - NY // 2) / FOVY], -1) if w.size else None  # wave about the nominal encode: its y component has a non-zero mean
if wk is not None and os.environ.get("DELAY"):
    _t = np.arange(len(wk)); _dl = float(os.environ["DELAY"])
    wk = np.stack([np.interp(_t - _dl, _t, wk[:, a]) for a in range(2)], -1)
if wk is not None and os.environ.get("WSCALE"): wk = wk * float(os.environ["WSCALE"])
frames = max(r["con"] for r in rows) + 1
by = [[r for r in img if r["con"] == f] for f in range(frames)]; shots = max(len(b) for b in by)
pos = -np.ones((frames, shots, 2), np.int64); y = np.zeros((VC, frames, shots, nro), np.complex64)
for f, b in enumerate(by):
    for s_, r in enumerate(b):
        pos[f, s_] = (r["e2"], r["e1"]); y[:, f, s_] = cc @ r["data"]
def deos(d):
    x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(d, -1), axis=-1), -1)[..., nro // 2 - nx // 2: nro // 2 + nx // 2]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, -1), axis=-1), -1)
k = np.zeros((VC, NZ, NY, nx), np.complex64)
for r in ref: k[:, r["e2"], r["e1"]] = deos(cc @ r["data"])
del rows, img, by, Rr
maps = nlinv_estimate(torch.from_numpy(k), **sob).to(torch.complex64); del k
t_prep = time.perf_counter() - t0
if wk is None or os.environ.get("NOWAVE"):
    psf = torch.ones(NZ, NY, nro, dtype=torch.complex64)
else:
    psf = linop.wave_psf(sign * wk, (NZ, NY, nx), (FOVZ, FOVY), centred=True).to(torch.complex64)
basis = torch.load(bpath)[:, :rank].T.contiguous().to(torch.complex64)
A = linop.WaveSense(maps, (rank, NZ, NY, nx), nro, positions=torch.from_numpy(pos), basis=basis, centred=True, psf=psf)
if os.environ.get("REFSOLVE"):
    # The wave-free calibration trains join the solve, each line at its own echo, through an operator without the wave.
    rby = [[r for r in ref if r["con"] == f] for f in range(frames)]; rshots = max(1, max(len(b) for b in rby))
    rpos = -np.ones((frames, rshots, 2), np.int64); ry = np.zeros((VC, frames, rshots, nro), np.complex64)
    for f, b in enumerate(rby):
        for s_, r in enumerate(b): rpos[f, s_] = (r["e2"], r["e1"]); ry[:, f, s_] = cc @ r["data"]
    Aref = linop.WaveSense(maps, (rank, NZ, NY, nx), nro, positions=torch.from_numpy(rpos), basis=basis, centred=True, psf=torch.ones(NZ, NY, nro, dtype=torch.complex64))
    A = linop.concatenate([A, Aref], axis=2); y = np.concatenate([y, ry], axis=2)
    print(f"calibration trains in the solve: {sum(len(b) for b in rby)} lines")
d = torch.from_numpy(y); s = optim.data_scaling(d, A=A) or 1.0
t1 = time.perf_counter()
if lam == 0:
    a = optim.CG(1e-6, maxiter=iters)(d / s, A) * s
else:
    a = optim.FISTA(priors.LocallyLowRank((-3, -2) if os.environ.get("LLR2D") else (-3, -2, -1), lam, joint_axes=(0,), block=8), maxiter=iters, step=0.95 / float(optim.maxeigen(A)))(d / s, A) * s
el = time.perf_counter() - t1
res = float((A(a) - d).norm() / d.norm())
out = torch.einsum("kf,kzyx->fzyx", basis[:, KEEP], a).numpy()
print(f"wave3d {VC} VC rank {rank} lam {lam} {iters} it: prep {t_prep:.1f} s, solve {el:.1f} s, resid {res:.3f}, peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB")
np.savez(f"{case}/recon_{tag}.npz", echoes=out, frames=np.array(KEEP), seconds=el + t_prep)
