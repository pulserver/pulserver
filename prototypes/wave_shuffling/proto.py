"""Prototype wave-shuffling subspace recon from a label dump: proto.py CASE RANK LAMBDA ITERS [nowave]."""
import os
import pickle
import sys
import time

import numpy as np
import torch
import bartorch
from bartorch import linop, optim, priors
from pulserver.recon.handlers.pics import nlinv_estimate

case, rank, lam, iters = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rows = pickle.load(open(f"{case}/dump.pkl", "rb"))
for r_ in rows:
    if r_["traj"].size == 0:
        n_ = r_["data"].shape[-1]
        fx_, fy_, fz_ = (float(v) for v in os.environ.get("FOV", "0.32,0.16,0.032").split(","))
        r_["traj"] = np.stack([(np.arange(n_) - n_ // 2) / fx_, np.full(n_, (r_["e1"] - 32) / fy_), np.full(n_, (r_["e2"] - 24) / fz_)], -1)
basis_full = torch.load(sys.argv[5]) if len(sys.argv) > 5 else torch.load(f"{case}/basis.pt")
VC = int(os.environ.get("VC", 8))

flat = [r for r in rows if np.ptp(r["traj"][:, 1]) < 1e-3 and np.ptp(r["traj"][:, 2]) < 1e-3]
waved = [r for r in rows if r not in flat] if len(flat) < len(rows) else []
imaging = flat if "flat" in sys.argv else (waved if waved else flat)
if "flat" in sys.argv: waved = []
print(f"{len(rows)} readouts, {len(flat)} wave-free, {len(waved)} waved")

# Geometry from the trajectory (1/m): line spacing gives the FOV, sample spacing the oversampled readout FOV.
e1 = np.array([r["e1"] for r in rows]); e2 = np.array([r["e2"] for r in rows])
ky = np.array([r["traj"][:, 1].mean() for r in rows]); kz = np.array([r["traj"][:, 2].mean() for r in rows])
dky = np.polyfit(e1, ky, 1)[0]; dkz = np.polyfit(e2, kz, 1)[0]
kx = rows[0]["traj"][:, 0]; dkx = np.median(np.diff(kx)); nro = kx.size
ny, nz = e1.max() + 1, e2.max() + 1
fov_y, fov_z, fov_xos = 1 / abs(dky), 1 / abs(dkz), 1 / abs(dkx)
nx = nro // 2
print(f"grid z{nz} y{ny} x{nx} (readout {nro}), FOV y {fov_y*1e3:.0f} z {fov_z*1e3:.0f} xos {fov_xos*1e3:.0f} mm")

# Channel compression from the wave-free calibration lines.
calib = np.stack([r["data"] for r in flat])  # (lines, coils, ro)
u, s, _ = np.linalg.svd(calib.transpose(1, 0, 2).reshape(calib.shape[1], -1), full_matrices=False)
cc = u[:, :VC].conj().T.astype(np.complex64)
comp = lambda d: cc @ d

# Coil maps: nlinv on the wave-free calibration region, readout oversampling removed.
def deos(d):
    x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(d, -1), axis=-1), -1)
    x = x[..., nro // 2 - nx // 2 : nro // 2 + nx // 2]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, -1), axis=-1), -1)

grid = np.zeros((VC, nz, ny, nx), np.complex64); cnt = np.zeros((nz, ny))
for r in flat:
    grid[:, r["e2"], r["e1"]] += deos(comp(r["data"])); cnt[r["e2"], r["e1"]] += 1
grid /= np.maximum(cnt, 1)[None, :, :, None]
t = time.perf_counter()
maps = nlinv_estimate(torch.from_numpy(grid))
print(f"maps {tuple(maps.shape)} in {time.perf_counter()-t:.1f} s")

# Frames = echo index; positions (frames, shots, 2) of (z, y); samples (coils, frames, shots, readout).
frames = max(r["con"] for r in rows) + 1
by = [[r for r in imaging if r["con"] == f] for f in range(frames)]
shots = max(len(b) for b in by)
pos = -np.ones((frames, shots, 2), np.int64); order = -np.ones((frames, shots), np.int64); y = np.zeros((VC, frames, shots, nro), np.complex64)
for f, b in enumerate(by):
    for s_, r in enumerate(b):
        pos[f, s_] = (r["e2"], r["e1"]); y[:, f, s_] = comp(r["data"]); order[f, s_] = next(i for i, q in enumerate(rows) if q is r)
basis = basis_full[:, :rank].T.contiguous().to(torch.complex64)  # (coeffs, frames)

kwargs = dict(positions=torch.from_numpy(pos), basis=basis, centred=True)
if waved:
    w = waved[0]["traj"]; wy = w[:, 1] - w[:, 1].mean(); wz = w[:, 2] - w[:, 2].mean()
    yy = (np.arange(ny) - ny // 2) * fov_y / ny; zz = (np.arange(nz) - nz // 2) * fov_z / nz
    sign = float(sys.argv[6]) if len(sys.argv) > 6 else -1.0
    psf = np.exp(sign * 2j * np.pi * (wy[None, None, :] * yy[None, :, None] + wz[None, None, :] * zz[:, None, None]))
    kwargs["psf"] = torch.from_numpy(psf.astype(np.complex64))
else:
    kwargs["psf"] = torch.ones(nz, ny, nro, dtype=torch.complex64)
A = linop.WaveSense(maps.to(torch.complex64), (rank, nz, ny, nx), nro, **kwargs)
data = torch.from_numpy(y)
scaling = optim.data_scaling(data, A=A) or 1.0
t = time.perf_counter(); eig = optim.maxeigen(A); print(f"eig {float(eig):.3g} in {time.perf_counter()-t:.1f} s")
prior = priors.LocallyLowRank((-3, -2, -1), lam, joint_axes=(0,), block=8)
t = time.perf_counter()
alpha = optim.FISTA(prior, maxiter=iters, step=0.95 / float(eig))(data / scaling, A) * scaling
print(f"solve {iters} it in {time.perf_counter()-t:.1f} s")
res_ = (A(alpha) - data); residual = float(res_.norm() / data.norm()); print("residual", residual)
np.savez(f"{case}/resid.npz", r=res_.norm(dim=(0, 3)).numpy(), d=data.norm(dim=(0, 3)).numpy(), order=order)
echoes = torch.einsum("kf,kzyx->fzyx", basis, alpha)
np.savez(f"{case}/recon_{sys.argv[7] if len(sys.argv) > 7 else 'out'}.npz", alpha=alpha.numpy(), echoes=echoes.numpy())
