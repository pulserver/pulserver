"""Sampling comparison on one plane with signal-equation data and compressed maps: ours vs Tamir-style blocks. sampcmp.py"""
import os, numpy as np, torch
from bartorch import linop, optim, priors
from scipy.ndimage import binary_erosion
rng = np.random.default_rng(0)
z = np.load("h_fse48/plane64_vc8.npz"); maps = torch.from_numpy(z["maps"]).to(torch.complex64); ours = z["pos"]
ser = np.load("h_fse48/series_x64.npy").astype(np.complex64); T, nz, ny = ser.shape
B = torch.load(os.environ.get("BASIS", "h_fse/basis_stream.pt")).to(torch.complex64)
corr = np.maximum(np.load("rss_helmet.npy")[..., 64], 0.5)
KEEP = [0, 4, 9, 19, 29, 39]
shots = ours.shape[1]; total = int((ours[..., 0] >= 0).sum())

def tamir(blocks, vd=3.0, cal=12):
    M = T // blocks; per = shots * M
    zz, yy = np.meshgrid(np.arange(nz) - nz // 2, np.arange(ny) - ny // 2, indexing="ij")
    r = np.sqrt((zz / (nz / 2)) ** 2 + (yy / (ny / 2)) ** 2)
    ell = r <= 1.0
    w = (1 - np.clip(r, 0, 1)) ** vd + 0.02; w[~ell] = 0
    centre = (np.abs(zz) < cal // 2) & (np.abs(yy) < cal // 2)
    pos = -np.ones((T, shots, 2), np.int64)
    for b in range(blocks):
        idx = list(np.flatnonzero(centre.ravel()))
        rest = np.flatnonzero((~centre & ell).ravel()); p = w.ravel()[rest]; p = p / p.sum()
        idx += list(rng.choice(rest, per - len(idx), replace=False, p=p))
        idx = np.array(idx)[rng.permutation(per)].reshape(M, shots)
        for e in range(M):
            pos[b * M + e, :, 0], pos[b * M + e, :, 1] = np.unravel_index(idx[e], (nz, ny))
    return pos

def solve(pos, rank=3, lam=0.003, it=150):
    P = torch.from_numpy(pos); ones = torch.ones(nz, ny, 1, dtype=torch.complex64)
    Af = linop.WaveSense(maps, (T, nz, ny, 1), 1, positions=P, basis=torch.eye(T, dtype=torch.complex64), centred=True, psf=ones)
    d = Af(torch.from_numpy(ser)[..., None])
    basis = B[:, :rank].T.contiguous()
    A = linop.WaveSense(maps, (rank, nz, ny, 1), 1, positions=P, basis=basis, centred=True, psf=ones)
    s = optim.data_scaling(d, A=A) or 1.0
    a = optim.FISTA(priors.LocallyLowRank((-3, -2), lam, joint_axes=(0,), block=8), maxiter=it, step=0.95 / float(optim.maxeigen(A)))(d / s, A) * s
    R = np.abs(torch.einsum("kf,kzyx->fzyx", basis[:, KEEP], a).numpy()[..., 0]) / corr
    Tr = np.abs(ser[KEEP]) / corr
    m = binary_erosion(Tr.max(0) > 0.1 * Tr.max(), iterations=3)
    sc = (R[:, m] * Tr[:, m]).sum() / (R[:, m] ** 2).sum()
    return [round(float(np.linalg.norm((sc * R[j] - Tr[j])[m]) / np.linalg.norm(Tr[j][m])), 3) for j in range(len(KEEP))]

print("readouts", total, "per echo", shots)
print("ours (each line once)  ", solve(ours))
for b in (8, 4, 16):
    p = tamir(b); print(f"Tamir-style {b} blocks    ", solve(p))
