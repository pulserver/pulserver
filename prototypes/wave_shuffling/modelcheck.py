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
        r_["traj"] = np.stack([(np.arange(n_) - n_ // 2) / fx_, np.full(n_, (r_["e1"] - int(os.environ.get("NY", 64)) // 2) / fy_), np.full(n_, (r_["e2"] - int(os.environ.get("NZ", 48)) // 2) / fz_)], -1)
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
ny, nz = int(os.environ.get("NY", e1.max() + 1)), int(os.environ.get("NZ", e2.max() + 1))
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
if os.environ.get("TRUEMAPS"):
    from pulserver.virtual import COILS
    zz_, yy_, xx_ = [(np.arange(n_) - n_ // 2) * f_ / n_ for n_, f_ in ((nz, fov_z), (ny, fov_y), (nx, fov_xos / 2))]
    Z_, Y_, X_ = np.meshgrid(zz_, yy_, xx_, indexing="ij")
    sens = COILS["head8/head32"].receive(np.stack([X_.ravel(), Y_.ravel(), Z_.ravel()], 1))
    sens = (cc @ np.asarray(sens).T).reshape(VC, nz, ny, nx).astype(np.complex64)
    maps = torch.from_numpy(sens / np.sqrt((np.abs(sens) ** 2).sum(0)).max())
    print("true coil maps")
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

DROP = int(os.environ.get("DROP", 0))
if DROP:
    # Lines read in the first DROP echoes of each train are left out of the solve.
    pos[:DROP] = -1; y[:, :DROP] = 0
    print(f"left out the lines of echoes 1-{DROP}")
kwargs = dict(positions=torch.from_numpy(pos), basis=basis, centred=True)
if waved:
    flat_at = {(r["e1"], r["e2"]): r for r in flat}
    pair = next(r for r in waved if (r["e1"], r["e2"]) in flat_at)
    w = pair["traj"] - flat_at[(pair["e1"], pair["e2"])]["traj"]; wy = w[:, 1]; wz = w[:, 2]
    print("wave mean offset (lines) y", round(float(wy.mean() * fov_y), 2), "z", round(float(wz.mean() * fov_z), 2))
    yy = (np.arange(ny) - ny // 2) * fov_y / ny; zz = (np.arange(nz) - nz // 2) * fov_z / nz
    sign = float(sys.argv[6]) if len(sys.argv) > 6 else -1.0
    psf = np.exp(sign * 2j * np.pi * (wy[None, None, :] * yy[None, :, None] + wz[None, None, :] * zz[:, None, None]))
    if os.environ.get("CALIBRATE"):
        e0 = 0
        ref_ = np.zeros((VC, nz, ny, nro), np.complex64); wav_ = np.zeros_like(ref_)
        for r in flat:
            if r["con"] == e0: ref_[:, r["e2"], r["e1"]] = comp(r["data"])
        for r in waved:
            if r["con"] == e0: wav_[:, r["e2"], r["e1"]] = comp(r["data"])
        k_ = np.stack([wz, wy], -1)
        if os.environ.get("CALDUMP"):
            np.savez(os.environ["CALDUMP"], ref=ref_, wav=wav_, k=k_, fov=(fov_z, fov_y), dwell=waved[0].get("dwell", 0)); sys.exit(0)
        sc_, dl_ = linop.wave_calibrate(ref_, wav_, k_, (fov_z, fov_y))
        print("wave calibration scale", np.round(sc_, 4), "delay", np.round(dl_, 3))
        psf = linop.wave_psf(k_, (nz, ny, nx), (fov_z, fov_y), scale=sc_, delay=dl_, centred=True).numpy()
        print("psf vs nominal phase agreement", float(np.abs(np.vdot(psf, np.exp(sign * 2j * np.pi * (wy[None, None, :] * yy[None, :, None] + wz[None, None, :] * zz[:, None, None])))) / psf.size))
    if os.environ.get("SUBCAL"):
        # Fit wave scale and delay against the subspace model: a low-resolution image from the
        # wave-free calibration lines predicts the wave lines whose wave excursion stays in the region.
        from scipy.ndimage import binary_erosion
        from scipy.optimize import least_squares
        t0 = time.perf_counter()
        def packed(rs):
            per = [[r for r in rs if r["con"] == f] for f in range(frames)]
            sh = max(1, max(len(b) for b in per))
            p_ = -np.ones((frames, sh, 2), np.int64); d_ = np.zeros((VC, frames, sh, nro), np.complex64)
            for f, b in enumerate(per):
                for i, r in enumerate(b):
                    p_[f, i] = (r["e2"], r["e1"]); d_[:, f, i] = comp(r["data"])
            return torch.from_numpy(p_), torch.from_numpy(d_)
        k_ = np.stack([wz, wy], -1)
        region = np.zeros((nz, ny), bool)
        for r in flat: region[r["e2"], r["e1"]] = True
        margin = [int(np.ceil(np.abs(k_[:, a]).max() * (fov_z, fov_y)[a])) for a in range(2)]
        inner = binary_erosion(region, np.ones([2 * m + 1 for m in margin], bool), border_value=0)
        targets = [r for r in waved if inner[r["e2"], r["e1"]]]
        pf, yf = packed(flat); pt, yt = packed(targets)
        ones = torch.ones(nz, ny, nro, dtype=torch.complex64)
        crank = int(os.environ.get("SUBCAL_RANK", rank)); cbasis = basis[:crank].contiguous()
        Af = linop.WaveSense(maps.to(torch.complex64), (crank, nz, ny, nx), nro, positions=pf, basis=cbasis, centred=True, psf=ones)
        sf = optim.data_scaling(yf, A=Af) or 1.0
        xc = optim.CG(float(os.environ.get('SUBCAL_LAM', 1e-3)), maxiter=int(os.environ.get('SUBCAL_IT', 30)))(yf / sf, Af) * sf
        print("calibration-image misfit on the wave-free lines", float(torch.linalg.vector_norm(Af(xc) - yf) / torch.linalg.vector_norm(yf)))
        norm = float(torch.linalg.vector_norm(yt))
        def resid(p):
            ps = linop.wave_psf(k_, (nz, ny, nx), (fov_z, fov_y), scale=tuple(p[:2]), delay=tuple(p[2:]), centred=True)
            At = linop.WaveSense(maps.to(torch.complex64), (crank, nz, ny, nx), nro, positions=pt, basis=cbasis, centred=True, psf=ps.to(torch.complex64))
            r = ((At(xc) - yt) / norm).numpy().ravel()
            return np.concatenate([r.real, r.imag])
        r0 = float(np.linalg.norm(resid(np.r_[1.0, 1.0, 0.0, 0.0])))
        fit = least_squares(resid, np.r_[1.0, 1.0, 0.0, 0.0], diff_step=1e-3)
        sc_, dl_ = tuple(fit.x[:2]), tuple(fit.x[2:])
        print(f"subspace calibration: {len(targets)} wave lines, margin {margin}, misfit {r0:.4f} -> {float(np.linalg.norm(fit.fun)):.4f}, scale {np.round(sc_, 4)} delay {np.round(dl_, 3)} in {time.perf_counter() - t0:.1f} s ({fit.nfev} evaluations)")
        psf = linop.wave_psf(k_, (nz, ny, nx), (fov_z, fov_y), scale=sc_, delay=dl_, centred=True).numpy()
    if os.environ.get("PSFDELAY"):
        dl_ = tuple(float(v) for v in os.environ["PSFDELAY"].split(",")); sc_ = tuple(float(v) for v in os.environ.get("PSFSCALE", "1,1").split(","))
        psf = linop.wave_psf(np.stack([wz, wy], -1), (nz, ny, nx), (fov_z, fov_y), scale=sc_, delay=dl_, centred=True).numpy()
        print("forced psf scale", sc_, "delay", dl_)
    kwargs["psf"] = torch.from_numpy(psf.astype(np.complex64))
else:
    kwargs["psf"] = torch.ones(nz, ny, nro, dtype=torch.complex64)
if waved and os.environ.get("JOINT"):
    # Fit wave scale and delay to all wave lines against an image reconstructed with the nominal wave.
    from scipy.optimize import least_squares
    t0 = time.perf_counter(); k_ = np.stack([wz, wy], -1); m64 = maps.to(torch.complex64)
    yd = torch.from_numpy(y); norm = float(torch.linalg.vector_norm(yd))
    A0 = linop.WaveSense(m64, (rank, nz, ny, nx), nro, **kwargs)
    s0 = optim.data_scaling(yd, A=A0) or 1.0
    x0 = optim.FISTA(priors.LocallyLowRank((-3, -2, -1), lam, joint_axes=(0,), block=8), maxiter=int(os.environ.get("JOINT_IT", 30)), eigen=True)(yd / s0, A0) * s0
    def resid(p):
        ps = linop.wave_psf(k_, (nz, ny, nx), (fov_z, fov_y), scale=tuple(p[:2]), delay=tuple(p[2:]), centred=True).to(torch.complex64)
        At = linop.WaveSense(m64, (rank, nz, ny, nx), nro, positions=kwargs["positions"], basis=basis, centred=True, psf=ps)
        r = ((At(x0) - yd) / norm).numpy().ravel()
        return np.concatenate([r.real, r.imag])
    r0 = float(np.linalg.norm(resid(np.r_[1.0, 1.0, 0.0, 0.0])))
    fit = least_squares(resid, np.r_[1.0, 1.0, 0.0, 0.0], diff_step=1e-3)
    sc_, dl_ = tuple(fit.x[:2]), tuple(fit.x[2:])
    print(f"joint calibration: misfit {r0:.4f} -> {float(np.linalg.norm(fit.fun)):.4f}, scale {np.round(sc_, 4)} delay {np.round(dl_, 3)} in {time.perf_counter() - t0:.1f} s ({fit.nfev} evaluations)")
    kwargs["psf"] = linop.wave_psf(k_, (nz, ny, nx), (fov_z, fov_y), scale=sc_, delay=dl_, centred=True).to(torch.complex64)

A = linop.WaveSense(maps.to(torch.complex64), (rank, nz, ny, nx), nro, **kwargs)
T = np.load(f"{case}/truth.npz"); E = torch.from_numpy(T["echoes"]).to(torch.complex64); K = list(T["keep"])
B = basis[:, K]  # (rank, 6)
alpha = torch.einsum("kf,fzyx->kzyx", torch.linalg.pinv(B.T), E)
print("truth fit by basis", float(torch.linalg.vector_norm(torch.einsum("kf,kzyx->fzyx", B, alpha) - E) / torch.linalg.vector_norm(E)))
np.savez(f"{case}/alpha_true_r{rank}.npz", alpha=alpha.numpy()); sim = A(alpha); data = torch.from_numpy(y)
ns = sim.norm(dim=(0, 3)).numpy(); nd = data.norm(dim=(0, 3)).numpy()
ok = pos[..., 0] >= 0
s_ = (ns[ok] * nd[ok]).sum() / (ns[ok] ** 2).sum()
r = nd / np.maximum(s_ * ns, 1e-30)
for f in range(0, frames, 8):
    m = ok[f] & (ns[f] > np.percentile(ns[ok], 70))
    print(f, round(float(np.median(r[f][m])), 3))
