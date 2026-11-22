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
CALNORM = os.environ.get("CALNORM")
if CALNORM:
    # Each calibration line divided by the train's mean signal at its echo: a contrast-consistent k-space for the maps.
    if CALNORM == "sim":
        import pickle as _p
        _b = _p.load(open(f"{case}/simbasis.pkl", "rb"))[0]
        curve = np.abs(np.einsum("gak,kt->t", _b["coefficients"], _b["temporal"][:, -basis_full.shape[0]:]))
    elif CALNORM.endswith(".npy"):
        curve = np.load(CALNORM)
    else:
        curve = np.abs(basis_full[:, 0].numpy())
    curve = curve / curve.max()
_calrows = [r for r in flat if r["flags"] & (1 << 19)] if os.environ.get("REFMAPS") else [r for r in flat if not r["flags"] & (1 << 19)]
if os.environ.get("REFMAPS"): print(f"maps from {len(_calrows)} calibration-train lines")
for r in _calrows:
    w_ = 1.0 / curve[r["con"]] if CALNORM else 1.0
    grid[:, r["e2"], r["e1"]] += w_ * deos(comp(r["data"])); cnt[r["e2"], r["e1"]] += 1
grid /= np.maximum(cnt, 1)[None, :, :, None]
t = time.perf_counter()
if os.environ.get("SUBNLINV"):
    # Maps fitted jointly with the subspace coefficient images of the central lines (nlinv -B), each line at its own echo.
    from bartorch import tools
    from bartorch.apps import _calibration as _nm
    nb = int(os.environ.get("SUBNLINV")); sz = int(os.environ.get("CALSIZE", 24))
    Tn = basis_full.shape[0]
    cz, cy, cx = (min(sz, n) for n in (nz, ny, nx))
    oz, oy, ox = nz // 2 - cz // 2, ny // 2 - cy // 2, nx // 2 - cx // 2
    ks = np.zeros((Tn, 1, VC, cz, cy, cx), np.complex64); pat = np.zeros((Tn, 1, 1, cz, cy, cx), np.complex64)
    for r in flat:
        z_, y_ = r["e2"] - oz, r["e1"] - oy
        if 0 <= z_ < cz and 0 <= y_ < cy:
            ks[r["con"], 0, :, z_, y_] = deos(comp(r["data"]))[:, ox:ox + cx]; pat[r["con"], 0, 0, z_, y_] = 1
    bas = basis_full[:, :nb].T.contiguous().to(torch.complex64).reshape(nb, Tn, 1, 1, 1, 1, 1)
    opts = _nm._scaled_sobolev([cz, cy, cx], (nz, ny, nx), {})
    t_ = time.perf_counter()
    _, sens = tools.nlinv(torch.from_numpy(ks), pattern=torch.from_numpy(pat), basis=bas, maps=1, return_sensitivities=True, **opts)
    print("subspace nlinv", tuple(sens.shape), f"{time.perf_counter() - t_:.1f} s")
    maps = _nm._upsampled(sens.reshape(VC, cz, cy, cx), (nz, ny, nx))
else:
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
if os.environ.get("NOREFDATA"): imaging = [r for r in imaging if not r["flags"] & (1 << 19)]
by = [[r for r in imaging if r["con"] == f] for f in range(frames)]
shots = max(len(b) for b in by)
pos = -np.ones((frames, shots, 2), np.int64); order = -np.ones((frames, shots), np.int64); y = np.zeros((VC, frames, shots, nro), np.complex64)
for f, b in enumerate(by):
    for s_, r in enumerate(b):
        pos[f, s_] = (r["e2"], r["e1"]); y[:, f, s_] = comp(r["data"]); order[f, s_] = next(i for i, q in enumerate(rows) if q is r)
basis = basis_full[:, :rank].T.contiguous().to(torch.complex64)  # (coeffs, frames)


# One sagittal plane: the readout is fully sampled, so an inverse FFT along it decouples the planes.
X0 = int(os.environ.get("X0", nx // 2))
img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y, -1), axis=-1), -1)[..., nro // 2 - nx // 2 : nro // 2 + nx // 2]
y2 = np.ascontiguousarray(img[..., X0:X0 + 1]).astype(np.complex64)
m2 = maps[..., X0:X0 + 1].to(torch.complex64).contiguous()

# Hybrid-x solve: the readout is fully sampled, so after an inverse FFT along it every sagittal plane is its own problem.
rank, lam, iters = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
basis = basis_full[:, :rank].T.contiguous().to(torch.complex64)
KEEP = [int(v) for v in os.environ.get("KEEP", "0,4,9,19,29,39").split(",")]
img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y, -1), axis=-1), -1)[..., nro // 2 - nx // 2 : nro // 2 + nx // 2]
P = torch.from_numpy(pos); ones = torch.ones(nz, ny, 1, dtype=torch.complex64)
out = np.zeros((len(KEEP), nz, ny, nx), np.complex64)
t = time.perf_counter()
for x0 in range(nx):
    d = torch.from_numpy(np.ascontiguousarray(img[..., x0:x0 + 1]).astype(np.complex64))
    if float(d.abs().max()) == 0: continue
    A = linop.WaveSense(maps[..., x0:x0 + 1].to(torch.complex64).contiguous(), (rank, nz, ny, 1), 1, positions=P, basis=basis, centred=True, psf=ones)
    s_ = optim.data_scaling(d, A=A) or 1.0
    a = optim.CG(lam, maxiter=iters)(d / s_, A) * s_
    out[..., x0] = torch.einsum("kf,kzyx->fzyx", basis[:, KEEP], a).numpy()[..., 0]
el = time.perf_counter() - t
import resource
print(f"hybrid-x {nx} planes, rank {rank}, {iters} CG it: {el:.1f} s, peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB")
np.savez(f"{case}/recon_{sys.argv[7]}.npz", echoes=out, frames=np.array(KEEP), seconds=el)
