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
_calrows = [r for r in flat if r["flags"] & (1 << 19)] if os.environ.get("REFMAPS") else flat
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
    maps = nlinv_estimate(torch.from_numpy(grid), **({"sobolev": float(os.environ["SOBOLEV"])} if os.environ.get("SOBOLEV") else {}))
if os.environ.get("TRUEMAPS"):
    from pulserver.virtual import COILS
    zz_, yy_, xx_ = [(np.arange(n_) - n_ // 2) * f_ / n_ for n_, f_ in ((nz, fov_z), (ny, fov_y), (nx, fov_xos / 2))]
    Z_, Y_, X_ = np.meshgrid(zz_, yy_, xx_, indexing="ij")
    sens = COILS["head8/head32"].receive(np.stack([X_.ravel(), Y_.ravel(), Z_.ravel()], 1))
    full = np.asarray(sens).T
    rss_full = np.sqrt((np.abs(full) ** 2).sum(0)).reshape(1, nz, ny, nx)
    sens = (cc @ full).reshape(VC, nz, ny, nx).astype(np.complex64)
    norm_ = rss_full if os.environ.get("TRUEMAPS") == "full" else np.sqrt((np.abs(sens) ** 2).sum(0, keepdims=True))
    maps = torch.from_numpy((sens / np.maximum(norm_, 1e-6)).astype(np.complex64))
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
full_basis = basis_full
X0 = int(os.environ.get("X0", nx // 2))
def _plane_data(maps_):
    img_ = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y, -1), axis=-1), -1)[..., nro // 2 - nx // 2 : nro // 2 + nx // 2]
    return np.ascontiguousarray(img_[..., X0:X0 + 1]).astype(np.complex64), maps_[..., X0:X0 + 1].to(torch.complex64).contiguous()
for _pass in range(int(os.environ.get("REFINE", 0))):
    # The mean echo-train signal of a first reconstruction weights the calibration lines; the maps are fitted again.
    y2_, m2_ = _plane_data(maps)
    b_ = basis_full[:, :3].T.contiguous().to(torch.complex64)
    A_ = linop.WaveSense(m2_, (3, nz, ny, 1), 1, positions=torch.from_numpy(pos), basis=b_, centred=True, psf=torch.ones(nz, ny, 1, dtype=torch.complex64))
    d_ = torch.from_numpy(y2_); s_ = optim.data_scaling(d_, A=A_) or 1.0
    a_ = optim.CG(1e-3, maxiter=200)(d_ / s_, A_) * s_
    sig_ = torch.einsum("kf,kzyx->fzyx", b_, a_)
    e_ = sig_.abs().sum(0); m_ = e_ > 0.3 * e_.max()
    curve = (sig_.abs()[:, m_].mean(1)).numpy(); curve = curve / curve.max()
    print("refined curve", np.round(curve[[0, 4, 9, 19, 39, 79, 127]], 2))
    grid[:] = 0
    for r in flat:
        grid[:, r["e2"], r["e1"]] += deos(comp(r["data"])) / curve[r["con"]]
    grid /= np.maximum(cnt, 1)[None, :, :, None]
    maps = nlinv_estimate(torch.from_numpy(grid), **({"sobolev": float(os.environ["SOBOLEV"])} if os.environ.get("SOBOLEV") else {}))
m2 = maps[..., X0:X0 + 1].to(torch.complex64).contiguous()
if os.environ.get("SAVEMAPS"): np.save(os.environ["SAVEMAPS"], m2.numpy())
if os.environ.get("PLANEMAPS"):
    # Compression first: this plane's coil combination from its calibration lines, then 2D nlinv on the compressed lines.
    _refrows = [r for r in flat if r["flags"] & (1 << 19)]
    _ri = np.stack([comp(r["data"]) for r in _refrows])
    _ri = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(_ri, -1), axis=-1), -1)[..., nro // 2 - nx // 2 + X0]  # (lines, VC)
    _u = np.linalg.svd(_ri.T, full_matrices=False)[0]
    PCC = torch.from_numpy(_u[:, :int(os.environ["PLANEMAPS"])].conj().T.astype(np.complex64))
    _k = np.zeros((PCC.shape[0], nz, ny, 1), np.complex64)
    for r, v in zip(_refrows, _ri): _k[:, r["e2"], r["e1"], 0] = PCC.numpy() @ v
    m2 = nlinv_estimate(torch.from_numpy(_k), **({"sobolev": float(os.environ["SOBOLEV"])} if os.environ.get("SOBOLEV") else {})).to(torch.complex64).contiguous()
    print("plane maps after compression", tuple(m2.shape))
    if os.environ.get("SAVEPLANE"): np.savez(os.environ["SAVEPLANE"], maps=m2.numpy(), pos=pos)
if os.environ.get("SYNTHREF"):
    # Maps fitted on one plane to signal-equation calibration lines: each at its own echo ("actual") or all at one echo.
    _ser = np.load(os.environ["SYNTHFULL"]).astype(np.complex64)
    _mt = np.load(os.environ["SYNTHMAPS"])[..., 0]  # (VC, nz, ny)
    _box = [int(v) for v in os.environ.get("REFBOX", "999,999").split(",")]
    _k = np.zeros((VC, nz, ny, 1), np.complex64)
    for r in [r for r in flat if r["flags"] & (1 << 19)]:
        if abs(r["e1"] - ny // 2) * 2 >= _box[0] or abs(r["e2"] - nz // 2) * 2 >= _box[1]: continue
        t_ = r["con"] if os.environ["SYNTHREF"] == "actual" else int(os.environ["SYNTHREF"])
        _k[:, r["e2"], r["e1"], 0] = np.nan  # mark
    _sel = np.isnan(_k[0, :, :, 0].real)
    _k[:] = 0
    _con = {}
    for r in [r for r in flat if r["flags"] & (1 << 19)]: _con[(r["e2"], r["e1"])] = r["con"]
    _cache = {}
    for (z_, y_) in zip(*np.nonzero(_sel)):
        t_ = _con[(z_, y_)] if os.environ["SYNTHREF"] == "actual" else int(os.environ["SYNTHREF"])
        if t_ not in _cache: _cache[t_] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(_mt * _ser[t_][None], (1, 2)), norm="ortho"), (1, 2))
        _k[:, z_, y_, 0] = _cache[t_][:, z_, y_]
    m2 = nlinv_estimate(torch.from_numpy(_k), **({"sobolev": float(os.environ["SOBOLEV"])} if os.environ.get("SOBOLEV") else {})).to(torch.complex64).contiguous()
    print(f"plane maps from {int(_sel.sum())} signal-equation calibration lines ({os.environ['SYNTHREF']})", tuple(m2.shape))
for cfg in os.environ.get("CONFIGS", f"{os.environ.get('SOLVER','fista')}:{rank}:{lam}:{iters}").split(";"):
    solver, rank, lam, iters = cfg.split(":"); rank, lam, iters = int(rank), float(lam), int(iters)
    tag = f"{solver}_{rank}_{lam}_{iters}"
    basis = full_basis[:, :rank].T.contiguous().to(torch.complex64)
    A = linop.WaveSense(m2, (rank, nz, ny, 1), 1, positions=torch.from_numpy(pos), basis=basis, centred=True, psf=torch.ones(nz, ny, 1, dtype=torch.complex64))
    data = torch.from_numpy(y2)
    if os.environ.get("PLANEMAPS"):
        data = torch.einsum("vc,c...->v...", PCC, data).contiguous()
        A = linop.WaveSense(m2, (rank, nz, ny, 1), 1, positions=torch.from_numpy(pos), basis=basis, centred=True, psf=torch.ones(nz, ny, 1, dtype=torch.complex64))
    if os.environ.get("SYNTHFULL"):
        _ser = torch.from_numpy(np.load(os.environ["SYNTHFULL"]).astype(np.complex64))[..., None]
        _fr = _ser.shape[0]
        _mfull = torch.from_numpy(np.load(os.environ["SYNTHMAPS"])).to(torch.complex64) if os.environ.get("SYNTHMAPS") else m2
        Afull = linop.WaveSense(_mfull, (_fr, nz, ny, 1), 1, positions=torch.from_numpy(pos), basis=torch.eye(_fr, dtype=torch.complex64), centred=True, psf=torch.ones(nz, ny, 1, dtype=torch.complex64))
        data = Afull(_ser); print("signal-equation data from the full series")
    if os.environ.get("SYNTH"):
        a0 = torch.from_numpy(np.load(os.environ["SYNTH"])["alpha"][..., X0:X0 + 1]); data = A(a0); print("synthetic")
    if os.environ.get("GCCX"):
        _u = np.linalg.svd(data.reshape(data.shape[0], -1).numpy(), full_matrices=False)[0]
        _cc = torch.from_numpy(_u[:, :int(os.environ["GCCX"])].conj().T.astype(np.complex64))
        data = torch.einsum("vc,c...->v...", _cc, data).contiguous()
        A = linop.WaveSense(torch.einsum("vc,c...->v...", _cc, m2).contiguous(), (rank, nz, ny, 1), 1, positions=torch.from_numpy(pos), basis=basis, centred=True, psf=torch.ones(nz, ny, 1, dtype=torch.complex64))
    scaling = optim.data_scaling(data, A=A) or 1.0
    eig = optim.maxeigen(A)
    blk = int(os.environ.get("BLOCK", 8))
    prior = priors.LocallyLowRank((-3, -2), lam, joint_axes=(0,), block=blk)
    t = time.perf_counter()
    if solver == "cg":
        alpha = optim.CG(lam, maxiter=iters)(data / scaling, A) * scaling
    elif solver == "admm":
        alpha = optim.ADMM(prior, maxiter=iters, rho=float(os.environ.get("RHO", 0.1)), cg_maxiter=10)(data / scaling, A) * scaling
    else:
        alpha = optim.FISTA(prior, maxiter=iters, step=0.95 / float(eig))(data / scaling, A) * scaling
    res = float((A(alpha) - data).norm() / data.norm())
    T = np.load(os.environ.get("TRUTHFILE", f"{case}/truth.npz")); K = list(T["keep"]); Tr = T["echoes"][..., X0]
    if os.environ.get("SYNTHFULL"): Tr = np.abs(np.load(os.environ["SYNTHFULL"]))[K]
    if os.environ.get("TRUTHWEIGHT"):
        from pulserver.virtual import COILS as _C
        _zz, _yy = [(np.arange(n_) - n_ // 2) * f_ / n_ for n_, f_ in ((nz, fov_z), (ny, fov_y))]
        _Z, _Y = np.meshgrid(_zz, _yy, indexing="ij"); _x = (X0 - nx // 2) * (fov_xos / 2) / nx
        _s = np.asarray(_C["head8/head32"].receive(np.stack([np.full(_Z.size, _x), _Y.ravel(), _Z.ravel()], 1))).T
        _cc2 = (PCC.numpy() @ cc) if os.environ.get("PLANEMAPS") else cc; _w = np.sqrt((np.abs(_cc2 @ _s) ** 2).sum(0)) / np.maximum(np.sqrt((np.abs(_s) ** 2).sum(0)), 1e-12)
        Tr = Tr * _w.reshape(1, nz, ny)
    R = np.abs(torch.einsum("kf,kzyx->fzyx", basis[:, K], alpha).numpy()[..., 0])
    if os.environ.get("CORR"):
        _c = np.maximum(np.load(os.environ["CORR"])[..., X0], 0.5); R = R / _c; Tr = Tr / _c
    from scipy.ndimage import binary_erosion
    m = binary_erosion(Tr.max(0) > 0.1 * Tr.max(), iterations=3)
    sc = (R[:, m] * Tr[:, m]).sum() / (R[:, m] ** 2).sum()
    err = [round(float(np.linalg.norm((sc * R[j] - Tr[j])[m]) / np.linalg.norm(Tr[j][m])), 3) for j in range(len(K))]
    print(f"{solver} VC{VC} r{rank} lam{lam} it{iters} blk{blk}: {time.perf_counter()-t:.1f} s resid {res:.3f} nrmse {err}")
    np.savez(f"{case}/plane_{tag}.npz", echoes=sc * R, truth=Tr, frames=np.array(K))
