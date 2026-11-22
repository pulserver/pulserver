"""Where measured plane data and the signal equation disagree: residual.py CASE SERIES X0"""
import os, pickle, sys, numpy as np, torch
from bartorch import linop
from pulserver.virtual import COILS
case, series_f, X0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = pickle.load(open(f"{case}/dump.pkl", "rb"))
for r_ in rows:
    if r_["traj"].size == 0:
        n_ = r_["data"].shape[-1]; fx_, fy_, fz_ = 0.512, 0.28, 0.24
        r_["traj"] = np.stack([(np.arange(n_) - n_ // 2) / fx_, np.full(n_, (r_["e1"] - 70) / fy_), np.full(n_, (r_["e2"] - 60) / fz_)], -1)
flat = [r for r in rows if np.ptp(r["traj"][:, 1]) < 1e-3 and np.ptp(r["traj"][:, 2]) < 1e-3]
e1 = np.array([r["e1"] for r in rows]); e2 = np.array([r["e2"] for r in rows])
ky = np.array([r["traj"][:, 1].mean() for r in rows]); kz = np.array([r["traj"][:, 2].mean() for r in rows])
dky = np.polyfit(e1, ky, 1)[0]; dkz = np.polyfit(e2, kz, 1)[0]
kx = rows[0]["traj"][:, 0]; nro = kx.size; nx = nro // 2; fov_xos = 1 / abs(np.median(np.diff(kx)))
ny, nz = e1.max() + 1, e2.max() + 1; fov_y, fov_z = 1 / abs(dky), 1 / abs(dkz)
print("grid", nz, ny, nx, "fov", fov_z, fov_y, fov_xos / 2, "ky0", ky[0] - (e1[0] - ny // 2) * dky)
VC = int(os.environ.get("VC", 8))
calib = np.stack([r["data"] for r in flat])
u = np.linalg.svd(calib.transpose(1, 0, 2).reshape(calib.shape[1], -1), full_matrices=False)[0]
cc = u[:, :VC].conj().T.astype(np.complex64)
if os.environ.get("PLANECC"):
    _ift = lambda d: np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(d, -1), axis=-1), -1)[..., nro // 2 - nx // 2 + X0]
    _src = [r for r in rows if r["flags"] & (1 << 19)] if os.environ["PLANECC"] == "ref" else rows
    _R = np.stack([_ift(r["data"]) for r in _src])  # (lines, coils)
    cc = np.linalg.svd(_R.T, full_matrices=False)[0][:, :VC].conj().T.astype(np.complex64)
    print("plane compression from", len(_src), "lines")
img_rows = [r for r in rows if not r["flags"] & (1 << 19)] if os.environ.get("NOREF") else rows
frames = max(r["con"] for r in img_rows) + 1
by = [[r for r in img_rows if r["con"] == f] for f in range(frames)]
shots = max(len(b) for b in by)
pos = -np.ones((frames, shots, 2), np.int64); y = np.zeros((VC, frames, shots, nro), np.complex64); isref = np.zeros((frames, shots), bool)
for f, b in enumerate(by):
    for s_, r in enumerate(b):
        pos[f, s_] = (r["e2"], r["e1"]); y[:, f, s_] = cc @ r["data"]; isref[f, s_] = bool(r["flags"] & (1 << 19))
img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y, -1), axis=-1), -1)[..., nro // 2 - nx // 2 + X0]
meas = torch.from_numpy(np.ascontiguousarray(img[..., None]).astype(np.complex64))
zz, yy = [(np.arange(n) - n // 2) * f / n for n, f in ((nz, fov_z), (ny, fov_y))]
_o = [float(v) * 1e-3 for v in os.environ.get("OFF", "0,0,0").split(",")]
Z, Y = np.meshgrid(zz + _o[2], yy + _o[1], indexing="ij"); xpos = (X0 - nx // 2) * (fov_xos / 2) / nx + _o[0]
s32 = np.asarray(COILS["head8/head32"].receive(np.stack([np.full(Z.size, xpos), Y.ravel(), Z.ravel()], 1))).T
rss32 = np.sqrt((np.abs(s32) ** 2).sum(0)).reshape(nz, ny)
maps = torch.from_numpy((cc @ s32).reshape(VC, nz, ny, 1).astype(np.complex64))
ser = np.load(series_f).astype(np.complex64)  # weighted by rss32
ser = ser if os.environ.get("BL") else ser / np.maximum(rss32, 1e-9)[None]
A = linop.WaveSense(maps, (frames, nz, ny, 1), 1, positions=torch.from_numpy(pos), basis=torch.eye(frames, dtype=torch.complex64), centred=True, psf=torch.ones(nz, ny, 1, dtype=torch.complex64))
model = A(torch.from_numpy(ser[..., None])).numpy(); m = meas.numpy()
valid = pos[..., 0] >= 0
def fit(a, b):
    s = (a.conj() * b).sum() / (np.abs(a) ** 2).sum(); return s, np.linalg.norm(s * a - b) / np.linalg.norm(b)
s, r = fit(model[:, valid], m[:, valid]); print(f"global scale {abs(s):.3g} phase {np.angle(s):.2f}  rel residual {r:.3f}")
res = s * model - m
# by echo
E = [np.linalg.norm(res[:, f][:, valid[f]]) / np.linalg.norm(m[:, f][:, valid[f]]) for f in range(frames)]
print("by echo (1,2,4,8,16,32,64,128):", [round(float(E[i]), 3) for i in (0, 1, 3, 7, 15, 31, 63, 127) if i < frames])
# best per-echo scale/phase
sc = [fit(model[:, f][:, valid[f]], m[:, f][:, valid[f]]) for f in range(frames)]
print("per-echo fitted |s| rel:", [round(float(abs(sc[i][0]) / abs(s)), 3) for i in (0, 1, 3, 7, 15, 31, 63, 127)], " phase:", [round(float(np.angle(sc[i][0] / s)), 2) for i in (0, 1, 3, 7, 15, 31, 63, 127)])
print("per-echo residual after own scale:", [round(float(sc[i][1]), 3) for i in (0, 1, 3, 7, 15, 31, 63, 127)])
# by k radius
rad = np.sqrt(((pos[..., 1] - ny // 2) / ny) ** 2 + ((pos[..., 0] - nz // 2) / nz) ** 2)
edges = [0, .02, .05, .1, .2, .3, .45, .8]
for a_, b_ in zip(edges[:-1], edges[1:]):
    sel = valid & (rad >= a_) & (rad < b_)
    if sel.any(): print(f"|k| {a_:.2f}-{b_:.2f}: n={sel.sum():5d} residual {np.linalg.norm(res[:, sel]) / np.linalg.norm(m[:, sel]):.3f}  energy share {np.linalg.norm(res[:, sel])**2 / np.linalg.norm(res[:, valid])**2:.3f}")
print("ref trains residual", np.linalg.norm(res[:, isref & valid]) / np.linalg.norm(m[:, isref & valid]))
np.savez(f"{case}/residual_x{X0}.npz", res=res, meas=m, model=s * model, pos=pos)
