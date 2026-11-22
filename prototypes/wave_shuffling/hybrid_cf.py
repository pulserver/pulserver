"""Compression-first hybrid-x subspace recon: hybrid_cf.py CASE BASIS.pt RANK LAM ITERS TAG

Each readout is inverse-Fourier transformed along x and compressed per x position to VC virtual
coils (SVD of that position's calibration-train lines) as it is read; maps come from a 2D nlinv
on each sagittal plane's compressed calibration lines; each plane is solved with LLR-FISTA.
"""
import os, pickle, resource, sys, time
import numpy as np, torch
from bartorch import linop, optim, priors
from pulserver.recon.handlers.pics import nlinv_estimate

case, bpath, rank, lam, iters, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
VC = int(os.environ.get("VC", 6)); NY, NZ = int(os.environ.get("NY", 140)), int(os.environ.get("NZ", 120))
KEEP = [int(v) for v in os.environ.get("KEEP", "0,4,9,19,29,39").split(",")]
sob = {"sobolev": float(os.environ["SOBOLEV"])} if os.environ.get("SOBOLEV") else {}
t0 = time.perf_counter()
rows = pickle.load(open(f"{case}/dump.pkl", "rb"))
nro = rows[0]["data"].shape[-1]; nx = nro // 2
ift = lambda d: np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(d, -1), axis=-1), -1)[..., nro // 2 - nx // 2: nro // 2 + nx // 2]
ref = [r for r in rows if r["flags"] & (1 << 19)]
R = np.stack([ift(r["data"]) for r in ref])                      # (lines, coils, x): calibration only
cc = np.stack([np.linalg.svd(R[:, :, x].T, full_matrices=False)[0][:, :VC].conj().T for x in range(nx)]).astype(np.complex64)  # (x, VC, coils)
if os.environ.get("ALIGN"):
    # Align each position's virtual coils to its neighbour's (unitary Procrustes), from the centre out,
    # so the virtual coils vary smoothly along the readout.
    for x in list(range(nx // 2 + 1, nx)) + list(range(nx // 2 - 1, -1, -1)):
        nb = x - 1 if x > nx // 2 else x + 1
        u_, _, vh_ = np.linalg.svd(cc[nb] @ cc[x].conj().T)
        cc[x] = (u_ @ vh_) @ cc[x]
img = [r for r in rows if not r["flags"] & (1 << 19)]
frames = max(r["con"] for r in rows) + 1
by = [[r for r in img if r["con"] == f] for f in range(frames)]; shots = max(len(b) for b in by)
pos = -np.ones((frames, shots, 2), np.int64); y = np.zeros((nx, VC, frames, shots), np.complex64)
for f, b in enumerate(by):
    for s_, r in enumerate(b):
        pos[f, s_] = (r["e2"], r["e1"]); y[:, :, f, s_] = np.einsum("xvc,cx->xv", cc, ift(r["data"]))  # compressed on arrival
del rows, img, by
Rc = np.einsum("xvc,lcx->xvl", cc, R).astype(np.complex64); del R
if os.environ.get("MAPS3D"):
    # One 3D nlinv on the compressed calibration region, back in k-space along the readout.
    k3 = np.zeros((VC, NZ, NY, nx), np.complex64)
    for i, r in enumerate(ref): k3[:, r["e2"], r["e1"], :] = Rc[:, :, i].T
    k3 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(k3, -1), axis=-1), -1)
    MAPS = nlinv_estimate(torch.from_numpy(k3), **sob).to(torch.complex64)
t_prep = time.perf_counter() - t0
basis = torch.load(bpath)[:, :rank].T.contiguous().to(torch.complex64)
P = torch.from_numpy(pos); ones = torch.ones(NZ, NY, 1, dtype=torch.complex64)
out = np.zeros((len(KEEP), NZ, NY, nx), np.complex64)
t1 = time.perf_counter()
XR = range(*[int(v) for v in os.environ["XR"].split(":")]) if os.environ.get("XR") else range(nx)
for x in XR:
    d = torch.from_numpy(np.ascontiguousarray(y[x][..., None]))
    if float(d.abs().max()) == 0: continue
    k = np.zeros((VC, NZ, NY, 1), np.complex64)
    for i, r in enumerate(ref): k[:, r["e2"], r["e1"], 0] = Rc[x, :, i]
    maps = MAPS[..., x:x + 1].contiguous() if os.environ.get("MAPS3D") else nlinv_estimate(torch.from_numpy(k), **sob).to(torch.complex64).contiguous()
    A = linop.WaveSense(maps, (rank, NZ, NY, 1), 1, positions=P, basis=basis, centred=True, psf=ones)
    s = optim.data_scaling(d, A=A) or 1.0
    prior = priors.LocallyLowRank((-3, -2), lam, joint_axes=(0,), block=8)
    a = optim.FISTA(prior, maxiter=iters, step=0.95 / float(optim.maxeigen(A)))(d / s, A) * s
    out[..., x] = torch.einsum("kf,kzyx->fzyx", basis[:, KEEP], a).numpy()[..., 0]
el = time.perf_counter() - t1
print(f"compression-first hybrid-x: {VC} VC, rank {rank}, {iters} it: prep {t_prep:.1f} s, solve {el:.1f} s, peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB")
np.savez(f"{case}/recon_{tag}.npz", echoes=out, frames=np.array(KEEP), seconds=el + t_prep)
