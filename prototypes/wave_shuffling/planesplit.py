"""Split one plane's error: real data vs signal-equation data, true compressed maps, corrected reference.
planesplit.py CASE SERIES X0 BASIS  (env VC, RANK, LAM, ITERS, OFF, ETL)"""
import os, sys, numpy as np, torch
from bartorch import linop, optim, priors
exec(open("residual.py").read().split("A = linop.WaveSense")[0].replace("case, series_f, X0 = sys.argv[1], sys.argv[2], int(sys.argv[3])", "case, series_f, X0 = sys.argv[1], sys.argv[2], int(sys.argv[3])"))
basis_full = torch.load(sys.argv[4]); rank = int(os.environ.get("RANK", 3)); lam = float(os.environ.get("LAM", 0.003)); iters = int(os.environ.get("ITERS", 40))
basis = basis_full[:, :rank].T.contiguous().to(torch.complex64)
psf = torch.ones(nz, ny, 1, dtype=torch.complex64); P = torch.from_numpy(pos)
Afull = linop.WaveSense(maps, (frames, nz, ny, 1), 1, positions=P, basis=torch.eye(frames, dtype=torch.complex64), centred=True, psf=psf)
synth = Afull(torch.from_numpy(ser[..., None]))
rssc = maps.abs().pow(2).sum(0).sqrt()  # (nz, ny, 1)
if os.environ.get("NORMMAPS"):
    ser_w = ser * rssc[..., 0].numpy()[None]
    maps_r = maps / rssc.clamp_min(1e-12)
else:
    ser_w = ser; maps_r = maps
if os.environ.get("NLMAPS"):
    from pulserver.recon.handlers.pics import nlinv_estimate
    _k = np.zeros((VC, nz, ny, 1), np.complex64)
    for f in range(frames):
        for s_ in range(shots):
            if isref[f, s_]: _k[:, pos[f, s_, 0], pos[f, s_, 1], 0] = meas.numpy()[:, f, s_, 0]
    _kw = {"sobolev": float(os.environ["SOBOLEV"])} if os.environ.get("SOBOLEV") else {}
    nm = nlinv_estimate(torch.from_numpy(_k), **_kw).to(torch.complex64).contiguous()
    coh = (nm.conj() * maps_r).sum(0).abs() / (nm.abs().pow(2).sum(0).sqrt() * maps_r.abs().pow(2).sum(0).sqrt() + 1e-12)
    print("nlinv maps from", int(isref.sum()), "calibration lines; coherence with the true maps in head", round(float(coh[torch.from_numpy(np.abs(ser[0]) > 0.1 * np.abs(ser[0]).max())[..., None]].mean()), 4))
    maps_r = nm
if os.environ.get("MAPS3D"):
    from pulserver.recon.handlers.pics import nlinv_estimate
    def _deos(d):
        x_ = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(d, -1), axis=-1), -1)[..., nro // 2 - nx // 2: nro // 2 + nx // 2]
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x_, -1), axis=-1), -1)
    _k3 = np.zeros((VC, nz, ny, nx), np.complex64)
    for r in rows:
        if r["flags"] & (1 << 19): _k3[:, r["e2"], r["e1"]] = _deos(cc @ r["data"])
    _kw = {"sobolev": float(os.environ["SOBOLEV"])} if os.environ.get("SOBOLEV") else {}
    m3 = nlinv_estimate(torch.from_numpy(_k3), **_kw).to(torch.complex64)
    nm = m3[..., X0:X0 + 1].contiguous(); del m3, _k3
    coh = (nm.conj() * maps_r).sum(0).abs() / (nm.abs().pow(2).sum(0).sqrt() * maps_r.abs().pow(2).sum(0).sqrt() + 1e-12)
    print("3D nlinv maps; coherence with the true maps in head", round(float(coh[torch.from_numpy(np.abs(ser[0]) > 0.1 * np.abs(ser[0]).max())[..., None]].mean()), 4))
    maps_r = nm
if os.environ.get("DROPREF"):
    pos = pos.copy(); pos[isref] = -1; P = torch.from_numpy(pos)
    meas = meas.clone(); meas[:, torch.from_numpy(isref)] = 0
    Afull = linop.WaveSense(maps, (frames, nz, ny, 1), 1, positions=P, basis=torch.eye(frames, dtype=torch.complex64), centred=True, psf=psf)
    synth = Afull(torch.from_numpy(ser[..., None])); print("calibration trains left out of the solve")
A = linop.WaveSense(maps_r, (rank, nz, ny, 1), 1, positions=P, basis=basis, centred=True, psf=psf)
eig = float(optim.maxeigen(A))
K = [int(v) for v in os.environ.get("KEEP", "0,4,9,19,29,39").split(",")]
Tr = np.abs(ser_w[K]); msk = Tr.max(0) > 0.1 * Tr.max()
from scipy.ndimage import binary_erosion
msk = binary_erosion(msk, iterations=3)
class KPre(linop.LinearOperator):
    """(Q_k + eps I)^-1 at each phase-encode k: Q_k the Gram matrix of the basis at the echoes sampling k."""
    def __init__(self, eps):
        Q = torch.zeros(nz, ny, rank, rank, dtype=torch.complex64)
        for f in range(frames):
            for z_, y_ in pos[f][pos[f, :, 0] >= 0]:
                b = basis[:, f]; Q[z_, y_] += torch.outer(b.conj(), b)
        lam_ = torch.linalg.eigvalsh(Q).amax(-1)
        e = eps * float(lam_.max())
        self.M = torch.linalg.inv(Q + e * torch.eye(rank)).to(torch.complex64)
        self.ishape = self.oshape = (rank, nz, ny, 1)
        super().__init__()
    def _ft(self, x): return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, (1, 2)), dim=(1, 2), norm="ortho"), (1, 2))
    def _ift(self, x): return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, (1, 2)), dim=(1, 2), norm="ortho"), (1, 2))
    def forward(self, x, out=None):
        k = self._ft(x.reshape(self.ishape))
        return self._ift(torch.einsum("zyrs,szyx->rzyx", self.M, k))
    def adjoint(self, y, out=None):
        k = self._ft(y.reshape(self.ishape))
        return self._ift(torch.einsum("zysr,szyx->rzyx", self.M.conj(), k))
PRE = KPre(float(os.environ["PRECOND"])) if os.environ.get("PRECOND") else None
def run(data, tag):
    s = optim.data_scaling(data, A=A) or 1.0
    prior = priors.LocallyLowRank((-3, -2), lam, joint_axes=(0,), block=8)
    a = optim.FISTA(prior, maxiter=iters, step=0.95 / eig)(data / s, A) * s if lam > 0 else optim.CG(0.0, maxiter=iters, precond=PRE)(data / s, A) * s
    R = np.abs(torch.einsum("kf,kzyx->fzyx", basis[:, K], a).numpy()[..., 0])
    sc = (R[:, msk] * Tr[:, msk]).sum() / (R[:, msk] ** 2).sum()
    err = [round(float(np.linalg.norm((sc * R[j] - Tr[j])[msk]) / np.linalg.norm(Tr[j][msk])), 3) for j in range(len(K))]
    print(f"{tag:10s} r{rank} lam{lam} it{iters}: {err}"); return sc * R
# scale the synthetic data to the measured
g = (synth.numpy().conj() * meas.numpy()).sum() / (np.abs(synth.numpy()) ** 2).sum(); synth = synth * torch.tensor(g, dtype=torch.complex64)
B = basis_full[:, :rank].numpy(); full = ser.reshape(frames, -1); proj = (B @ np.linalg.pinv(B) @ full).reshape(ser.shape)
if os.environ.get("INSPAN"):
    synth = Afull(torch.from_numpy(proj[..., None].astype(np.complex64))); run(synth, "in-span")
    for r_ in (4, 6, 8, 12):
        if r_ <= basis_full.shape[1]:
            B_ = basis_full[:, :r_].numpy(); pr = (B_ @ np.linalg.pinv(B_) @ full).reshape(ser.shape)
            Pj_ = np.abs(pr[K]); print("rank", r_, "floor", [round(float(np.linalg.norm((Pj_[j] - Tr[j])[msk]) / np.linalg.norm(Tr[j][msk])), 3) for j in range(len(K))])
    sys.exit()
Rr = run(meas, "real"); Rs = run(synth, "synthetic")
# the ideal rank-r projection of the truth
B = basis_full[:, :rank].numpy(); full = ser.reshape(frames, -1); proj = (B @ np.linalg.pinv(B) @ full).reshape(ser.shape)
Pj = np.abs(proj[K]); sc = (Pj[:, msk] * Tr[:, msk]).sum() / (Pj[:, msk] ** 2).sum()
print("rank floor", [round(float(np.linalg.norm((sc * Pj[j] - Tr[j])[msk]) / np.linalg.norm(Tr[j][msk])), 3) for j in range(len(K))])
np.savez(f"{case}/split_x{X0}.npz", real=Rr, synth=Rs, truth=Tr, keep=np.array(K))
