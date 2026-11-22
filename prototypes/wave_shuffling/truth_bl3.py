"""Registered, band-limited echo images on the recon grid: truth_bl3.py CASE KEEP OUT

Each image is the object a noise-free fully sampled scan of that echo sees: the entries' signals
(transmit phase included) weighted by the RSS of all receive coils, transformed at their own
positions, times the 2 mm cube's spectrum, on the scanner's voxel grid, whose centres sit at
(j - n/2 + 1/2) * delta.
"""
import glob, os, sys, numpy as np, torch
from pulserver.virtual import BrainWeb, COILS
from pulserver.virtual import _fourier as F
case, KEEP, out = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3]
N = np.array([120, 140, 128]); FOV = np.array([0.24, 0.28, 0.256]); sp = 2e-3; D = FOV / N
seq = glob.glob(f"{case}/designs/*/*/sequence.seq")[0]
coil = COILS["head8/head32"]
tissue = BrainWeb().tissue(sp, field_t=7.0, coil=coil)
kept = {}
_orig = F.FourierPlayer._temporal
def keep(self, *a, **k):
    kept.update(entries=self._entries, group_of=self._group_of, atoms=self._atoms); return _orig(self, *a, **k)
F.FourierPlayer._temporal = keep
p = F.FourierPlayer(seq, tissue, device="cpu")
bs = max((b for b in p._bases if b is not None), key=lambda b: len(b.readouts))
E, G, A = kept["entries"], kept["group_of"], kept["atoms"]
local = bs.local
cols = torch.searchsorted(bs.atoms, A.index[local]); rows = torch.searchsorted(bs.members, G[local])
T = bs.temporal[:, -int(os.environ.get("ETL", 128)):][:, KEEP]
sig = torch.einsum("njk,kf,nj->nf", bs.coefficients[rows[:, None], cols], T, A.weight[local].to(bs.coefficients.dtype)).numpy()
pos = E.positions[local].numpy().astype(np.float64)
w = E.density[local].numpy().astype(np.complex128)
if E.transmit is not None: w = w * np.exp(1j * np.angle(E.transmit[local].numpy()))
if not os.environ.get("NORSS"): w = w * np.sqrt((np.abs(np.asarray(coil.receive(pos))) ** 2).sum(-1))
zyx = pos[:, ::-1] - D / 2  # relative to voxel centres
j = np.floor(zyx / D + N / 2).astype(int); d = zyx - (j - N / 2) * D
inside = np.all((j >= 0) & (j < N), 1); d0 = d[inside][0]
assert np.abs(d[inside] - d0).max() < 1e-6, "entries not on one lattice"
print("sub-voxel offsets mm", np.round(d0 * 1e3, 3), "entries", int(inside.sum()))
lin = np.ravel_multi_index(j[inside].T, N)
k = [(np.arange(n) - n // 2) / f for n, f in zip(N, FOV)]
H = 1
for ax in range(3):
    h = np.exp(-2j * np.pi * k[ax] * d0[ax]) * np.sinc(k[ax] * sp)
    H = H * h.reshape([-1 if a == ax else 1 for a in range(3)])
ft = lambda a: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a)))
ift = lambda a: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(a)))
outv = np.zeros((len(KEEP), *N), np.complex64 if os.environ.get("COMPLEX") else np.float32)
for f in range(len(KEEP)):
    v = sig[inside, f] * w[inside]
    img = (np.bincount(lin, v.real, N.prod()) + 1j * np.bincount(lin, v.imag, N.prod())).reshape(N)
    outv[f] = ift(ft(img) * H) if os.environ.get("COMPLEX") else np.abs(ift(ft(img) * H))
np.savez(out, echoes=outv, keep=np.array(KEEP)); print("saved", outv.shape)
