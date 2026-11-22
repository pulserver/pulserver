"""Band-limited temporal series of one sagittal plane on the recon grid: sigplane_bl.py CASE X0 OUT [NX NY NZ FOVX FOVY FOVZ]

The phantom's entries sit on a 2 mm lattice offset from the recon grid; each echo image is the
object as a fully sampled scan sees it: entries transformed at their own positions, times the
2 mm cube's spectrum, inverse transformed onto the recon grid.
"""
import glob, os, sys, numpy as np, torch
from pulserver.virtual import BrainWeb, COILS
from pulserver.virtual import _fourier as F
case, X0, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
nx, ny, nz = (int(v) for v in os.environ.get("GRID", "128,140,120").split(","))
fx, fy, fz = (float(v) for v in os.environ.get("FOVS", "0.256,0.28,0.24").split(","))
seq = glob.glob(f"{case}/designs/*/*/sequence.seq")[0]
tissue = BrainWeb().tissue(2e-3, field_t=7.0, coil=COILS["head8/head32"])
kept = {}
_orig = F.FourierPlayer._temporal
def keep(self, *a, **k):
    kept.update(entries=self._entries, group_of=self._group_of, atoms=self._atoms); return _orig(self, *a, **k)
F.FourierPlayer._temporal = keep
p = F.FourierPlayer(seq, tissue, device="cpu")
bs = max((b for b in p._bases if b is not None), key=lambda b: len(b.readouts))
E, G, A = kept["entries"], kept["group_of"], kept["atoms"]
local = bs.local; pos = E.positions[local].numpy().astype(np.float64)
cols = torch.searchsorted(bs.atoms, A.index[local]); rows = torch.searchsorted(bs.members, G[local])
T = bs.temporal[:, -int(os.environ.get("ETL", 128)):].numpy()
c = torch.einsum("njk,nj->nk", bs.coefficients[rows[:, None], cols], A.weight[local].to(bs.coefficients.dtype)).numpy()
dens = E.density[local].numpy().astype(np.complex128)
if E.transmit is not None and not os.environ.get("NOTX"): dens = dens * np.exp(1j * np.angle(E.transmit[local].numpy()))
sp = 2e-3
ox, oy, oz = (float(v) * 1e-3 for v in os.environ.get("OFF", "0,0,0").split(","))
pos = pos - np.array([ox, oy, oz])  # voxel centres sit OFF past (j - n/2) * delta
# Readout: band-limited weight of each entry in the recon voxel X0 (the readout's k extent, cube spectrum included).
kx = (np.arange(nx) - nx // 2) / fx
x0 = (X0 - nx // 2) * fx / nx
wx = (np.exp(2j * np.pi * kx[None, :] * (x0 - pos[:, 0:1])) * np.sinc(kx * sp)[None]).sum(1) / nx
near = np.abs(wx) > 1e-4
print("entries weighted into the plane", int(near.sum()))
# Phase encodes: bin on the recon grid (floor), then shift each bin's content to its true position in k.
jy = np.floor(pos[:, 1] / (fy / ny) + ny / 2).astype(int); jz = np.floor(pos[:, 2] / (fz / nz) + nz / 2).astype(int)
dy = pos[:, 1] - (jy - ny / 2) * fy / ny; dz = pos[:, 2] - (jz - nz / 2) * fz / nz
print("sub-voxel offsets y", np.unique(np.round(dy * 1e3, 3))[:3], "z", np.unique(np.round(dz * 1e3, 3))[:3], "x frac", np.unique(np.round((pos[:, 0] / (fx / nx)) % 1, 3))[:3])
ok = near & (jy >= 0) & (jy < ny) & (jz >= 0) & (jz < nz)
assert np.ptp(dy[ok]) < 1e-6 and np.ptp(dz[ok]) < 1e-6
lin = jz[ok] * ny + jy[ok]; w = (dens * wx)[ok]; S = c[ok] @ T  # (n, frames)
ky = (np.arange(ny) - ny // 2) / fy; kz = (np.arange(nz) - nz // 2) / fz
H = (np.exp(-2j * np.pi * kz * dz[ok][0]) * np.sinc(kz * sp))[:, None] * (np.exp(-2j * np.pi * ky * dy[ok][0]) * np.sinc(ky * sp))[None, :]
ft = lambda a: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a, (-2, -1)), axes=(-2, -1)), (-2, -1))
ift = lambda a: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(a, (-2, -1)), axes=(-2, -1)), (-2, -1))
outv = np.zeros((S.shape[1], nz, ny), np.complex64)
for f in range(S.shape[1]):
    v = S[:, f] * w
    img = (np.bincount(lin, v.real, nz * ny) + 1j * np.bincount(lin, v.imag, nz * ny)).reshape(nz, ny)
    outv[f] = ift(ft(img) * H)
np.save(out, outv); print("saved", outv.shape)
