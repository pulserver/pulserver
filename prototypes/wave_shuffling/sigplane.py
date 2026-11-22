"""Full temporal signal series of one sagittal plane from the virtual scanner's own per-entry signals: sigplane.py CASE X0 OUT"""
import glob, os, sys, numpy as np, torch
from pulserver.virtual import BrainWeb, COILS
from pulserver.virtual import _fourier as F
case, X0, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seq = glob.glob(f"{case}/designs/*/*/sequence.seq")[0]
coil = COILS["head8/head32"]
tissue = BrainWeb().tissue(2e-3, field_t=7.0, coil=coil)
kept = {}
_orig = F.FourierPlayer._temporal
def keep(self, *a, **k):
    kept.update(entries=self._entries, group_of=self._group_of, atoms=self._atoms); return _orig(self, *a, **k)
F.FourierPlayer._temporal = keep
p = F.FourierPlayer(seq, tissue, device="cpu")
bs = max((b for b in p._bases if b is not None), key=lambda b: len(b.readouts))
E, G, A = kept["entries"], kept["group_of"], kept["atoms"]
local = bs.local
pos = E.positions[local].numpy()
x0 = (X0 - 64) * 2e-3
sel = np.abs(pos[:, 0] - x0 - 1e-3) <= 1e-3  # entries inside the 2 mm plane
local = local[torch.from_numpy(sel)]; pos = pos[sel]
cols = torch.searchsorted(bs.atoms, A.index[local]); rows = torch.searchsorted(bs.members, G[local])
T = bs.temporal[:, -int(os.environ.get("ETL", 128)):]
c = torch.einsum("njk,nj->nk", bs.coefficients[rows[:, None], cols], A.weight[local].to(bs.coefficients.dtype))
sig = (c @ T).numpy()  # (n, frames), complex: keeps the echo phases
dens = E.density[local].numpy()
rss = np.sqrt((np.abs(np.asarray(coil.receive(pos))) ** 2).sum(-1))
Fz, Fy = 120, 140
iz = np.floor(pos[:, 2] / 2e-3 + Fz / 2).astype(int); iy = np.floor(pos[:, 1] / 2e-3 + Fy / 2).astype(int)
ok = (iz >= 0) & (iz < Fz) & (iy >= 0) & (iy < Fy); lin = iz[ok] * Fy + iy[ok]
w = (dens * rss)[ok]
outv = np.zeros((sig.shape[1], 120, 140), np.complex64)
for f in range(sig.shape[1]):
    img = (np.bincount(lin, weights=(sig[ok, f] * w).real, minlength=Fz * Fy) + 1j * np.bincount(lin, weights=(sig[ok, f] * w).imag, minlength=Fz * Fy)).reshape(Fz, Fy)
    outv[f] = img
np.save(out, outv); print("saved", outv.shape, "entries", int(ok.sum()))
