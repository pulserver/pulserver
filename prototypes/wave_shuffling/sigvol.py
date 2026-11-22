"""Full temporal signal series on a coarse 3D grid from the virtual scanner's own per-entry signals: sigvol.py CASE SPACING_MM OUT"""
import glob, os, sys, numpy as np, torch
from pulserver.virtual import BrainWeb, COILS
from pulserver.virtual import _fourier as F
case, sp, out = sys.argv[1], float(sys.argv[2]) * 1e-3, sys.argv[3]
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
local = bs.local; pos = E.positions[local].numpy()
cols = torch.searchsorted(bs.atoms, A.index[local]); rows = torch.searchsorted(bs.members, G[local])
T = bs.temporal[:, -int(os.environ.get("ETL", 128)):]
c = torch.einsum("njk,nj->nk", bs.coefficients[rows[:, None], cols], A.weight[local].to(bs.coefficients.dtype)).numpy()
dens = E.density[local].numpy()
shape = tuple(int(round(f / sp)) for f in (0.24, 0.28, 0.256))
idx = np.floor(np.stack([pos[:, 2], pos[:, 1], pos[:, 0]], 1) / sp + np.array(shape) / 2).astype(int)
ok = np.all((idx >= 0) & (idx < np.array(shape)), 1); lin = np.ravel_multi_index(idx[ok].T, shape)
w = dens[ok] / (sp / 2e-3) ** 3
outv = np.zeros((T.shape[1], *shape), np.complex64); Tn = T.numpy()
for f in range(T.shape[1]):
    s = (c[ok] @ Tn[:, f]) * w
    outv[f] = (np.bincount(lin, s.real, np.prod(shape)) + 1j * np.bincount(lin, s.imag, np.prod(shape))).reshape(shape)
np.save(out, outv); print("saved", outv.shape)
