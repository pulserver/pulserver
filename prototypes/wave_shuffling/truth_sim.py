"""Ground truth from the virtual scanner's own per-entry signals (slab profiles of each pulse, B1, B0): truth_sim.py CASE SPACING_MM KEEP."""
import glob, os, sys, numpy as np, torch
from pulserver.virtual import BrainWeb, COILS
from pulserver.virtual import _fourier as F
case, spacing = sys.argv[1], float(sys.argv[2]); KEEP = [int(v) for v in sys.argv[3].split(",")]
seq = glob.glob(f"{case}/designs/*/*/sequence.seq")[0]
coil = COILS["head8/head32"]
tissue = BrainWeb().tissue(spacing * 1e-3, field_t=7.0, coil=coil)
kept = {}
_orig = F.FourierPlayer._temporal
def keep(self, *a, **k):
    kept.update(entries=self._entries, group_of=self._group_of, atoms=self._atoms)
    return _orig(self, *a, **k)
F.FourierPlayer._temporal = keep
p = F.FourierPlayer(seq, tissue, device="cpu")
bs = max((b for b in p._bases if b is not None), key=lambda b: len(b.readouts))
E, G, A = kept["entries"], kept["group_of"], kept["atoms"]
local = bs.local
cols = torch.searchsorted(bs.atoms, A.index[local])
rows = torch.searchsorted(bs.members, G[local])
T = bs.temporal[:, -int(os.environ.get("ETL", 128)):][:, KEEP]  # (rank, keep) last, settled train
coef = bs.coefficients[rows[:, None], cols]  # (n, 4, rank)
sig = torch.einsum("njk,kf,nj->nf", coef, T, A.weight[local].to(coef.dtype))
pos = E.positions[local].numpy(); dens = E.density[local].numpy()
rx = np.asarray(coil.receive(pos)) if hasattr(coil, "receive") else tissue.receive(pos)
rss = np.sqrt((np.abs(rx) ** 2).sum(-1))
N = tuple(int(v) for v in os.environ["MATRIX"].split(",")); FOV = tuple(float(v) for v in os.environ["FOVZYX"].split(","))
Fg = tuple(int(round(f / 1e-3)) for f in FOV)
idx = np.floor(np.stack([pos[:, 2] / 1e-3 + Fg[0] / 2, pos[:, 1] / 1e-3 + Fg[1] / 2, pos[:, 0] / 1e-3 + Fg[2] / 2], 1)).astype(int)
inside = np.all((idx >= 0) & (idx < np.array(Fg)), 1); lin = np.ravel_multi_index(idx[inside].T, Fg)
out = np.zeros((len(KEEP), *N), np.float32)
for j in range(len(KEEP)):
    s = np.abs(sig[:, j].numpy()) * dens * rss
    img = np.bincount(lin, weights=s[inside], minlength=np.prod(Fg)).reshape(Fg)
    k = np.fft.fftshift(np.fft.fftn(img)); c0 = [f // 2 - n // 2 for f, n in zip(Fg, N)]
    k = k[c0[0]:c0[0] + N[0], c0[1]:c0[1] + N[1], c0[2]:c0[2] + N[2]]
    out[j] = np.abs(np.fft.ifftn(np.fft.ifftshift(k)))
np.savez(f"{case}/truth_sim.npz", echoes=out, keep=np.array(KEEP)); print("saved", out.shape, "entries", len(local))
