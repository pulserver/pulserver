import pickle, numpy as np, torch, blochsim
from blochsim import sequence as S
from pulserver.virtual import BrainWeb, COILS
from pulserver.virtual._fourier import FourierPlayer
from basis import fse_interpretation
import glob
seq = glob.glob("g_shfl2/designs/*/*/sequence.seq")[0]
tissue = BrainWeb().tissue(2e-3, field_t=7.0, coil=COILS["head8/head32"])
_orig = FourierPlayer._temporal
def _keep(self, *a, **k):
    self._kept = self._atoms
    return _orig(self, *a, **k)
FourierPlayer._temporal = _keep
p = FourierPlayer(seq, tissue, device="cpu")
bs = [b for b in p._bases if b is not None][0]
h = p._kept; a = bs.atoms
t1, t2, f, b1 = (np.asarray(getattr(h, k)[a]) for k in ("t1", "t2", "frequency", "b1"))
print("atoms", len(a), "b1", np.round(np.percentile(b1, [0, 50, 100]), 2), "f", np.round(np.percentile(f, [0, 50, 100])))
C = bs.coefficients.numpy(); T = bs.temporal.numpy()
sim = np.einsum("gak,kt->gat", C, T[:, -128:])
gw = np.linalg.norm(sim, axis=2).sum(1); g = int(np.argmax(gw)); sim = sim[g]
desc = fse_interpretation(blochsim.description_from_waveforms(pickle.load(open("g_shfl2/dump.pkl.wf", "rb")))[0])
for b0 in (True, False):
    kw = dict(t1_ms=torch.tensor(t1 * 1e3, dtype=torch.float32), t2_ms=torch.tensor(t2 * 1e3, dtype=torch.float32), b1=torch.tensor(b1, dtype=torch.float32))
    if b0: kw["b0_hz"] = torch.tensor(-f, dtype=torch.float32)
    ours = S.simulate_subspace(desc, S.TissueProperties(**kw), rank=1, repetitions=3).dictionary.numpy()
    ours = ours.reshape(len(a), -1)[:, -128:]
    def nrm(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    A_, B_ = nrm(np.abs(sim)), nrm(np.abs(ours))
    d = np.linalg.norm(A_ - B_, axis=1)
    print("b0" if b0 else "no b0", "group", g, "atom |shape| diff median/max", np.round(np.median(d), 3), np.round(d.max(), 3))
    i = int(np.argmax(d)); print("  worst atom t1 t2 b1 f", t1[i], t2[i], b1[i], f[i]); print("  sim", np.round(A_[i, :8], 3), "\n  ours", np.round(B_[i, :8], 3))
