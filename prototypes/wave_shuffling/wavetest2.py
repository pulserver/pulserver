"""Wave vs no wave for T2 shuffling at 4 mm with signal-equation data (helmet, SVD-compressed): wavetest.py"""
import os, time, numpy as np, torch
from bartorch import linop, optim, priors
from pulserver.virtual import COILS
from scipy.ndimage import binary_erosion
rng = np.random.default_rng(1)
ser = np.load("h_fse48/series4mm.npy").astype(np.complex64); T, nz, ny, nx = ser.shape; nro = 2 * nx
fov = (0.24, 0.28, 0.256); VC = int(os.environ.get("VC", 8))
zz, yy, xx = [(np.arange(n) - n // 2) * f / n for n, f in zip((nz, ny, nx), fov)]
Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
S = np.asarray(COILS["head8/head32"].receive(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))).T  # (32, n)
rss = np.sqrt((np.abs(S) ** 2).sum(0))
u = np.linalg.svd(S * (np.abs(ser[0]).ravel() > 0), full_matrices=False)[0]  # SVD compression over the object
M = ((u[:, :VC].conj().T @ S) / np.maximum(rss, 1e-6)).reshape(VC, nz, ny, nx).astype(np.complex64)
maps = torch.from_numpy(M)
# Shuffled sampling: R2 along y outside a 12x8 centre, every view once at a random echo.
views = [(z, y) for z in range(nz) for y in range(ny) if y % 2 == 0 or (abs(y - ny // 2) < 6 and abs(z - nz // 2) < 4)]
views = np.array(views)[rng.permutation(len(views))]
shots = -(-len(views) // T); pos = -np.ones((T, shots, 2), np.int64)
for i, v in enumerate(views): pos[i % T, i // T] = v
P = torch.from_numpy(pos)
B = torch.load("h_fse/basis_stream.pt").to(torch.complex64)
KEEP = [0, 4, 9, 19, 29, 39]
m = binary_erosion(np.abs(ser[KEEP]).max(0) > 0.1 * np.abs(ser[KEEP]).max(), iterations=2)

def psf(cycles, kfov):
    t = np.arange(nro) / nro
    ky = kfov / fov[1] * np.sin(2 * np.pi * cycles * t); kz = kfov / fov[0] * np.cos(2 * np.pi * cycles * t)
    return linop.wave_psf(np.stack([kz, ky], -1), (nz, ny, nx), (fov[0], fov[1]), centred=True).to(torch.complex64)

def run(label, ps, rank=3, lam=0.003, it=int(os.environ.get("IT", 40))):
    Af = linop.WaveSense(maps, (T, nz, ny, nx), nro, positions=P, basis=torch.eye(T, dtype=torch.complex64), centred=True, psf=ps)
    d = Af(torch.from_numpy(ser))
    basis = B[:, :rank].T.contiguous()
    A = linop.WaveSense(maps, (rank, nz, ny, nx), nro, positions=P, basis=basis, centred=True, psf=ps)
    s = optim.data_scaling(d, A=A) or 1.0; t0 = time.perf_counter()
    if lam == 0:
        a = optim.CG(1e-6, maxiter=it)(d / s, A) * s
    else:
        a = optim.FISTA(priors.LocallyLowRank((-3, -2, -1), lam, joint_axes=(0,), block=6), maxiter=it, step=0.95 / float(optim.maxeigen(A)))(d / s, A) * s
    R = np.abs(torch.einsum("kf,kzyx->fzyx", basis[:, KEEP], a).numpy()); Tr = np.abs(ser[KEEP])
    sc = (R[:, m] * Tr[:, m]).sum() / (R[:, m] ** 2).sum()
    err = [round(float(np.linalg.norm((sc * R[j] - Tr[j])[m]) / np.linalg.norm(Tr[j][m])), 3) for j in range(len(KEEP))]
    print(f"{label:28s} {time.perf_counter() - t0:6.1f} s", err, flush=True)
    np.save(f"h_fse48/wavetest_{label.replace(' ', '_')}.npy", sc * R)

ones = torch.ones(nz, ny, nro, dtype=torch.complex64)
print("views", len(views), "per echo", shots)
import ast
for c, k in ast.literal_eval(os.environ.get("WAVES", "[(6, 2.0), (6, 3.0), (10, 2.0)]")):
    run(f"wave {c}cyc k{k} lam0.0", psf(c, k), lam=0.0)
