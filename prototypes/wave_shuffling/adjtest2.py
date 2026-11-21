import numpy as np
import torch
from bartorch import linop

exec(
    open("proto.py")
    .read()
    .split("kwargs = dict(")[0]
    .replace(
        "case, rank, lam, iters = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])",
        "case='fse'; rank=1",
    )
    .replace(
        'basis_full = torch.load(sys.argv[5]) if len(sys.argv) > 5 else torch.load(f"{case}/basis.pt")',
        "basis_full=torch.ones(45,1)/45**.5",
    )
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = linop.WaveSense(
    maps.to(torch.complex64),
    (1, nz, ny, nx),
    nro,
    positions=torch.from_numpy(pos),
    basis=basis,
    centred=True,
    psf=torch.ones(nz, ny, nro, dtype=torch.complex64),
)
x = A.H(torch.from_numpy(y))
print(A.ishape, A.oshape, x.shape)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(np.abs(x[0, 24].numpy()), cmap="gray")
ax[1].imshow(
    np.abs(maps[:, 24]).pow(2).sum(0).sqrt().numpy() if hasattr(maps, "pow") else 0,
    cmap="gray",
)
fig.savefig("fse/adj2.png", dpi=70)
