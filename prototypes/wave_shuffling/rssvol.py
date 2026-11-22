"""RSS of the 32 receive coils on the recon grid (intensity correction): rssvol.py OUT"""
import sys, numpy as np
from pulserver.virtual import COILS
nz, ny, nx = 120, 140, 128
zz, yy, xx = [(np.arange(n) - n // 2) * f / n for n, f in ((nz, .24), (ny, .28), (nx, .256))]
Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
s = np.asarray(COILS["head8/head32"].receive(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)))
np.save(sys.argv[1], np.sqrt((np.abs(s) ** 2).sum(1)).reshape(nz, ny, nx).astype(np.float32))
