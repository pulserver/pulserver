import sys, os, numpy as np
from scipy.ndimage import binary_erosion
T = np.load(sys.argv[1])["echoes"]; a, b = [int(v) for v in sys.argv[2].split(":")]
C = np.maximum(np.load("rss_helmet.npy"), 0.5)
T = (T / C)[..., a:b]
m = binary_erosion(T.max(0) > 0.1 * T.max(), iterations=3)
for f in sys.argv[3:]:
    R = (np.abs(np.load(f)["echoes"]) / C)[..., a:b]
    s = (R[:, m] * T[:, m]).sum() / (R[:, m] ** 2).sum()
    print(f"{f:34s}", [round(float(np.linalg.norm((s * R[j] - T[j])[m]) / np.linalg.norm(T[j][m])), 3) for j in range(len(T))])
