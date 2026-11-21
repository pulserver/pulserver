import sys, numpy as np
import os
z=np.load(os.environ.get("TRUTH","h_ref")+"/truth.npz"); T=z["echoes"]; K=list(z["keep"])
from scipy.ndimage import binary_erosion
m = binary_erosion(T[0] > 0.1*T[0].max(), iterations=4)
def score(f):
    R=np.abs(np.load(f)["echoes"])[K]
    s=(R[:, m]*T[:, m]).sum()/(R[:, m]**2).sum()
    return [round(float(np.linalg.norm((s*R[j]-T[j])[m])/np.linalg.norm(T[j][m])),3) for j in range(len(K))]
if __name__ == "__main__":
    print("echoes", [k+1 for k in K])
    for f in sys.argv[1:]: print(f"{f:32s}", score(f))
