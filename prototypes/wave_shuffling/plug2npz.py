"""plug2npz.py CASE: plugin images.h5 -> CASE/recon_plugin.npz with echoes (echo, z, y, x)."""
import sys, re, h5py, numpy as np, collections
c=sys.argv[1]; f=h5py.File(f"{c}/plug-out/images.h5","r"); h=f["dataset/images/header"][:]; d=f["dataset/images/data"][:,0,0]; at=f["dataset/images/attributes"][:]
echo=np.array([int(re.search(rb"<name>EchoNumber</name><value>(\d+)", a).group(1)) for a in at])
sl=h["slice"].astype(int)
print("per (echo) counts", collections.Counter(echo), "unique slices", len(set(sl)), "image_index range", h["image_index"].min(), h["image_index"].max())
E=echo.max(); Z=len(set(sl))
vol=np.zeros((E,Z)+d.shape[1:],np.float32); seen=np.zeros((E,Z),int)
for i in range(len(d)): vol[echo[i]-1, sl[i]]=d[i]; seen[echo[i]-1, sl[i]]+=1
print("max dup", seen.max())
np.savez(f"{c}/recon_plugin.npz", echoes=vol)
