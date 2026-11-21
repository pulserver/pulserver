import numpy as np, math, sys
from scipy.ndimage import binary_erosion
from bartorch.linop._wave import _played, _kernel, _along, _support, _trajectory
z=np.load(sys.argv[1]); ref,data,k=z["ref"].astype(complex),z["wav"].astype(complex),z["k"]; fovs=tuple(z["fov"])
axes=2; grid=ref.shape[1:-1]
region=np.any(ref!=0,axis=(0,-1))
margin=[int(math.ceil(np.abs(k[:,a]).max()*fovs[a])) for a in range(axes)]
inner=binary_erosion(region,np.ones([2*m+1 for m in margin],bool),border_value=0)
fitted=inner&np.any(data!=0,axis=(0,-1))
lines_in=[_support(_along(region,a)) for a in range(axes)]; lines_out=[_support(_along(fitted,a)) for a in range(axes)]
source=ref[(slice(None),*lines_in)]; keep=fitted[tuple(lines_out)]; target=data[(slice(None),*lines_out)][:,keep]
def model(p):
    played=_played(k,p[:axes],p[axes:]); m=source
    for a in range(axes):
        K=_kernel(np.arange(grid[a])[lines_out[a]],np.arange(grid[a])[lines_in[a]],played[:,a]*fovs[a],grid[a])
        m=np.moveaxis(np.einsum("jit,...it->...jt",K,np.moveaxis(m,1+a,-2)),-2,1+a)
    return m[:,keep]
def cost(p): return np.linalg.norm(model(p)-target)/np.linalg.norm(target)
print("margin",margin,"fitted lines",fitted.sum())
for p in ([1,1,0,0],[0,0,0,0],[13.1,6.74,-1.289,4.449],[1,1,0.5,0.5],[1,1,-0.5,-0.5]): print(p, round(cost(np.array(p,float)),4))
# same-line check: flat vs wave of the same line, no model
print("flat-vs-wave same line", np.linalg.norm(ref[(slice(None),*lines_out)][:,keep]-target)/np.linalg.norm(target))
