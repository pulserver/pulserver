# Wave-shuffling prototype (work in progress)

Scratch harness for simulating wave-shuffled 3D FSE (and later MPRAGE, ME-GRE)
at 7 T on the virtual scanner and reconstructing it in a temporal subspace with
bartorch (`linop.WaveSense` with `basis=`, LLR, FISTA). Not part of the package.

- `scan.py NAME PLUGIN key=value ...`: virtual 7 T scan (BrainWeb, head8/head32).
- `plugins/*_wshfl.py`, `*_shfl.py`: zoo sequences loaded with wave/shuffling options.
- `dump.py`: replay plugin that records every enriched readout (labels, trajectory, data)
  and the sequence-description waveforms.
- `proto.py CASE RANK LAMBDA ITERS BASIS.pt [SIGN] [TAG] [flat]`: subspace recon prototype.

Status: runs end to end; the fully sampled shuffled reference still leaves an
11% residual that rank does not reduce, under investigation.
