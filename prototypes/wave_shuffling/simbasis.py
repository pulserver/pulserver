import pickle, time, numpy as np, torch
from pulserver.virtual import BrainWeb, COILS
from pulserver.virtual._fourier import FourierPlayer
seq = "h_ref/designs/20261121T202118-f58470/77efa4627462bcb1ea/sequence.seq"
coil = COILS["head8/head32"]
tissue = BrainWeb().tissue(2e-3, field_t=7.0, coil=coil)
t = time.perf_counter()
p = FourierPlayer(seq, tissue, device="cpu")
print("player", time.perf_counter() - t, "stations", len(p._bases))
out = []
for b in p._bases:
    if b is None: continue
    print("readouts", len(b.readouts), "coef", tuple(b.coefficients.shape), "temporal", tuple(b.temporal.shape), "column", tuple(b.column.shape), int(b.column.max()))
    out.append(dict(readouts=b.readouts, coefficients=b.coefficients.numpy(), temporal=b.temporal.numpy(), column=b.column.numpy()))
pickle.dump(out, open("h_ref/simbasis.pkl", "wb"))
