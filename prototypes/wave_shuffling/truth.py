"""Ground-truth echo images of the virtual 7 T scan: truth.py CASE  (BrainWeb tissue, head8 B1+, head32 RSS receive, EPG of the interpreted description)."""

import dataclasses
import pickle
import sys
import time

import blochsim
import numpy as np
import torch
from basis import fse_interpretation
from blochsim import sequence as S

from pulserver.virtual import COILS, BrainWeb


def mprage_interpretation(d):
    ev = tuple(
        dataclasses.replace(e, action=S.EventAction.SPOIL_AFTER)
        if e.type == S.EventType.ADC
        else e
        for e in d.events
    )
    return dataclasses.replace(d, events=ev)


def megre_interpretation(d):
    ev = list(d.events)
    last = None
    for i, e in enumerate(ev):
        if e.type == S.EventType.RF and last is not None:
            ev[last] = dataclasses.replace(ev[last], action=S.EventAction.SPOIL_AFTER)
            last = None
        if e.type == S.EventType.ADC:
            last = i
    if last is not None:
        ev[last] = dataclasses.replace(ev[last], action=S.EventAction.SPOIL_AFTER)
    return dataclasses.replace(d, events=tuple(ev))


KINDS = {
    "fse": fse_interpretation,
    "mprage": mprage_interpretation,
    "megre": megre_interpretation,
}

case = sys.argv[1]
kind = sys.argv[2] if len(sys.argv) > 2 else "fse"
import os

N = tuple(int(v) for v in os.environ.get("MATRIX", "48,64,64").split(","))
FOV = tuple(
    float(v) for v in os.environ.get("FOVZYX", "0.192,0.24,0.24").split(",")
)  # (z, y, x)
coil = COILS["head8/head32"]
tissue = BrainWeb().tissue(1e-3, field_t=7.0, coil=coil)
b1 = (
    np.abs(tissue.transmit)
    if tissue.transmit is not None
    else np.ones(len(tissue.density))
)
rx = tissue.receive(tissue.positions)
rss = np.sqrt((np.abs(rx) ** 2).sum(-1)) if rx is not None else 1.0
print("entries", len(tissue.density), "B1 range", b1.min().round(2), b1.max().round(2))
desc = KINDS[kind](
    blochsim.description_from_waveforms(pickle.load(open(f"{case}/dump.pkl.wf", "rb")))[
        0
    ]
)
grid = np.linspace(0.0, 1.6, 81)
t2p = np.minimum(tissue.t2_prime, 1e3)
classes = np.unique(np.stack([tissue.t1, tissue.t2, t2p], 1), axis=0)
signal = np.zeros((len(tissue.density), 0), np.complex64)
t = time.perf_counter()
T1 = torch.tensor(np.repeat(classes[:, 0] * 1e3, grid.size), dtype=torch.float32)
T2 = torch.tensor(np.repeat(classes[:, 1] * 1e3, grid.size), dtype=torch.float32)
B = torch.tensor(np.tile(grid, len(classes)), dtype=torch.float32)
T2P = torch.tensor(np.repeat(classes[:, 2] * 1e3, grid.size), dtype=torch.float32)
D = S.simulate_subspace(
    desc,
    S.TissueProperties(t1_ms=T1, t2_ms=T2, b1=B, t2_prime_ms=T2P),
    rank=1,
    repetitions=400 if kind == "megre" else 3,
).dictionary.numpy()
D = D.reshape(len(classes), grid.size, -1)
print("dictionary", D.shape, f"{time.perf_counter() - t:.1f} s")
_, cls = np.unique(
    np.stack([tissue.t1, tissue.t2, t2p], 1), axis=0, return_inverse=True
)
cls = cls.reshape(-1)
pos = tissue.positions
w = tissue.density * rss
frames = D.shape[-1]
KEEP = (
    sorted({int(round(f * (frames - 1))) for f in (0.0, 0.1, 0.2, 0.45, 0.65, 0.9)})
    if kind == "mprage"
    else ([0, 4, 9, 19, 29, 39] if kind == "fse" else list(range(frames)))
)
F = tuple(int(round(f / 1e-3)) for f in FOV)  # fine 1 mm grid
echoes = np.zeros((len(KEEP), *F), np.float32)
idx = np.floor(
    np.stack(
        [
            pos[:, 2] / 1e-3 + F[0] / 2,
            pos[:, 1] / 1e-3 + F[1] / 2,
            pos[:, 0] / 1e-3 + F[2] / 2,
        ],
        1,
    )
).astype(int)
inside = np.all((idx >= 0) & (idx < np.array(F)), 1)
for c in range(len(classes)):
    m = inside & (cls == c)
    if not m.any():
        continue
    lin = np.ravel_multi_index(idx[m].T, F)
    for j, e in enumerate(KEEP):
        sig = np.interp(b1[m], grid, np.abs(D[c, :, e]))
        echoes[j].reshape(-1)[:] += np.bincount(
            lin, weights=w[m] * sig, minlength=np.prod(F)
        ).astype(np.float32)
out = np.zeros((len(KEEP), *N), np.float32)
for j in range(len(KEEP)):
    k = np.fft.fftshift(np.fft.fftn(echoes[j]))
    c0 = [f // 2 - n // 2 for f, n in zip(F, N)]
    k = k[c0[0] : c0[0] + N[0], c0[1] : c0[1] + N[1], c0[2] : c0[2] + N[2]]
    out[j] = np.abs(np.fft.ifftn(np.fft.ifftshift(k)))
np.savez(f"{case}/truth.npz", echoes=out, keep=np.array(KEEP))
print("saved", out.shape)
