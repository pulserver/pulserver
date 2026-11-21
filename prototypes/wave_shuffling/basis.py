"""On-the-fly FSE subspace basis from the datastream sequence description (prototype of the plugin's model)."""
import dataclasses
import numpy as np
import torch
import blochsim
from blochsim import sequence as S

CPMG = S.EventAction.CRUSH_BEFORE | S.EventAction.CRUSH_AFTER


def fse_interpretation(description):
    """Return the description with every refocusing pulse between crushers: one EPG shift each side."""
    events = tuple(
        dataclasses.replace(e, action=CPMG)
        if e.type == S.EventType.RF and e.params[1] == S.RfUse.REFOCUSING
        else e
        for e in description.events
    )
    return dataclasses.replace(description, events=events)


def fse_basis(waveforms, rank=12, b1=(0.25, 1.25, 21), slab=None):
    desc = fse_interpretation(blochsim.description_from_waveforms(waveforms)[0])
    t2 = torch.logspace(np.log10(20), np.log10(2000), 48)
    t1 = torch.tensor([400.0, 800.0, 1200.0, 1600.0, 2000.0, 2800.0, 4400.0])
    T1, T2, B1 = torch.meshgrid(t1, t2, torch.linspace(*b1), indexing="ij")
    tis = S.TissueProperties(t1_ms=T1.reshape(-1), t2_ms=T2.reshape(-1), b1=B1.reshape(-1))
    return S.simulate_subspace(desc, tis, rank=rank, repetitions=3, across_slice=slab)
