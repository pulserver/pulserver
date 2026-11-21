"""Replay plugin that records every readout's labels, flags and trajectory after enrichment."""
import pickle, os
import numpy as np
from pulserver.recon.handlers.pics import PLUGIN as _P

OUT = os.environ.get("DUMP", "dump.pkl")
rows = []
_orig = type(_P).gadgets if hasattr(type(_P), "gadgets") else None

class Dump(type(_P)):
    def receive(self, acquisition, context):
        i = acquisition.idx
        rows.append(dict(flags=int(acquisition.flags), e1=i.kspace_encode_step_1, e2=i.kspace_encode_step_2,
                         seg=i.segment, con=i.contrast, rep=i.repetition, set=i.set, ph=i.phase,
                         user=list(i.user), uf=list(acquisition.user_float), space=acquisition.encoding_space_ref,
                         traj=np.array(acquisition.traj), data=np.array(acquisition.data), center=acquisition.center_sample, n=acquisition.number_of_samples))
        pickle.dump(rows, open(OUT, "wb")) if acquisition.flags & (1 << 24) else None
        return super().receive(acquisition, context)

PLUGIN = Dump()

_wf = []
def _rw(self, waveform):
    _wf.append((getattr(waveform, "waveform_id", None), np.array(waveform.data)))
    pickle.dump(_wf, open(OUT + ".wf", "wb"))
    return type(_P).receive_waveform(self, waveform)
Dump.receive_waveform = _rw
