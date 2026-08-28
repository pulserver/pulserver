# Background

Enough of the formats and tools Pulserver sits between to follow everything
else. These pages are a recap of published work, kept deliberately free of
Pulserver specifics — where Pulserver takes a different route, the
{doc}`sequence model <../sequence_model/index>` pages say so and why.

| Page | What it covers |
|---|---|
| {doc}`pulseq` | What a Pulseq file says, and the three things it leaves open. |
| {doc}`fov_transformation` | How a built sequence is moved to a prescription: rotation as an axis change, translation as a phase. |
| {doc}`pulseg` | What the PulSeg representation adds to it: definitions separated from instance parameters, base blocks, virtual segments. |
| {doc}`nimpulseqgui` | How Nimpulseq and Nimpulseqgui approach the same problems, and what they trade. |
| {doc}`ismrmrd` | What MRD carries back to the reconstruction: the header, the data messages, the session protocol. |

```{toctree}
:hidden:
:maxdepth: 1

pulseq
fov_transformation
pulseg
nimpulseqgui
ismrmrd
```
