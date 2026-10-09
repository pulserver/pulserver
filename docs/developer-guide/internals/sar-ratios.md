# SAR ratios against a reference pulse

How {func}`~pulserver.ir.sar_ratios` turns VOPs into the two numbers the IR
cache carries per subsequence, and what the interpreter does with them. The
concept is in {doc}`../../explanations/safety-checks`.

VOPs are used only for pTx: a multichannel transmit coil is connected, and
the file contains RF shim events or multichannel RF waveforms. The interpreter
then sends the coil's virtual observation points (VOPs). The vendor's SAR
routine is calibrated for a pulse played in the coil's default channel weights,
so pulserver relates every pulse of a subsequence to such a pulse, the
*reference pulse*: hard, 180° and 1 ms, in the default channel weights.

The energy a pulse deposits at VOP $v$ is
$\int \mathbf{b}(t)^H Q_v\, \mathbf{b}(t)\,dt$, with $\mathbf{b}$ the drive of
each transmit channel and $Q_v$ the VOP's matrix. In the head of body model $b$
it is the same integral with that model's head SAR matrix $G_b$. For each
repetition $w$ of a subsequence, {func}`~pulserver.ir.sar_ratios` computes

$$
r_{\mathrm{local}} = \frac{L_{\mathrm{head}}}{L_{\mathrm{local}}} \max_w
\frac{M \max_v E_{v,w}}{N_w \min_b E^{\mathrm{ref}}_{G_b}},
\qquad
r_{\mathrm{head}} = \max_w \max_b \frac{E_{G_b,w}}{N_w\, E^{\mathrm{ref}}_{G_b}},
$$

with $E_{v,w}$ and $E_{G_b,w}$ the energy of the repetition at VOP $v$ and in
the head of body model $b$, $N_w$ the number of pulses it plays,
$E^{\mathrm{ref}}_{G_b}$ the head energy of one reference pulse, $M$ the VOP
file's safety factor, and $L_{\mathrm{head}}$ and $L_{\mathrm{local}}$ the
scanner's head and local SAR limits, `vop_head_limit` and `vop_local_limit`.
The blocks before the first repetition and after the last are included, as
pypulseqpp's SAR check averages over them.

The interpreter gives the reference pulse the shortest time its calibration
allows, the time at which the reference's head SAR reaches $L_{\mathrm{head}}$.
This fixes the drive scale without a measurement of transmitted power: at that
scale, a pulse given $\max(r_{\mathrm{local}}, r_{\mathrm{head}})$ times the
reference pulse's time deposits at most $L_{\mathrm{local}}$ of peak local SAR
and $L_{\mathrm{head}}$ of head SAR. Two consequences are worth checking a
design against:

- a 1 ms hard pulse of 90° counts a quarter of the reference in both terms,
  and a repetition of reference pulses has a head term of 1;
- a scale common to every channel's drive and to the matrices cancels in both
  terms, but the relative channel gains do not.

The smallest reference energy over the body models sets the largest drive
scale, so the local term holds in every model; the head term is taken body
model by body model, since a subject's head SAR is that of one body. The bound
rests on the interpreter's head SAR for the reference not falling below the
true one, and on the VOPs and the safety factor bounding peak local SAR.

The cache carries the two terms of each subsequence in its
`pulseg_subseq_info`, zero without VOPs or without RF. The interpreter passes
each pulse of the subsequence to the vendor's SAR routine as a reference pulse
lasting the reference pulse's time times the larger of the two.
